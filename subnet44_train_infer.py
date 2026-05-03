#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tqdm import tqdm


FPS = 25
CLIP_SECONDS = 30
RAW_FRAMES = FPS * CLIP_SECONDS  # 750 frames
STRIDE = 5                       # 150 model frames per 30 sec
MODEL_FRAMES = RAW_FRAMES // STRIDE

SUBNET44_CLASSES = [
    "pass",
    "pass_received",
    "recovery",
    "tackle",
    "interception",
    "ball_out_of_play",
    "clearance",
    "take_on",
    "substitution",
    "block",
    "aerial_duel",
    "shot",
    "save",
    "foul",
    "goal",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(SUBNET44_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_raw_annotations(raw_list):
    anns = []
    for ann in raw_list:
        label = ann["label"].strip()

        if label not in CLASS_TO_IDX:
            print(f"Skipping non-subnet44 label: {label}")
            continue

        pos_ms = int(float(ann["position"]))
        raw_frame = round(pos_ms / 1000 * FPS)
        model_frame = raw_frame // STRIDE

        anns.append({
            "label": label,
            "class_idx": CLASS_TO_IDX[label],
            "model_frame": model_frame,
            "position": pos_ms,
            "team": ann.get("team", "unknown"),
            "visibility": ann.get("visibility", "visible"),
        })

    print(f"Loaded subnet44 annotations: {len(anns)}")

    if len(anns) == 0:
        raise RuntimeError("No valid subnet44 annotations found.")

    return anns


def load_annotations_from_labels_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError(
            f"Label file {path} has no top-level 'annotations'; "
            "use clip-label.json via --clip_label_json instead."
        )

    return parse_raw_annotations(data["annotations"])


def iter_clip_label_training_triples(json_path, videos_root):
    """
    Yields (absolute_video_path, raw_annotation_dicts, clip_work_key) for each
    entry in clip-label.json (sequential training, approach B).
    """
    json_path = Path(json_path)
    videos_root = Path(videos_root)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "videos" not in data:
        raise ValueError(f"{json_path} must contain a top-level 'videos' array.")

    for entry in data["videos"]:
        if entry.get("input_type") != "video":
            continue

        rel = entry["path"]
        video_path = videos_root / rel

        if not video_path.is_file():
            print(f"Skipping missing video: {video_path}")
            continue

        raw_anns = entry.get("annotations") or []
        parent = Path(rel).parent
        clip_key = parent.as_posix() if str(parent) != "." else Path(rel).stem
        clip_key = clip_key.replace("\\", "/").replace("/", "_")

        yield video_path.resolve(), raw_anns, clip_key


def extract_frames(video_path, frames_dir):
    frames_dir = Path(frames_dir)
    existing = list(frames_dir.glob("frame*.jpg"))

    if len(existing) > 100:
        print(f"Using existing frames: {len(existing)}")
        return len(existing)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "scale=398:224",
        "-r", str(FPS),
        "-start_number", "0",
        str(frames_dir / "frame%d.jpg"),
    ])

    count = len(list(frames_dir.glob("frame*.jpg")))
    print(f"Extracted frames: {count}")
    return count


class Subnet44BigModel(nn.Module):
    """
    From-zero model.
    Random weights.
    About 55M-65M parameters depending PyTorch implementation.
    Input:  B,T,C,H,W
    Output: B,T,15
    """

    def __init__(self, num_classes=15, d_model=768, layers=8, heads=8):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Linear(512, d_model)

        self.pos = nn.Parameter(torch.randn(1, MODEL_FRAMES, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        feat = self.cnn(x).flatten(1)
        feat = self.proj(feat)
        feat = feat.reshape(b, t, -1)

        feat = feat + self.pos[:, :t, :]
        feat = self.temporal(feat)

        return self.head(feat)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class Subnet44Dataset(Dataset):
    def __init__(
        self,
        video,
        labels,
        work_dir,
        samples_per_epoch=500,
        radius=1,
    ):
        self.video = video
        self.labels = labels
        self.work_dir = Path(work_dir)
        self.frames_dir = self.work_dir / "frames"
        self.samples_per_epoch = samples_per_epoch
        self.radius = radius

        self.num_frames = extract_frames(video, self.frames_dir)
        self.annotations = self.resolve_labels(labels)

    def resolve_labels(self, labels):
        if isinstance(labels, (list, tuple)):
            return parse_raw_annotations(labels)
        return load_annotations_from_labels_file(labels)

    def __len__(self):
        return self.samples_per_epoch

    def read_frame(self, raw_idx):
        raw_idx = max(0, min(raw_idx, self.num_frames - 1))
        path = self.frames_dir / f"frame{raw_idx}.jpg"
        return read_image(str(path)).float() / 255.0

    def __getitem__(self, idx):
        max_start = max(1, self.num_frames - RAW_FRAMES)

        if torch.rand(1).item() < 0.8:
            ann = self.annotations[torch.randint(0, len(self.annotations), (1,)).item()]
            center_raw = ann["model_frame"] * STRIDE
            start_raw = center_raw - RAW_FRAMES // 2
            start_raw = max(0, min(start_raw, max_start))
        else:
            start_raw = torch.randint(0, max_start, (1,)).item()

        frames = []

        for i in range(MODEL_FRAMES):
            raw_idx = start_raw + i * STRIDE
            frames.append(self.read_frame(raw_idx))

        frames = torch.stack(frames, dim=0)

        labels = torch.zeros((MODEL_FRAMES, len(SUBNET44_CLASSES)), dtype=torch.float32)

        start_model = start_raw // STRIDE
        end_model = start_model + MODEL_FRAMES

        for ann in self.annotations:
            pos = ann["model_frame"]

            if start_model <= pos < end_model:
                rel = pos - start_model
                lo = max(0, rel - self.radius)
                hi = min(MODEL_FRAMES, rel + self.radius + 1)
                labels[lo:hi, ann["class_idx"]] = 1.0

        return frames, labels


def _checkpoint_dict(model, args):
    return {
        "model_state": model.state_dict(),
        "classes": SUBNET44_CLASSES,
        "fps": FPS,
        "stride": STRIDE,
        "model_frames": MODEL_FRAMES,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
    }


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.clip_label_json:
        clip_triples = list(iter_clip_label_training_triples(
            args.clip_label_json,
            args.videos_dir,
        ))
        if not clip_triples:
            raise RuntimeError(
                "No trainable clips found (--clip_label_json / --videos_dir)."
            )
    else:
        clip_triples = [
            (Path(args.video).resolve(), args.labels, "single"),
        ]

    model = Subnet44BigModel(
        num_classes=len(SUBNET44_CLASSES),
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print("Resumed from:", args.resume)

    print(f"Model parameters: {count_params(model):,}")
    print(f"Model size in B params: {count_params(model) / 1e9:.4f}B")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    pos_weight = torch.ones(len(SUBNET44_CLASSES), device=device) * args.pos_weight

    out_dir = Path(args.output_dir)
    global_epoch = 0

    for clip_idx, (video_path, raw_anns, clip_key) in enumerate(clip_triples):
        print(
            f"\n=== Clip {clip_idx + 1}/{len(clip_triples)}: {clip_key} "
            f"({video_path}) ===\n"
        )

        clip_work = Path(args.work_dir) / clip_key
        dataset = Subnet44Dataset(
            video=str(video_path),
            labels=raw_anns,
            work_dir=clip_work,
            samples_per_epoch=args.samples_per_epoch,
            radius=args.radius,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        for epoch in range(args.epochs):
            global_epoch += 1
            model.train()
            total = 0.0

            pbar = tqdm(
                loader,
                desc=f"{clip_key} epoch {epoch + 1}/{args.epochs}",
            )

            for frames, labels in pbar:
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = model(frames)

                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                total += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            print(
                f"{clip_key} epoch {epoch + 1}: "
                f"loss={total / len(loader):.4f}"
            )

            torch.save(
                _checkpoint_dict(model, args),
                out_dir / f"{clip_key}_epoch_{epoch + 1}.pt",
            )
            torch.save(
                _checkpoint_dict(model, args),
                out_dir / f"global_epoch_{global_epoch}.pt",
            )

        torch.save(
            _checkpoint_dict(model, args),
            out_dir / f"{clip_key}_final.pt",
        )

    final = out_dir / "subnet44_big_from_zero.pt"
    torch.save(_checkpoint_dict(model, args), final)
    print("Saved:", final)


def local_max_events(probs, max_events_per_30s, min_gap_frames):
    """
    No threshold.
    Select strongest local peaks.
    """

    candidates = []

    t_len, c_len = probs.shape

    for t in range(1, t_len - 1):
        for c in range(c_len):
            score = float(probs[t, c])

            if score >= float(probs[t - 1, c]) and score >= float(probs[t + 1, c]):
                candidates.append((score, t, c))

    candidates.sort(reverse=True, key=lambda x: x[0])

    selected = []

    for score, t, c in candidates:
        too_close = False

        for _, old_t, old_c in selected:
            if old_c == c and abs(t - old_t) < min_gap_frames:
                too_close = True
                break

        if too_close:
            continue

        selected.append((score, t, c))

        if len(selected) >= max_events_per_30s:
            break

    selected.sort(key=lambda x: x[1])

    return selected


@torch.no_grad()
def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    model = Subnet44BigModel(
        num_classes=len(SUBNET44_CLASSES),
        d_model=ckpt.get("d_model", 768),
        layers=ckpt.get("layers", 8),
        heads=ckpt.get("heads", 8),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    frames_dir = Path(args.work_dir) / "frames"
    num_frames = extract_frames(args.video, frames_dir)

    annotations = []

    step_raw = RAW_FRAMES
    total_windows = math.ceil(num_frames / step_raw)

    min_gap_frames = max(1, int((args.min_gap_ms / 1000.0) * FPS / STRIDE))

    for win in tqdm(range(total_windows), desc="Infer"):
        start_raw = win * step_raw

        frames = []

        for i in range(MODEL_FRAMES):
            raw_idx = min(start_raw + i * STRIDE, num_frames - 1)
            img = read_image(str(frames_dir / f"frame{raw_idx}.jpg")).float() / 255.0
            frames.append(img)

        x = torch.stack(frames, dim=0).unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu()

        events = local_max_events(
            probs=probs,
            max_events_per_30s=args.max_events_per_30s,
            min_gap_frames=min_gap_frames,
        )

        for score, t, c in events:
            raw_frame = start_raw + t * STRIDE
            pos_ms = int(raw_frame / FPS * 1000)

            annotations.append({
                "gameTime": f"1 - {pos_ms // 60000:02d}:{(pos_ms % 60000) // 1000:02d}",
                "label": IDX_TO_CLASS[c],
                "position": str(pos_ms),
                "team": "unknown",
                "visibility": "visible",
                "confidence": round(float(score), 4),
            })

    annotations.sort(key=lambda x: int(x["position"]))

    result = {
        "UrlLocal": str(args.video),
        "UrlYoutube": "",
        "annotations": annotations,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved:", args.output_json)
    print("Events:", len(annotations))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--video", default=None)
    p_train.add_argument("--labels", default=None)
    p_train.add_argument(
        "--clip_label_json",
        default=None,
        help="clip-label.json: train each video clip sequentially (same model).",
    )
    p_train.add_argument(
        "--videos_dir",
        default="videos",
        help="Root folder for paths in clip-label.json (default: videos).",
    )
    p_train.add_argument("--work_dir", default="subnet44_work")
    p_train.add_argument("--output_dir", default="subnet44_checkpoints")

    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch_size", type=int, default=1)
    p_train.add_argument("--samples_per_epoch", type=int, default=500)
    p_train.add_argument("--num_workers", type=int, default=2)

    p_train.add_argument("--lr", type=float, default=3e-5)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--pos_weight", type=float, default=20.0)
    p_train.add_argument("--radius", type=int, default=1)
    p_train.add_argument("--grad_clip", type=float, default=1.0)

    p_train.add_argument("--d_model", type=int, default=768)
    p_train.add_argument("--layers", type=int, default=8)
    p_train.add_argument("--heads", type=int, default=8)

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--video", required=True)
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--output_json", default="subnet44_predictions.json")
    p_infer.add_argument("--work_dir", default="subnet44_infer")

    # Not threshold.
    # This controls how many peak events to export.
    p_infer.add_argument("--max_events_per_30s", type=int, default=25)
    p_infer.add_argument("--min_gap_ms", type=int, default=400)

    p_train.add_argument("--resume", default=None)

    args = parser.parse_args()

    if args.mode == "train":
        if args.clip_label_json:
            if args.video is not None or args.labels is not None:
                parser.error(
                    "With --clip_label_json, omit --video and --labels."
                )
        else:
            if not args.video or not args.labels:
                parser.error(
                    "Provide --clip_label_json or both --video and --labels."
                )
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
