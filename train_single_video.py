#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from tqdm import tqdm

from model.model import TDEEDModel
from util.dataset import load_classes


FPS = 25
STRIDE = 2
CLIP_LEN = 100


LABEL_MAP = {
    "pass": "PASS",
    "take_on": "DRIVE",
    "drive": "DRIVE",
    "clearance": "OUT",
    "ball_out_of_play": "OUT",
    "out": "OUT",
    "aerial_duel": "HEADER",
    "header": "HEADER",
    "block": "BALL PLAYER BLOCK",
    "shot": "SHOT",
    "tackle": "PLAYER SUCCESSFUL TACKLE",
    "cross": "CROSS",
    "throw_in": "THROW IN",
    "throw-in": "THROW IN",
    "free_kick": "FREE KICK",
    "free-kick": "FREE KICK",
    "goal": "GOAL",
    "high_pass": "HIGH PASS",
    "high-pass": "HIGH PASS",
}


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_frames(video_path, frames_dir):
    frames_dir = Path(frames_dir)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", "scale=-1:224",
        "-r", str(FPS),
        "-start_number", "0",
        str(frames_dir / "frame%d.jpg"),
    ])

    num_frames = len(list(frames_dir.glob("frame*.jpg")))
    print(f"Extracted frames: {num_frames}")
    return num_frames


def load_annotations(json_path, class_to_idx):
    with open(json_path, "r") as f:
        data = json.load(f)

    annotations = []

    for ann in data["annotations"]:
        raw_label = ann["label"].strip()
        raw_label_key = raw_label.lower().replace(" ", "_")

        if raw_label_key not in LABEL_MAP:
            print(f"Skipping unknown label: {raw_label}")
            continue

        mapped_label = LABEL_MAP[raw_label_key]

        if mapped_label not in class_to_idx:
            print(f"Skipping label not in class file: {mapped_label}")
            continue

        position_ms = int(float(ann["position"]))
        frame = round(position_ms / 1000.0 * FPS / STRIDE)

        team = ann.get("team", "left").lower()
        team_value = 0.0 if team == "left" else 1.0

        annotations.append({
            "frame": frame,
            "class_idx": class_to_idx[mapped_label],
            "team": team_value,
            "label": mapped_label,
            "position_ms": position_ms,
        })

    print(f"Loaded usable annotations: {len(annotations)}")
    return annotations


class SingleVideoTrainDataset(Dataset):
    def __init__(
        self,
        frames_dir,
        num_frames,
        annotations,
        num_classes,
        clip_len=100,
        stride=2,
        radius=4,
        samples_per_epoch=800,
    ):
        self.frames_dir = Path(frames_dir)
        self.num_raw_frames = num_frames
        self.num_frames = math.ceil(num_frames / stride)
        self.annotations = annotations
        self.num_classes = num_classes
        self.clip_len = clip_len
        self.stride = stride
        self.radius = radius
        self.samples_per_epoch = samples_per_epoch

        self.event_frames = [a["frame"] for a in annotations]

    def __len__(self):
        return self.samples_per_epoch

    def _read_frame(self, raw_frame_idx):
        raw_frame_idx = max(0, min(raw_frame_idx, self.num_raw_frames - 1))
        path = self.frames_dir / f"frame{raw_frame_idx}.jpg"

        if not path.exists():
            # Find nearest available frame
            path = self.frames_dir / f"frame{max(0, raw_frame_idx - 1)}.jpg"

        img = read_image(str(path)).float() / 255.0
        return img

    def __getitem__(self, idx):
        # 80% event-centered clips, 20% random clips
        if self.event_frames and np.random.rand() < 0.8:
            center = int(np.random.choice(self.event_frames))
            start = center - np.random.randint(10, self.clip_len - 10)
        else:
            start = np.random.randint(0, max(1, self.num_frames - self.clip_len))

        start = max(0, min(start, max(0, self.num_frames - self.clip_len)))
        end = start + self.clip_len

        frames = []
        for t in range(start, end):
            raw_idx = t * self.stride
            frames.append(self._read_frame(raw_idx))

        frames = torch.stack(frames, dim=0)

        # Model expects B,T,C,H,W from previous inference script style.
        labels = torch.zeros((self.clip_len, self.num_classes + 1), dtype=torch.float32)
        team_targets = torch.full((self.clip_len,), -1.0, dtype=torch.float32)

        # Background class at index 0 by convention.
        labels[:, 0] = 1.0

        for ann in self.annotations:
            rel = ann["frame"] - start

            if 0 <= rel < self.clip_len:
                class_index = ann["class_idx"] + 1

                lo = max(0, rel - self.radius)
                hi = min(self.clip_len, rel + self.radius + 1)

                labels[lo:hi, 0] = 0.0
                labels[lo:hi, class_index] = 1.0
                team_targets[lo:hi] = ann["team"]

        return {
            "frame": frames,
            "label": labels,
            "team": team_targets,
        }


def build_model(checkpoint_path, device):
    args = SimpleNamespace(
        modality="rgb",
        temporal_arch="ed_sgp_mixer",
        radi_displacement=4,
        feature_arch="rny002_gsf",
        event_team=True,
        clip_len=100,
        crop_dim=None,
        num_classes=12,
        n_layers=2,
        sgp_ks=9,
        sgp_r=4,
        joint_train={"num_classes": 17},
    )

    classes = load_classes("data/soccernetball/class.txt", event_team=True)
    joint_classes = load_classes("data/soccernet/class.txt", event_team=True)

    model = TDEEDModel(device=device, args=args)

    # Same head setup as inference script.
    n_classes = [len(classes) // 2 + 1, len(joint_classes) // 2 + 1]
    model._model.update_pred_head(n_classes)
    model._num_classes = np.array(n_classes).sum()

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load(ckpt)

    return model, classes


def get_trainable_module(model):
    # TDEEDModel is a wrapper. The actual torch module is usually model._model.
    if hasattr(model, "_model"):
        return model._model
    return model


def fine_tune(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    frames_dir = Path(args.work_dir) / "frames"
    num_frames = extract_frames(args.video, frames_dir)

    model_wrapper, classes = build_model(args.checkpoint, device)
    model = get_trainable_module(model_wrapper)
    model.train()

    # load_classes(..., event_team=True) returns a dict in this repo:
    # example: {"Pass-left": 0, "Pass-right": 1, ...}
    # We convert it into action-only names: Pass, Drive, Shot, etc.
    action_class_names = []
    
    for name in classes.keys():
        clean = name
    
        if clean.endswith("-left"):
            clean = clean[:-5]
        elif clean.endswith("-right"):
            clean = clean[:-6]
        elif clean.endswith(" left"):
            clean = clean[:-5]
        elif clean.endswith(" right"):
            clean = clean[:-6]
    
        if clean not in action_class_names:
            action_class_names.append(clean)
    
    action_classes = action_class_names
    class_to_idx = {c: i for i, c in enumerate(action_classes)}

    print("Classes:")
    for i, c in enumerate(action_classes):
        print(i, c)

    annotations = load_annotations(args.labels, class_to_idx)

    dataset = SingleVideoTrainDataset(
        frames_dir=frames_dir,
        num_frames=num_frames,
        annotations=annotations,
        num_classes=len(action_classes),
        clip_len=CLIP_LEN,
        stride=STRIDE,
        radius=args.radius,
        samples_per_epoch=args.samples_per_epoch,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_cls_loss = 0.0
        total_team_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            frames = batch["frame"].to(device)
            labels = batch["label"].to(device)
            team_targets = batch["team"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Try raw model forward.
            output = model(frames)

            # Repo model may return dict, tuple/list, or raw tensor.
            if isinstance(output, dict):
                print_once = not hasattr(fine_tune, "_printed_output_keys")
                if print_once:
                    print("Model output keys:", output.keys())
                    fine_tune._printed_output_keys = True
            
                # Common possible keys in this repo style
                if "pred" in output:
                    pred = output["pred"]
                elif "logits" in output:
                    pred = output["logits"]
                elif "spotting" in output:
                    pred = output["spotting"]
                elif "output" in output:
                    pred = output["output"]
                else:
                    # fallback: use first tensor value in dict
                    tensor_values = [v for v in output.values() if torch.is_tensor(v)]
                    if not tensor_values:
                        raise RuntimeError(f"No tensor found in model output dict: {output.keys()}")
                    pred = tensor_values[0]
            
            elif isinstance(output, (tuple, list)):
                pred = output[0]
            else:
                pred = output
            
            # Expected pred: B,T,C or B,C,T depending repo version.
            if pred.ndim == 3 and pred.shape[1] != CLIP_LEN and pred.shape[2] == CLIP_LEN:
                pred = pred.permute(0, 2, 1)

            pred_classes = pred[..., :labels.shape[-1]]

            cls_loss = F.binary_cross_entropy_with_logits(
                pred_classes,
                labels,
            )

            team_loss = torch.tensor(0.0, device=device)

            # If model outputs a team logit as last channel, use it.
            if pred.shape[-1] > labels.shape[-1]:
                team_logits = pred[..., labels.shape[-1]]
                mask = team_targets >= 0

                if mask.any():
                    team_loss = F.binary_cross_entropy_with_logits(
                        team_logits[mask],
                        team_targets[mask],
                    )

            loss = cls_loss + args.team_loss_weight * team_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_team_loss += team_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "team": f"{team_loss.item():.4f}",
            })

        avg_loss = total_loss / len(loader)
        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"cls={total_cls_loss / len(loader):.4f}, "
            f"team={total_team_loss / len(loader):.4f}"
        )

        save_path = Path(args.output_dir) / f"finetuned_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)
        print("Saved:", save_path)

    final_path = Path(args.output_dir) / "finetuned_best.pt"
    torch.save(model.state_dict(), final_path)
    print("Final saved:", final_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--work_dir", default="single_video_train")
    parser.add_argument("--output_dir", default="single_video_train/checkpoints")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--samples_per_epoch", type=int, default=800)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--team_loss_weight", type=float, default=0.5)

    args = parser.parse_args()
    fine_tune(args)


if __name__ == "__main__":
    main()
