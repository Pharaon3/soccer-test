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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.model import TDEEDModel
from util.dataset import load_classes
from util.eval import process_frame_predictions, soft_non_maximum_supression


FPS = 25
STRIDE_SNB = 2
WINDOW_SNB = 12
INFERENCE_BATCH_SIZE = 4


class SingleVideoDataset(Dataset):
    def __init__(self, video_name, frame_dir, num_frames, clip_len=100, stride=2, overlap_len=75):
        self.video_name = video_name
        self.frame_dir = Path(frame_dir)
        self.num_frames = num_frames
        self.clip_len = clip_len
        self.stride = stride
        self.overlap_len = overlap_len

        step = (clip_len - overlap_len) * stride
        self.clips = []
        for start in range(-5 * stride, max(1, num_frames - overlap_len * stride), step):
            self.clips.append(start)

    @property
    def videos(self):
        # mAP/eval utils expect: video name, number of frames after stride, fps after stride
        return [(self.video_name, math.ceil(self.num_frames / self.stride), FPS / self.stride)]

    def __len__(self):
        return len(self.clips)

    def _read_frame(self, idx):
        path = self.frame_dir / f"frame{idx}.jpg"
        if not path.exists():
            return None
        from torchvision.io import read_image
        return read_image(str(path))

    def __getitem__(self, idx):
        start = self.clips[idx]
        frames = []
        n_pad_start = 0
        n_pad_end = 0

        for frame_num in range(start, start + self.clip_len * self.stride, self.stride):
            if frame_num < 0:
                n_pad_start += 1
                continue

            img = self._read_frame(frame_num)
            if img is None:
                n_pad_end += 1
            else:
                frames.append(img)

        if len(frames) == 0:
            raise RuntimeError(f"No frames found for clip starting at {start}")

        frames = torch.stack(frames, dim=0)

        if n_pad_start > 0 or n_pad_end > 0:
            frames = torch.nn.functional.pad(
                frames,
                (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end)
            )

        return {
            "video": self.video_name,
            "start": start // self.stride,
            "frame": frames,
        }


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_video(video_url, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Video already exists: {output_path}")
        return

    run([
        "python", "-c",
        (
            "import urllib.request, sys; "
            "urllib.request.urlretrieve(sys.argv[1], sys.argv[2])"
        ),
        video_url,
        str(output_path),
    ])


def extract_frames(video_path, frames_dir):
    frames_dir = Path(frames_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: start_number 0 gives frame0.jpg ... frame749.jpg
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", "scale=-1:224",
        "-r", str(FPS),
        "-start_number", "0",
        str(frames_dir / "frame%d.jpg"),
    ])

    num_frames = len(list(frames_dir.glob("frame*.jpg")))
    print(f"Extracted {num_frames} frames")
    return num_frames


def load_baseline_model(checkpoint_path, device):
    # These are from config/SoccerNetBall/SoccerNetBall_baseline.json
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

    # The published baseline config uses joint training, so the checkpoint head is double-head.
    n_classes = [len(classes) // 2 + 1, len(joint_classes) // 2 + 1]
    model._model.update_pred_head(n_classes)
    model._num_classes = np.array(n_classes).sum()

    ckpt = torch.load(checkpoint_path, map_location=device)

    # checkpoint_best.pt from this repo is usually a raw state_dict.
    # This also handles checkpoints wrapped as {"state_dict": ...}.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load(ckpt)
    return model, classes


def predict_events(model, dataset, classes, output_json, threshold=0.01):
    pred_dict = {}

    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32),
        )

    loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    for clip in tqdm(loader, desc="Inference"):
        _, batch_pred_scores = model.predict(clip["frame"])

        for i in range(clip["frame"].shape[0]):
            video = clip["video"][i]
            scores, support = pred_dict[video]

            pred_scores = batch_pred_scores[i]
            start = clip["start"][i].item()

            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0

            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0).astype(np.int32)

    pred_events = process_frame_predictions(
        dataset,
        classes,
        pred_dict,
        threshold=threshold,
    )

    pred_events = soft_non_maximum_supression(
        pred_events,
        window=WINDOW_SNB,
        threshold=threshold,
    )

    # Convert frame index back to milliseconds because stride=2 was used.
    final = {
        "UrlLocal": dataset.video_name,
        "predictions": [],
    }

    for e in pred_events[0]["events"]:
        position_ms = int(e["frame"] / FPS * 1000 * STRIDE_SNB)
        final["predictions"].append({
            "gameTime": f"1 - {position_ms // 60000}:{int((position_ms % 60000) // 1000):02d}",
            "label": e["label"],
            "team": e.get("team", None),
            "position": position_ms,
            "time_sec": round(position_ms / 1000, 3),
            "confidence": float(e["score"]),
        })

    final["predictions"].sort(key=lambda x: x["position"])

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nSaved: {output_json}")
    print(f"Detected events: {len(final['predictions'])}\n")

    for e in final["predictions"]:
        print(
            f"{e['time_sec']:7.2f}s | "
            f"{e['label']:<28} | "
            f"team={e['team']:<5} | "
            f"conf={e['confidence']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_url",
        default="https://scoredata.me/chunks/8cfe43e516de4ee6bcb77e3716e5e6.mp4",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pt",
        help="Path to downloaded checkpoint_best.pt",
    )
    parser.add_argument("--work_dir", default="single_video_test")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--output", default="single_video_test/events.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    work_dir = Path(args.work_dir)
    video_path = work_dir / "input.mp4"
    frames_dir = work_dir / "frames" / "custom_video"

    download_video(args.video_url, video_path)
    num_frames = extract_frames(video_path, frames_dir)

    model, classes = load_baseline_model(args.checkpoint, device)

    dataset = SingleVideoDataset(
        video_name="custom_video",
        frame_dir=frames_dir,
        num_frames=num_frames,
        clip_len=100,
        stride=STRIDE_SNB,
        overlap_len=75,
    )

    predict_events(
        model=model,
        dataset=dataset,
        classes=classes,
        output_json=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
