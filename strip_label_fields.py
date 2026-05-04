#!/usr/bin/env python3
"""
Remove gameTime, team, and visibility from label JSON files, and convert each
annotation's position (milliseconds) to a frame index.

Frame index matches subnet44_train_infer.parse_raw_annotations:
  raw_frame = round(int(float(position_ms)) / 1000 * FPS)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS_TO_REMOVE = frozenset({"gameTime", "team", "visibility"})


def process_tree(obj: object, fps: float) -> None:
    """Strip configured keys; convert annotation position (ms) -> frame."""
    if isinstance(obj, dict):
        for key in FIELDS_TO_REMOVE:
            obj.pop(key, None)

        if "label" in obj and "position" in obj:
            pos_ms = int(float(obj["position"]))
            obj["frame"] = round(pos_ms / 1000.0 * fps)
            del obj["position"]

        for value in obj.values():
            process_tree(value, fps)
    elif isinstance(obj, list):
        for item in obj:
            process_tree(item, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("clip-label.json"),
        help="Source JSON path (default: clip-label.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here instead of overwriting the input file",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frames per second for ms->frame conversion (default: 25, same as subnet44_train_infer)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output if args.output is not None else input_path

    text = input_path.read_text(encoding="utf-8")
    data = json.loads(text)
    process_tree(data, args.fps)

    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
