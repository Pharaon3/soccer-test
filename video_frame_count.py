#!/usr/bin/env python3
"""
Report total frame count for a video file.

Uses ffprobe (same toolchain as subnet44_train_infer frame extraction).
Default uses container/metadata when available; use --exact to decode and count
(accurate for any file, slower).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8")
    return json.loads(out)


def _parse_rate(s: str | None) -> float | None:
    if not s or s == "0/0":
        return None
    return float(Fraction(s))


def frame_count_metadata(path: Path) -> tuple[int | None, str]:
    """
    Fast path: nb_frames from metadata, or estimate from duration * avg_frame_rate.
    Returns (count or None, note describing how count was obtained).
    """
    data = _ffprobe_json(path)
    streams = data.get("streams") or []
    if not streams:
        return None, "no video stream"
    s = streams[0]

    nb = s.get("nb_frames")
    if nb not in (None, "N/A", ""):
        try:
            n = int(nb)
            if n > 0:
                return n, "nb_frames (container/metadata)"
        except ValueError:
            pass

    duration = s.get("duration")
    avg_fps = _parse_rate(s.get("avg_frame_rate"))
    if duration is not None and avg_fps is not None:
        try:
            est = int(round(float(duration) * avg_fps))
            return est, f"estimated: duration * avg_frame_rate ({avg_fps} fps)"
        except ValueError:
            pass

    return None, "could not determine (try --exact)"


def frame_count_exact(path: Path) -> tuple[int, str]:
    """Decode and count frames (slow for long files, accurate)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8").strip()
    n = int(out)
    return n, "nb_read_frames (decoded count)"


def frame_count_opencv(path: Path) -> tuple[int, str]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if n <= 0:
        raise RuntimeError("OpenCV reported non-positive frame count")
    return n, "CAP_PROP_FRAME_COUNT (OpenCV; may be wrong for some codecs)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Decode full stream to count frames (accurate, slower; ffprobe only)",
    )
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Use OpenCV instead of ffprobe (no FFmpeg needed)",
    )
    args = parser.parse_args()

    path: Path = args.video
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    count: int | None = None
    note = ""

    if args.opencv:
        try:
            count, note = frame_count_opencv(path)
        except ImportError:
            print("Install opencv-python: pip install opencv-python", file=sys.stderr)
            return 3
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
    else:
        try:
            if args.exact:
                count, note = frame_count_exact(path)
            else:
                count, note = frame_count_metadata(path)
                if count is None:
                    print(note, file=sys.stderr)
                    print("Retrying with --exact ...", file=sys.stderr)
                    count, note = frame_count_exact(path)
        except FileNotFoundError:
            print("ffprobe not on PATH; trying OpenCV ...", file=sys.stderr)
            try:
                count, note = frame_count_opencv(path)
            except ImportError:
                print(
                    "Install FFmpeg (ffprobe) or: pip install opencv-python "
                    "and run with --opencv",
                    file=sys.stderr,
                )
                return 3
            except RuntimeError as e:
                print(str(e), file=sys.stderr)
                return 2
        except subprocess.CalledProcessError as e:
            print(f"ffprobe failed: {e}", file=sys.stderr)
            return 2

    print(count)
    print(f"# {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
