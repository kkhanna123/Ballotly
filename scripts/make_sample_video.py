"""Generate a small synthetic 'speaker' video for smoke-testing the video path.

NOTE: this draws a cartoon head/torso. MediaPipe's models are trained on real
imagery and generally will *not* detect landmarks on this stylized figure — it
exists to exercise video decode/encode/annotation plumbing and the
``--engine``-agnostic pipeline with deterministic data, not to validate the
MediaPipe models. For a real MediaPipe end-to-end run, point the CLI at footage
of an actual person (see README).

Usage:
    python scripts/make_sample_video.py out.mp4 --seconds 4 --fps 15
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np


def render(path: str, seconds: float, fps: int, width: int, height: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    n = int(seconds * fps)
    for i in range(n):
        t = i / fps
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        # Gentle sway to simulate a moving speaker.
        cx = int(width / 2 + 20 * math.sin(t * 1.5))
        cy = int(height * 0.4 + 8 * math.sin(t * 2.0))
        # Head
        cv2.circle(frame, (cx, cy), 70, (180, 170, 160), -1)
        # Eyes
        cv2.circle(frame, (cx - 25, cy - 10), 8, (40, 40, 40), -1)
        cv2.circle(frame, (cx + 25, cy - 10), 8, (40, 40, 40), -1)
        # Mouth (opens/closes)
        mouth_h = int(6 + 6 * abs(math.sin(t * 6.0)))
        cv2.ellipse(frame, (cx, cy + 35), (24, mouth_h), 0, 0, 360, (40, 40, 40), -1)
        # Shoulders / torso
        cv2.rectangle(frame, (cx - 90, cy + 90), (cx + 90, height), (90, 110, 150), -1)
        writer.write(frame)
    writer.release()
    print(f"Wrote {n} frames to {path} ({width}x{height} @ {fps}fps)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()
    render(args.output, args.seconds, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
