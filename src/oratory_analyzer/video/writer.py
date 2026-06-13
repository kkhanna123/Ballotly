"""Annotated-video encoding via OpenCV. Thin I/O edge (no coverage target)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class OpenCVVideoWriter:
    """Writes BGR frames to an mp4 file."""

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        import cv2

        self._cv2 = cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {path}")
        self.path = path
        self.width = width
        self.height = height

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "OpenCVVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
