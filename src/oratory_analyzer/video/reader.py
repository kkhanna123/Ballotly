"""Video decoding via OpenCV.

Thin I/O edge — excluded from unit-test coverage; exercised by the video
round-trip integration test and the end-to-end run.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from ..domain.frame import VideoMetadata


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


class OpenCVVideoReader:
    """Iterates frames from a video file, with optional frame-rate subsampling.

    Parameters
    ----------
    path:
        Path to the video file.
    sample_fps:
        If set and lower than the source fps, frames are skipped so roughly
        ``sample_fps`` frames per second are yielded. This bounds processing
        cost on high-frame-rate footage without affecting timestamps.
    """

    def __init__(self, path: str, *, sample_fps: Optional[float] = None) -> None:
        import cv2

        self._cv2 = cv2
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise VideoReadError(f"Could not open video: {path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0  # fall back to a sane default for odd containers
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = max(frame_count, 0)

        self.metadata = VideoMetadata(
            width=width, height=height, fps=fps, frame_count=frame_count
        )
        if sample_fps and sample_fps < fps:
            self._stride = max(1, round(fps / sample_fps))
        else:
            self._stride = 1

    @property
    def stride(self) -> int:
        return self._stride

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield ``(frame_index, timestamp_seconds, frame_bgr)`` tuples."""
        cv2 = self._cv2
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if index % self._stride == 0:
                timestamp = index / self.metadata.fps
                yield index, timestamp, frame
            index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "OpenCVVideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
