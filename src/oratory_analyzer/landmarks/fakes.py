"""Scripted extractors used in tests and offline demos.

They ignore the pixels entirely and replay a predetermined list of landmark
objects, which makes pipeline/integration tests fully deterministic and free of
any ML dependency.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..domain.landmarks import FaceLandmarks, HandLandmarks, PoseLandmarks
from .base import FaceExtractor, HandExtractor, PoseExtractor


class ScriptedFaceExtractor(FaceExtractor):
    """Returns successive entries from ``frames`` on each call (``None`` allowed)."""

    def __init__(self, frames: Sequence[Optional[FaceLandmarks]]) -> None:
        self._frames: List[Optional[FaceLandmarks]] = list(frames)
        self._i = 0
        self.closed = False

    def extract(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        if self._i >= len(self._frames):
            return None
        result = self._frames[self._i]
        self._i += 1
        return result

    def close(self) -> None:
        self.closed = True


class ScriptedPoseExtractor(PoseExtractor):
    """Returns successive entries from ``frames`` on each call (``None`` allowed)."""

    def __init__(self, frames: Sequence[Optional[PoseLandmarks]]) -> None:
        self._frames: List[Optional[PoseLandmarks]] = list(frames)
        self._i = 0
        self.closed = False

    def extract(self, frame_bgr: np.ndarray) -> Optional[PoseLandmarks]:
        if self._i >= len(self._frames):
            return None
        result = self._frames[self._i]
        self._i += 1
        return result

    def close(self) -> None:
        self.closed = True


class ScriptedHandExtractor(HandExtractor):
    """Returns successive hand tuples from ``frames`` on each call."""

    def __init__(self, frames: Sequence[Sequence[HandLandmarks]]) -> None:
        self._frames: List[Tuple[HandLandmarks, ...]] = [tuple(f) for f in frames]
        self._i = 0
        self.closed = False

    def extract(self, frame_bgr: np.ndarray) -> Tuple[HandLandmarks, ...]:
        if self._i >= len(self._frames):
            return ()
        result = self._frames[self._i]
        self._i += 1
        return result

    def close(self) -> None:
        self.closed = True
