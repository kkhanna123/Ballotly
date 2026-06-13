"""Abstract landmark-extraction interfaces.

The analytical core never depends on MediaPipe directly — it depends on these
interfaces. That makes the engine swappable (MediaPipe today; DeepLabCut,
Roboflow ``supervision``, or MMPose tomorrow) and lets tests inject scripted
fakes instead of running real models on real pixels.

Implementations receive a single video frame as an ``H x W x 3`` BGR ``ndarray``
(OpenCV's native layout) and return normalized landmarks, or ``None`` when
nothing is detected in that frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from typing import Tuple

from ..domain.landmarks import (
    BoundingBox,
    FaceLandmarks,
    FrameLandmarks,
    HandLandmarks,
    PoseLandmarks,
)


class FaceExtractor(ABC):
    """Extracts a single face's mesh landmarks from one frame."""

    @abstractmethod
    def extract(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        """Return face landmarks, or ``None`` if no face is detected."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying model resources."""


class PoseExtractor(ABC):
    """Extracts a single body's pose landmarks from one frame."""

    @abstractmethod
    def extract(self, frame_bgr: np.ndarray) -> Optional[PoseLandmarks]:
        """Return pose landmarks, or ``None`` if no body is detected."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying model resources."""


class HandExtractor(ABC):
    """Extracts all visible hands' landmarks from one frame."""

    @abstractmethod
    def extract(self, frame_bgr: np.ndarray) -> Tuple[HandLandmarks, ...]:
        """Return a tuple of detected hands (possibly empty)."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any underlying model resources."""


class LandmarkPipeline:
    """Combines a face and/or pose extractor into per-frame ``FrameLandmarks``.

    Either extractor may be omitted (``None``) to analyze only face or only
    posture. The face bounding box is derived from the detected face mesh and is
    used downstream for speaker selection and tracking.
    """

    def __init__(
        self,
        face_extractor: Optional[FaceExtractor] = None,
        pose_extractor: Optional[PoseExtractor] = None,
        hand_extractor: Optional[HandExtractor] = None,
        *,
        face_box_padding: float = 0.02,
    ) -> None:
        if face_extractor is None and pose_extractor is None and hand_extractor is None:
            raise ValueError("LandmarkPipeline needs at least one extractor")
        self._face = face_extractor
        self._pose = pose_extractor
        self._hands = hand_extractor
        self._padding = face_box_padding

    def process(
        self, frame_bgr: np.ndarray, frame_index: int, timestamp: float
    ) -> FrameLandmarks:
        face = self._face.extract(frame_bgr) if self._face else None
        pose = self._pose.extract(frame_bgr) if self._pose else None
        hands: Tuple[HandLandmarks, ...] = (
            tuple(self._hands.extract(frame_bgr)) if self._hands else ()
        )
        face_box: Optional[BoundingBox] = None
        if face is not None:
            face_box = BoundingBox.from_points(face.points, padding=self._padding)
        return FrameLandmarks(
            index=frame_index,
            timestamp=timestamp,
            face=face,
            pose=pose,
            hands=hands,
            face_box=face_box,
        )

    def close(self) -> None:
        if self._face:
            self._face.close()
        if self._pose:
            self._pose.close()
        if self._hands:
            self._hands.close()

    def __enter__(self) -> "LandmarkPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
