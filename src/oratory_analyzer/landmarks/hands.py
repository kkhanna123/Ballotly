"""MediaPipe Hands adapter.

Wraps ``mediapipe.solutions.hands`` behind :class:`HandExtractor`. Returns up to
``max_num_hands`` hands, each with 21 normalized landmarks and a handedness
label. Detailed finger landmarks enable richer gesture analysis than the few
hand points the Pose model provides.

Thin I/O edge — excluded from unit-test coverage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..domain.landmarks import HandLandmarks, Point3D
from .base import HandExtractor


class MediaPipeHandExtractor(HandExtractor):
    def __init__(
        self,
        *,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> Tuple[HandLandmarks, ...]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return ()

        handedness_labels = []
        if results.multi_handedness:
            for cls in results.multi_handedness:
                try:
                    handedness_labels.append(cls.classification[0].label)
                except (IndexError, AttributeError):
                    handedness_labels.append("Unknown")

        hands = []
        for i, hand in enumerate(results.multi_hand_landmarks):
            pts = tuple(
                Point3D(x=lm.x, y=lm.y, z=lm.z, visibility=1.0) for lm in hand.landmark
            )
            label = handedness_labels[i] if i < len(handedness_labels) else "Unknown"
            if label not in ("Left", "Right"):
                label = "Unknown"
            hands.append(HandLandmarks(points=pts, handedness=label))
        return tuple(hands)

    def close(self) -> None:
        self._hands.close()
