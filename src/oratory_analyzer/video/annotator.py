"""Draws speaker box and landmarks onto frames for the annotated output video.

Thin I/O edge — uses OpenCV drawing primitives; excluded from coverage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..domain import Hands
from ..domain.landmarks import BoundingBox, FrameLandmarks

# A compact subset of pose connections (BlazePose) for a readable skeleton.
_POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms + shoulders
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (25, 27), (24, 26), (26, 28),             # legs
]


class FrameAnnotator:
    """Renders landmark overlays for a single speaker onto a BGR frame."""

    def __init__(
        self,
        *,
        face_color=(0, 200, 0),
        pose_color=(0, 160, 255),
        hand_color=(255, 80, 200),
        box_color=(0, 255, 255),
        draw_face_mesh: bool = True,
        draw_pose: bool = True,
        draw_hands: bool = True,
    ) -> None:
        self.face_color = face_color
        self.pose_color = pose_color
        self.hand_color = hand_color
        self.box_color = box_color
        self.draw_face_mesh = draw_face_mesh
        self.draw_pose = draw_pose
        self.draw_hands = draw_hands

    def annotate(self, frame_bgr: np.ndarray, fl: FrameLandmarks) -> np.ndarray:
        import cv2

        h, w = frame_bgr.shape[:2]
        out = frame_bgr

        if fl.face_box is not None:
            self._draw_box(cv2, out, fl.face_box, w, h, "speaker")

        if self.draw_face_mesh and fl.face is not None:
            for p in fl.face.points:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(out, (cx, cy), 1, self.face_color, -1)

        if self.draw_pose and fl.pose is not None:
            pts = fl.pose.points
            for a, b in _POSE_EDGES:
                if a < len(pts) and b < len(pts):
                    pa, pb = pts[a], pts[b]
                    if pa.visibility >= 0.4 and pb.visibility >= 0.4:
                        cv2.line(
                            out,
                            (int(pa.x * w), int(pa.y * h)),
                            (int(pb.x * w), int(pb.y * h)),
                            self.pose_color,
                            2,
                        )
            for p in pts:
                if p.visibility >= 0.4:
                    cv2.circle(out, (int(p.x * w), int(p.y * h)), 3, self.pose_color, -1)

        if self.draw_hands and fl.hands:
            for hand in fl.hands:
                pts = hand.points
                for a, b in Hands.CONNECTIONS:
                    pa, pb = pts[a], pts[b]
                    cv2.line(
                        out,
                        (int(pa.x * w), int(pa.y * h)),
                        (int(pb.x * w), int(pb.y * h)),
                        self.hand_color,
                        2,
                    )
                for p in pts:
                    cv2.circle(out, (int(p.x * w), int(p.y * h)), 3, self.hand_color, -1)

        return out

    def _draw_box(self, cv2, img, box: BoundingBox, w: int, h: int, label: str) -> None:
        x1, y1 = int(box.x_min * w), int(box.y_min * h)
        x2, y2 = int(box.x_max * w), int(box.y_max * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), self.box_color, 2)
        cv2.putText(
            img, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2, cv2.LINE_AA,
        )
