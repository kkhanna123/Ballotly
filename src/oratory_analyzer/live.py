"""Real-time webcam viewer that overlays tracked landmarks live.

Opens the laptop camera, runs the same MediaPipe extractors used by the offline
pipeline, and draws the face mesh + pose skeleton onto each frame in real time —
so you can watch the lines move on your face as you speak. A small HUD shows FPS,
detection status, and a few per-frame delivery cues (gaze + posture).

The pixel-pushing loop lives in :meth:`LiveTracker.run`; the per-frame logic is
isolated in :meth:`LiveTracker.process_frame` so it can be unit-tested with fake
extractors and a synthetic frame (no camera, no window).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .domain.landmarks import BoundingBox, FrameLandmarks
from .landmarks.base import FaceExtractor, PoseExtractor
from .metrics.eye_contact import _gaze_deviation
from .metrics.posture import _shoulder_tilt, _torso_lean
from .video.annotator import FrameAnnotator


class LiveTracker:
    """Drives a live camera preview with landmark overlays and a status HUD."""

    def __init__(
        self,
        face_extractor: Optional[FaceExtractor] = None,
        pose_extractor: Optional[PoseExtractor] = None,
        hand_extractor=None,
        *,
        annotator: Optional[FrameAnnotator] = None,
        mirror: bool = True,
        show_hud: bool = True,
        forward_threshold: float = 0.18,
    ) -> None:
        if face_extractor is None and pose_extractor is None and hand_extractor is None:
            raise ValueError("LiveTracker needs at least one extractor")
        self.face_extractor = face_extractor
        self.pose_extractor = pose_extractor
        self.hand_extractor = hand_extractor
        self.annotator = annotator or FrameAnnotator(
            draw_face_mesh=face_extractor is not None,
            draw_pose=pose_extractor is not None,
            draw_hands=hand_extractor is not None,
        )
        self.mirror = mirror
        self.show_hud = show_hud
        self.forward_threshold = forward_threshold
        self._fps_window: Deque[float] = deque(maxlen=30)

    @classmethod
    def with_mediapipe(
        cls,
        *,
        analyze_face: bool = True,
        analyze_pose: bool = True,
        analyze_hands: bool = True,
        **kwargs,
    ) -> "LiveTracker":
        from .landmarks import get_mediapipe_extractors

        # Single face in live mode = your face, the most prominent one.
        ex = get_mediapipe_extractors(
            extract_face=analyze_face,
            extract_pose=analyze_pose,
            extract_hands=analyze_hands,
            max_num_faces=1,
            max_num_hands=2,
        )
        return cls(ex.face, ex.pose, ex.hands, **kwargs)

    # --- per-frame logic (testable without a camera) -------------------

    def _frame_landmarks(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        face = self.face_extractor.extract(frame_bgr) if self.face_extractor else None
        pose = self.pose_extractor.extract(frame_bgr) if self.pose_extractor else None
        hands = (
            tuple(self.hand_extractor.extract(frame_bgr)) if self.hand_extractor else ()
        )
        face_box = (
            BoundingBox.from_points(face.points, padding=0.02) if face is not None else None
        )
        return FrameLandmarks(
            index=0, timestamp=0.0, face=face, pose=pose, hands=hands, face_box=face_box
        )

    def live_cues(self, fl: FrameLandmarks) -> Dict[str, str]:
        """Compute a few human-readable per-frame delivery cues for the HUD."""
        cues: Dict[str, str] = {}
        if fl.face is not None:
            dev = _gaze_deviation(fl.face)
            if dev is not None:
                cues["eye_contact"] = "on audience" if dev <= self.forward_threshold else "looking away"
        if fl.pose is not None:
            tilt = _shoulder_tilt(fl.pose)
            lean = _torso_lean(fl.pose)
            if tilt is not None:
                cues["shoulders"] = "level" if tilt <= 10 else f"tilted {tilt:.0f} deg"
            if lean is not None:
                cues["lean"] = "upright" if lean <= 10 else f"leaning {lean:.0f} deg"
        return cues

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
        """Return (annotated_frame, status) for one camera frame."""
        if self.mirror:
            import cv2

            frame_bgr = cv2.flip(frame_bgr, 1)
        fl = self._frame_landmarks(frame_bgr)
        annotated = self.annotator.annotate(frame_bgr, fl)
        status = {
            "face": "yes" if fl.has_face else "no",
            "pose": "yes" if fl.has_pose else "no",
            "hands": str(fl.num_hands),
        }
        status.update(self.live_cues(fl))
        if self.show_hud:
            self._draw_hud(annotated, status)
        return annotated, status

    def _draw_hud(self, img: np.ndarray, status: Dict[str, str]) -> None:
        import cv2

        fps = self._current_fps()
        lines = [
            f"FPS: {fps:4.1f}",
            f"Face: {status['face']}  Pose: {status['pose']}  Hands: {status.get('hands', '0')}",
        ]
        if "eye_contact" in status:
            lines.append(f"Eye contact: {status['eye_contact']}")
        if "shoulders" in status:
            lines.append(f"Shoulders: {status['shoulders']}")
        if "lean" in status:
            lines.append(f"Posture: {status['lean']}")

        # translucent backdrop for legibility
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (330, 22 + 22 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        for i, text in enumerate(lines):
            cv2.putText(
                img, text, (10, 26 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
            )
        cv2.putText(
            img, "press q to quit", (10, img.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

    def _tick(self, now: float) -> None:
        self._fps_window.append(now)

    def _current_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        if span <= 0:
            return 0.0
        return (len(self._fps_window) - 1) / span

    # --- camera loop ----------------------------------------------------

    def run(
        self,
        camera_index: int = 0,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        window_name: str = "Oratory Analyzer - live",
        clock=time.time,
    ) -> None:
        """Open the camera and display the annotated preview until 'q' is pressed."""
        import cv2

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}. "
                "Is another app using it, or is camera permission denied?"
            )
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self._tick(clock())
                annotated, _status = self.process_frame(frame)
                cv2.imshow(window_name, annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):  # q or Esc
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()

    def close(self) -> None:
        if self.face_extractor is not None:
            self.face_extractor.close()
        if self.pose_extractor is not None:
            self.pose_extractor.close()
        if self.hand_extractor is not None:
            self.hand_extractor.close()
