"""Tests for the live webcam tracker's per-frame logic (no camera, no window)."""

from __future__ import annotations

import numpy as np
import pytest

from oratory_analyzer.landmarks import ScriptedFaceExtractor, ScriptedPoseExtractor
from oratory_analyzer.live import LiveTracker

from ..conftest import make_face, make_pose

FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def _tracker(faces, poses, **kwargs):
    return LiveTracker(
        ScriptedFaceExtractor(faces) if faces is not None else None,
        ScriptedPoseExtractor(poses) if poses is not None else None,
        mirror=False,  # avoid cv2 dependency in the pure path
        **kwargs,
    )


def test_requires_an_extractor():
    with pytest.raises(ValueError):
        LiveTracker(None, None)


def test_process_frame_reports_detection_status():
    tracker = _tracker([make_face(yaw=0.0)], [make_pose()], show_hud=False)
    out, status = tracker.process_frame(FRAME)
    assert out.shape == FRAME.shape
    assert status["face"] == "yes"
    assert status["pose"] == "yes"


def test_process_frame_handles_no_detection():
    tracker = _tracker([None], [None], show_hud=False)
    _out, status = tracker.process_frame(FRAME)
    assert status["face"] == "no" and status["pose"] == "no"


def test_live_cue_eye_contact_forward():
    tracker = _tracker([make_face(yaw=0.0)], None, show_hud=False)
    _out, status = tracker.process_frame(FRAME)
    assert status["eye_contact"] == "on audience"


def test_live_cue_eye_contact_away():
    tracker = _tracker([make_face(yaw=0.08)], None, show_hud=False)
    _out, status = tracker.process_frame(FRAME)
    assert status["eye_contact"] == "looking away"


def test_live_cue_posture():
    tracker = _tracker(None, [make_pose(shoulder_tilt=0.0, lean=0.0)], show_hud=False)
    _out, status = tracker.process_frame(FRAME)
    assert status["shoulders"] == "level"
    assert status["lean"] == "upright"


def test_fps_starts_at_zero():
    tracker = _tracker([make_face()], None)
    assert tracker._current_fps() == 0.0


def test_fps_computes_from_ticks():
    tracker = _tracker([make_face()], None)
    tracker._tick(100.0)
    tracker._tick(101.0)
    tracker._tick(102.0)
    # 2 intervals over 2 seconds => 1 fps
    assert tracker._current_fps() == pytest.approx(1.0)


def test_close_releases_extractors():
    fe = ScriptedFaceExtractor([])
    pe = ScriptedPoseExtractor([])
    LiveTracker(fe, pe).close()
    assert fe.closed and pe.closed
