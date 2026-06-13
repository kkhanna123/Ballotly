"""Tests for hand landmarks, the hand extractor fake, and the hand metric."""

from __future__ import annotations

import numpy as np
import pytest

from oratory_analyzer.domain import Hands
from oratory_analyzer.domain.landmarks import FrameLandmarks, HandLandmarks, Point3D
from oratory_analyzer.landmarks import LandmarkPipeline, ScriptedHandExtractor
from oratory_analyzer.metrics import HandGestureMetric

from ..conftest import make_frame, make_hand

DUMMY = np.zeros((4, 4, 3), dtype=np.uint8)


class TestHandLandmarks:
    def test_basic(self):
        hand = make_hand()
        assert len(hand) == 21
        assert hand.handedness == "Right"
        assert isinstance(hand[Hands.WRIST], Point3D)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            HandLandmarks(points=())

    def test_invalid_handedness(self):
        with pytest.raises(ValueError):
            HandLandmarks(points=tuple(Point3D(0, 0) for _ in range(21)), handedness="Middle")

    def test_frame_hand_flags(self):
        frame = make_frame(with_hands=True)
        assert frame.has_hands and frame.num_hands == 1
        empty = FrameLandmarks(index=0, timestamp=0.0)
        assert not empty.has_hands and empty.num_hands == 0
        assert empty.is_empty


class TestScriptedHandExtractor:
    def test_replays_frames(self):
        ex = ScriptedHandExtractor([(make_hand(),), ()])
        assert len(ex.extract(DUMMY)) == 1
        assert ex.extract(DUMMY) == ()
        assert ex.extract(DUMMY) == ()  # exhausted

    def test_landmark_pipeline_includes_hands(self):
        pipe = LandmarkPipeline(None, None, ScriptedHandExtractor([(make_hand(), make_hand())]))
        frame = pipe.process(DUMMY, 0, 0.0)
        assert frame.num_hands == 2

    def test_pipeline_hands_only_allowed(self):
        # at least one extractor present (hands) => valid
        pipe = LandmarkPipeline(None, None, ScriptedHandExtractor([(make_hand(),)]))
        assert pipe.process(DUMMY, 0, 0.0).has_hands


class TestHandGestureMetric:
    def _clip(self, n, *, amplitude, openness=1.0):
        frames = []
        for i in range(n):
            cx = 0.5 + (amplitude if i % 2 else -amplitude)
            frames.append(make_frame(i, i * 0.1, hands=(make_hand(center=(cx, 0.6), openness=openness),)))
        return frames

    def test_no_hands_is_empty(self):
        frames = [make_frame(i, i * 0.1, with_hands=False) for i in range(5)]
        result = HandGestureMetric().compute(frames)
        assert result.coverage == 0.0
        assert result.score == 0.0

    def test_frozen_hands_score_low(self):
        result = HandGestureMetric().compute(self._clip(20, amplitude=0.0))
        assert result.score <= 5
        assert result.stats["gesture_speed_hw_per_s"] == pytest.approx(0.0)

    def test_moderate_motion_scores_high(self):
        # ~0.01 normalized move per 0.1s, scale 0.05 => ~2 hand-widths/s (in band)
        result = HandGestureMetric().compute(self._clip(20, amplitude=0.005))
        assert result.score >= 60

    def test_frantic_beats_down_score(self):
        moderate = HandGestureMetric().compute(self._clip(20, amplitude=0.005)).score
        frantic = HandGestureMetric().compute(self._clip(20, amplitude=0.08)).score
        assert frantic < moderate

    def test_openness_recorded(self):
        result = HandGestureMetric().compute(self._clip(10, amplitude=0.0, openness=1.0))
        assert result.stats["mean_openness"] > 0
        assert result.coverage == pytest.approx(1.0)
