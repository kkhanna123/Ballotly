"""Behavioural tests for each oratory metric.

Each metric is fed synthetic 'good' and 'bad' clips built from the conftest
factories and asserted to score the good clip clearly higher than the bad one,
plus edge cases (no data, missing landmark group).
"""

from __future__ import annotations

from typing import List

import pytest

from oratory_analyzer.domain.landmarks import FrameLandmarks
from oratory_analyzer.metrics import (
    EyeContactMetric,
    FacialExpressivityMetric,
    GestureMetric,
    HeadStabilityMetric,
    MetricRegistry,
    PostureMetric,
)

from ..conftest import make_face, make_frame, make_pose


def _clip(frames: List[FrameLandmarks]) -> List[FrameLandmarks]:
    return frames


# --- Eye contact --------------------------------------------------------


class TestEyeContact:
    def test_forward_gaze_scores_high(self):
        frames = [make_frame(i, i * 0.1, face=make_face(yaw=0.0)) for i in range(20)]
        result = EyeContactMetric().compute(frames)
        assert result.score >= 95
        assert result.stats["forward_fraction"] == pytest.approx(1.0)

    def test_turned_away_scores_low(self):
        frames = [make_frame(i, i * 0.1, face=make_face(yaw=0.06)) for i in range(20)]
        result = EyeContactMetric().compute(frames)
        assert result.score <= 10

    def test_good_beats_bad(self):
        good = [make_frame(i, i * 0.1, face=make_face(yaw=0.0)) for i in range(20)]
        bad = [make_frame(i, i * 0.1, face=make_face(yaw=0.07)) for i in range(20)]
        assert EyeContactMetric().compute(good).score > EyeContactMetric().compute(bad).score

    def test_no_face_is_empty(self):
        frames = [make_frame(i, i * 0.1, with_face=False) for i in range(5)]
        result = EyeContactMetric().compute(frames)
        assert result.coverage == 0.0
        assert result.score == 0.0

    def test_coverage_reflects_missing_frames(self):
        frames = [make_frame(0, 0.0, face=make_face(yaw=0.0))]
        frames += [make_frame(i, i * 0.1, with_face=False) for i in range(1, 4)]
        result = EyeContactMetric().compute(frames)
        assert result.coverage == pytest.approx(0.25)


# --- Head stability -----------------------------------------------------


class TestHeadStability:
    def test_still_head_scores_high(self):
        frames = [make_frame(i, i * 0.1, face=make_face(center_x=0.5)) for i in range(20)]
        result = HeadStabilityMetric().compute(frames)
        assert result.score >= 95

    def test_swaying_head_scores_low(self):
        frames = []
        for i in range(20):
            cx = 0.5 + (0.08 if i % 2 else -0.08)  # big oscillation
            frames.append(make_frame(i, i * 0.1, face=make_face(center_x=cx)))
        result = HeadStabilityMetric().compute(frames)
        assert result.score <= 20

    def test_good_beats_bad(self):
        still = [make_frame(i, i * 0.1, face=make_face(center_x=0.5)) for i in range(20)]
        sway = [
            make_frame(i, i * 0.1, face=make_face(center_x=0.5 + (0.05 if i % 2 else -0.05)))
            for i in range(20)
        ]
        assert HeadStabilityMetric().compute(still).score > HeadStabilityMetric().compute(sway).score

    def test_insufficient_frames(self):
        result = HeadStabilityMetric().compute([make_frame(0, 0.0)])
        assert result.coverage == 0.0


# --- Posture ------------------------------------------------------------


class TestPosture:
    def test_upright_scores_high(self):
        frames = [make_frame(i, i * 0.1, pose=make_pose()) for i in range(15)]
        result = PostureMetric().compute(frames)
        assert result.score >= 85

    def test_tilted_and_leaning_scores_lower(self):
        good = [make_frame(i, i * 0.1, pose=make_pose()) for i in range(15)]
        bad = [
            make_frame(i, i * 0.1, pose=make_pose(shoulder_tilt=0.12, lean=0.12))
            for i in range(15)
        ]
        assert PostureMetric().compute(bad).score < PostureMetric().compute(good).score

    def test_no_pose_is_empty(self):
        frames = [make_frame(i, i * 0.1, with_pose=False) for i in range(5)]
        assert PostureMetric().compute(frames).coverage == 0.0


# --- Gestures -----------------------------------------------------------


class TestGestures:
    def test_frozen_hands_score_low(self):
        # identical wrist positions => zero speed => below ideal band
        frames = [
            make_frame(i, i * 0.1, pose=make_pose(left_wrist=(0.40, 0.68), right_wrist=(0.60, 0.68)))
            for i in range(20)
        ]
        result = GestureMetric().compute(frames)
        assert result.score <= 5
        assert result.stats["gesture_speed_sw_per_s"] == pytest.approx(0.0)

    def test_moderate_gestures_score_high(self):
        frames = []
        for i in range(20):
            # small oscillation ~0.02 normalized units per 0.1s => moderate speed
            off = 0.02 if i % 2 else 0.0
            frames.append(
                make_frame(
                    i, i * 0.1,
                    pose=make_pose(left_wrist=(0.40 + off, 0.55), right_wrist=(0.60 - off, 0.55)),
                )
            )
        result = GestureMetric().compute(frames)
        assert result.score >= 60

    def test_frantic_gestures_score_lower_than_moderate(self):
        def clip(amp):
            frames = []
            for i in range(20):
                off = amp if i % 2 else 0.0
                frames.append(
                    make_frame(
                        i, i * 0.1,
                        pose=make_pose(left_wrist=(0.40 + off, 0.55), right_wrist=(0.60, 0.55)),
                    )
                )
            return frames
        moderate = GestureMetric().compute(clip(0.02)).score
        frantic = GestureMetric().compute(clip(0.25)).score
        assert frantic < moderate


# --- Facial expressivity ------------------------------------------------


class TestFacialExpressivity:
    def test_flat_face_scores_low(self):
        frames = [make_frame(i, i * 0.1, face=make_face(mouth_open=0.0, brow_raise=0.0)) for i in range(20)]
        result = FacialExpressivityMetric().compute(frames)
        assert result.score <= 10

    def test_animated_face_scores_higher(self):
        frames = []
        for i in range(20):
            mo = 0.05 if i % 2 else 0.0
            br = 0.02 if i % 2 else 0.0
            frames.append(make_frame(i, i * 0.1, face=make_face(mouth_open=mo, brow_raise=br)))
        animated = FacialExpressivityMetric().compute(frames).score
        flat = FacialExpressivityMetric().compute(
            [make_frame(i, i * 0.1, face=make_face(mouth_open=0.0)) for i in range(20)]
        ).score
        assert animated > flat


# --- Registry -----------------------------------------------------------


class TestRegistry:
    def test_default_runs_all_when_both_present(self):
        frames = [make_frame(i, i * 0.1) for i in range(10)]
        results = MetricRegistry.default().evaluate(frames)
        assert set(results) == {
            "eye_contact",
            "head_stability",
            "facial_expressivity",
            "posture",
            "gestures",
        }

    def test_skips_pose_metrics_when_no_pose(self):
        frames = [make_frame(i, i * 0.1, with_pose=False) for i in range(10)]
        results = MetricRegistry.default().evaluate(frames)
        assert "posture" not in results and "gestures" not in results
        assert "eye_contact" in results

    def test_skips_face_metrics_when_no_face(self):
        frames = [make_frame(i, i * 0.1, with_face=False) for i in range(10)]
        results = MetricRegistry.default().evaluate(frames)
        assert "eye_contact" not in results
        assert "posture" in results

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError):
            MetricRegistry([EyeContactMetric(), EyeContactMetric()])
