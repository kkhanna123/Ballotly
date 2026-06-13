"""Tests for pure geometry helpers."""

from __future__ import annotations

import pytest

from oratory_analyzer.domain.geometry import (
    angle_between,
    centroid,
    clamp,
    euclidean,
    eye_aspect_ratio,
    horizontal_tilt_deg,
    line_angle_deg,
    mean,
    moving_average,
    percentile,
    stddev,
    vertical_lean_deg,
)
from oratory_analyzer.domain.landmarks import Point3D


def test_clamp():
    assert clamp(-1.0) == 0.0
    assert clamp(2.0) == 1.0
    assert clamp(0.5) == 0.5
    assert clamp(5.0, 0.0, 10.0) == 5.0


def test_euclidean():
    assert euclidean(Point3D(0, 0), Point3D(3, 4)) == pytest.approx(5.0)


class TestAngles:
    def test_right_angle(self):
        a = Point3D(1.0, 0.0)
        vertex = Point3D(0.0, 0.0)
        c = Point3D(0.0, 1.0)
        assert angle_between(a, vertex, c) == pytest.approx(90.0)

    def test_straight_angle(self):
        assert angle_between(Point3D(-1, 0), Point3D(0, 0), Point3D(1, 0)) == pytest.approx(180.0)

    def test_degenerate_angle_is_zero(self):
        assert angle_between(Point3D(0, 0), Point3D(0, 0), Point3D(1, 0)) == 0.0

    def test_line_angle_horizontal(self):
        assert line_angle_deg(Point3D(0, 0), Point3D(1, 0)) == pytest.approx(0.0)

    def test_line_angle_down_is_positive(self):
        # y grows downward, so a point below reads positive.
        assert line_angle_deg(Point3D(0, 0), Point3D(0, 1)) == pytest.approx(90.0)


class TestTiltAndLean:
    def test_level_shoulders(self):
        assert horizontal_tilt_deg(Point3D(0.4, 0.5), Point3D(0.6, 0.5)) == pytest.approx(0.0)

    def test_tilted_shoulders_right_lower(self):
        tilt = horizontal_tilt_deg(Point3D(0.4, 0.5), Point3D(0.6, 0.6))
        assert tilt > 0  # right point lower on screen

    def test_tilt_folds_to_small_angle(self):
        # near-horizontal regardless of left/right ordering stays small
        tilt = horizontal_tilt_deg(Point3D(0.6, 0.5), Point3D(0.4, 0.5))
        assert abs(tilt) < 1e-6

    def test_upright_lean_is_zero(self):
        top = Point3D(0.5, 0.3)
        bottom = Point3D(0.5, 0.7)
        assert vertical_lean_deg(top, bottom) == pytest.approx(0.0)

    def test_forward_lean_sign(self):
        # top shifted right of bottom => positive lean
        assert vertical_lean_deg(Point3D(0.6, 0.3), Point3D(0.5, 0.7)) > 0


def test_centroid():
    c = centroid([Point3D(0, 0), Point3D(2, 0), Point3D(1, 3)])
    assert (c.x, c.y) == pytest.approx((1.0, 1.0))


def test_centroid_empty():
    with pytest.raises(ValueError):
        centroid([])


class TestEAR:
    def test_open_eye(self):
        ear = eye_aspect_ratio(
            outer=Point3D(0.0, 0.0),
            inner=Point3D(0.1, 0.0),
            top=Point3D(0.05, -0.03),
            bottom=Point3D(0.05, 0.0),
        )
        assert ear == pytest.approx(0.3)

    def test_closed_eye(self):
        ear = eye_aspect_ratio(
            Point3D(0.0, 0.0), Point3D(0.1, 0.0), Point3D(0.05, 0.0), Point3D(0.05, 0.0)
        )
        assert ear == pytest.approx(0.0)

    def test_degenerate_width(self):
        ear = eye_aspect_ratio(
            Point3D(0.0, 0.0), Point3D(0.0, 0.0), Point3D(0.0, 0.1), Point3D(0.0, 0.0)
        )
        assert ear == 0.0


class TestSeriesStats:
    def test_moving_average_smooths(self):
        out = moving_average([0.0, 0.0, 3.0, 0.0, 0.0], window=3)
        assert len(out) == 5
        assert out[2] == pytest.approx(1.0)  # (0+3+0)/3

    def test_moving_average_window_one_is_identity(self):
        vals = [1.0, 2.0, 3.0]
        assert moving_average(vals, 1) == vals

    def test_moving_average_invalid_window(self):
        with pytest.raises(ValueError):
            moving_average([1.0], 0)

    def test_mean_empty_is_zero(self):
        assert mean([]) == 0.0

    def test_mean(self):
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_stddev(self):
        assert stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.0)

    def test_stddev_single_sample(self):
        assert stddev([5.0]) == 0.0

    def test_percentile(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert percentile(vals, 0) == pytest.approx(1.0)
        assert percentile(vals, 100) == pytest.approx(4.0)
        assert percentile(vals, 50) == pytest.approx(2.5)

    def test_percentile_empty(self):
        assert percentile([], 50) is None

    def test_percentile_out_of_range(self):
        with pytest.raises(ValueError):
            percentile([1.0], 150)
