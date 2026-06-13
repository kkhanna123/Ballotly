"""Tests for the metric framework and scoring helpers."""

from __future__ import annotations

import pytest

from oratory_analyzer.metrics.base import (
    MetricResult,
    band_score,
    fraction_score,
    tolerance_score,
)


class TestMetricResult:
    def test_score_clamped_and_rounded(self):
        r = MetricResult(name="x", score=150.0, summary="", category="face")
        assert r.score == 100.0
        r2 = MetricResult(name="x", score=-5.0, summary="", category="face")
        assert r2.score == 0.0

    def test_coverage_clamped(self):
        r = MetricResult(name="x", score=50, summary="", category="face", coverage=2.0)
        assert r.coverage == 1.0


class TestFractionScore:
    def test_bounds(self):
        assert fraction_score(0.0) == 0.0
        assert fraction_score(1.0) == 100.0
        assert fraction_score(0.5) == 50.0

    def test_clamped(self):
        assert fraction_score(2.0) == 100.0
        assert fraction_score(-1.0) == 0.0


class TestBandScore:
    def test_inside_band_is_100(self):
        assert band_score(5.0, 4.0, 6.0, 0.0, 10.0) == 100.0

    def test_at_hard_bounds_is_0(self):
        assert band_score(0.0, 4.0, 6.0, 0.0, 10.0) == 0.0
        assert band_score(10.0, 4.0, 6.0, 0.0, 10.0) == 0.0

    def test_linear_below(self):
        # halfway between hard_low(0) and ideal_low(4) => 50
        assert band_score(2.0, 4.0, 6.0, 0.0, 10.0) == pytest.approx(50.0)

    def test_linear_above(self):
        # halfway between ideal_high(6) and hard_high(10) => 50
        assert band_score(8.0, 4.0, 6.0, 0.0, 10.0) == pytest.approx(50.0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            band_score(5.0, 4.0, 6.0, 5.0, 10.0)  # hard_low > ideal_low


class TestToleranceScore:
    def test_exact_is_100(self):
        assert tolerance_score(5.0, 5.0, 2.0) == 100.0

    def test_at_tolerance_is_0(self):
        assert tolerance_score(7.0, 5.0, 2.0) == pytest.approx(0.0)
        assert tolerance_score(3.0, 5.0, 2.0) == pytest.approx(0.0)

    def test_halfway(self):
        assert tolerance_score(6.0, 5.0, 2.0) == pytest.approx(50.0)

    def test_beyond_tolerance_clamped(self):
        assert tolerance_score(100.0, 5.0, 2.0) == 0.0

    def test_invalid_tolerance(self):
        with pytest.raises(ValueError):
            tolerance_score(5.0, 5.0, 0.0)
