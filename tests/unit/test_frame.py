"""Tests for VideoMetadata."""

from __future__ import annotations

import pytest

from oratory_analyzer.domain.frame import VideoMetadata


def test_duration_and_aspect():
    meta = VideoMetadata(width=1920, height=1080, fps=30.0, frame_count=900)
    assert meta.duration_seconds == pytest.approx(30.0)
    assert meta.aspect_ratio == pytest.approx(16 / 9)


def test_timestamp_for():
    meta = VideoMetadata(width=640, height=480, fps=25.0, frame_count=100)
    assert meta.timestamp_for(50) == pytest.approx(2.0)


def test_timestamp_negative_rejected():
    meta = VideoMetadata(width=640, height=480, fps=25.0, frame_count=100)
    with pytest.raises(ValueError):
        meta.timestamp_for(-1)


@pytest.mark.parametrize(
    "w,h,fps,count",
    [(0, 480, 30, 10), (640, 0, 30, 10), (640, 480, 0, 10), (640, 480, 30, -1)],
)
def test_invalid_metadata(w, h, fps, count):
    with pytest.raises(ValueError):
        VideoMetadata(width=w, height=h, fps=fps, frame_count=count)
