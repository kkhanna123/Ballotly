"""Tests for PipelineConfig validation."""

from __future__ import annotations

import pytest

from oratory_analyzer.config import PipelineConfig


def test_defaults_valid():
    cfg = PipelineConfig()
    assert cfg.sample_fps == 12.0
    assert cfg.analyze_face and cfg.analyze_pose


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_fps": 0},
        {"sample_fps": -1},
        {"max_num_faces": 0},
        {"max_num_hands": 0},
        {"analyze_face": False, "analyze_pose": False, "analyze_hands": False},
    ],
)
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)
