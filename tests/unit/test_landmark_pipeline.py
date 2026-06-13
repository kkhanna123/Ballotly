"""Tests for the LandmarkPipeline orchestration and scripted fakes."""

from __future__ import annotations

import numpy as np
import pytest

from oratory_analyzer.landmarks import (
    LandmarkPipeline,
    ScriptedFaceExtractor,
    ScriptedPoseExtractor,
)

from ..conftest import make_face, make_pose

DUMMY = np.zeros((4, 4, 3), dtype=np.uint8)


def test_pipeline_combines_face_and_pose():
    face = make_face()
    pose = make_pose()
    pipe = LandmarkPipeline(
        ScriptedFaceExtractor([face]), ScriptedPoseExtractor([pose])
    )
    frame = pipe.process(DUMMY, frame_index=2, timestamp=0.5)
    assert frame.index == 2
    assert frame.timestamp == 0.5
    assert frame.has_face and frame.has_pose
    assert frame.face_box is not None  # derived from the face mesh


def test_pipeline_face_only():
    pipe = LandmarkPipeline(ScriptedFaceExtractor([make_face()]), None)
    frame = pipe.process(DUMMY, 0, 0.0)
    assert frame.has_face and not frame.has_pose


def test_pipeline_handles_dropped_detection():
    pipe = LandmarkPipeline(
        ScriptedFaceExtractor([None]), ScriptedPoseExtractor([None])
    )
    frame = pipe.process(DUMMY, 0, 0.0)
    assert frame.is_empty
    assert frame.face_box is None


def test_pipeline_requires_an_extractor():
    with pytest.raises(ValueError):
        LandmarkPipeline(None, None)


def test_pipeline_close_propagates():
    fe = ScriptedFaceExtractor([])
    pe = ScriptedPoseExtractor([])
    pipe = LandmarkPipeline(fe, pe)
    pipe.close()
    assert fe.closed and pe.closed


def test_pipeline_context_manager_closes():
    fe = ScriptedFaceExtractor([])
    with LandmarkPipeline(fe, None):
        pass
    assert fe.closed


def test_scripted_extractor_exhaustion_returns_none():
    fe = ScriptedFaceExtractor([make_face()])
    assert fe.extract(DUMMY) is not None
    assert fe.extract(DUMMY) is None  # exhausted
