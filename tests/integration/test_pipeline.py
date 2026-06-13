"""End-to-end pipeline tests using scripted fake extractors (no ML, no video).

These exercise the full chain: extraction -> speaker selection -> metrics ->
aggregation -> report bundle, deterministically.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from oratory_analyzer.config import PipelineConfig
from oratory_analyzer.domain.frame import VideoMetadata
from oratory_analyzer.landmarks import (
    ScriptedFaceExtractor,
    ScriptedHandExtractor,
    ScriptedPoseExtractor,
)
from oratory_analyzer.pipeline import OratoryAnalysisPipeline

from ..conftest import make_face, make_hand, make_pose

DUMMY = np.zeros((8, 8, 3), dtype=np.uint8)


def _source(n):
    return [(i, i / 10.0, DUMMY) for i in range(n)]


def _good_pipeline(n=30):
    faces = [make_face(yaw=0.0, mouth_open=(0.04 if i % 2 else 0.0)) for i in range(n)]
    poses = [make_pose() for _ in range(n)]
    return OratoryAnalysisPipeline(
        ScriptedFaceExtractor(faces),
        ScriptedPoseExtractor(poses),
        config=PipelineConfig(output_dir="unused"),
    )


class TestPipelineAnalysis:
    def test_runs_all_metrics(self):
        result = _good_pipeline(30).analyze_frames_source(_source(30))
        a = result.assessment
        assert set(a.metrics) == {
            "eye_contact", "head_stability", "facial_expressivity", "posture", "gestures",
        }
        assert a.frames_analyzed == 30
        assert 0 <= a.overall_score <= 100

    def test_good_speaker_scores_well(self):
        result = _good_pipeline(30).analyze_frames_source(_source(30))
        # forward gaze + still head + upright posture should score strongly.
        assert result.assessment.metrics["eye_contact"].score >= 90
        assert result.assessment.metrics["posture"].score >= 80

    def test_frames_with_speaker_counted(self):
        result = _good_pipeline(20).analyze_frames_source(_source(20))
        assert result.assessment.frames_with_speaker == 20

    def test_hands_flow_through_to_metric(self):
        n = 20
        faces = [make_face() for _ in range(n)]
        poses = [make_pose() for _ in range(n)]
        # moving hands so the metric registers gesturing
        hand_frames = [
            (make_hand(center=(0.5 + (0.005 if i % 2 else -0.005), 0.6)),) for i in range(n)
        ]
        pipe = OratoryAnalysisPipeline(
            ScriptedFaceExtractor(faces),
            ScriptedPoseExtractor(poses),
            ScriptedHandExtractor(hand_frames),
        )
        result = pipe.analyze_frames_source(_source(n))
        assert "hand_gestures" in result.assessment.metrics
        assert result.frames[0].num_hands == 1

    def test_hands_only_pipeline(self):
        n = 12
        hand_frames = [(make_hand(),) for _ in range(n)]
        pipe = OratoryAnalysisPipeline(None, None, ScriptedHandExtractor(hand_frames))
        result = pipe.analyze_frames_source(_source(n))
        assert set(result.assessment.metrics) <= {"hand_gestures"}
        assert result.assessment.frames_with_speaker == n

    def test_face_only_skips_pose_metrics(self):
        faces = [make_face() for _ in range(10)]
        pipe = OratoryAnalysisPipeline(ScriptedFaceExtractor(faces), None)
        result = pipe.analyze_frames_source(_source(10))
        assert "posture" not in result.assessment.metrics
        assert "eye_contact" in result.assessment.metrics

    def test_empty_source_raises(self):
        with pytest.raises(ValueError):
            _good_pipeline().analyze_frames_source([])

    def test_no_detections_raises(self):
        pipe = OratoryAnalysisPipeline(
            ScriptedFaceExtractor([None] * 5), ScriptedPoseExtractor([None] * 5)
        )
        with pytest.raises(ValueError, match="No metrics"):
            pipe.analyze_frames_source(_source(5))

    def test_requires_an_extractor(self):
        with pytest.raises(ValueError):
            OratoryAnalysisPipeline(None, None)

    def test_video_meta_threads_through(self):
        meta = VideoMetadata(width=640, height=480, fps=30, frame_count=300)
        result = _good_pipeline(30).analyze_frames_source(_source(30), video_meta=meta)
        assert result.assessment.duration_seconds == pytest.approx(10.0)


@pytest.mark.integration
class TestPipelineReportBundle:
    def test_run_writes_full_bundle(self, tmp_path):
        faces = [make_face(mouth_open=(0.04 if i % 2 else 0.0)) for i in range(20)]
        poses = [make_pose() for _ in range(20)]
        config = PipelineConfig(output_dir=str(tmp_path), with_plots=True)
        pipe = OratoryAnalysisPipeline(
            ScriptedFaceExtractor(faces), ScriptedPoseExtractor(poses), config=config
        )
        # Patch analyze_video to use our synthetic source instead of a real file.
        result = pipe.analyze_frames_source(_source(20))
        from oratory_analyzer.report.builder import ReportBuilder

        written = ReportBuilder(with_plots=True).build(result.assessment, str(tmp_path))
        assert os.path.exists(written["html"])
        with open(written["json"]) as fh:
            data = json.load(fh)
        assert data["overall_score"] == result.assessment.overall_score
