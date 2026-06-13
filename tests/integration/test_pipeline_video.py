"""Full pipeline.run() over a real (synthetic) video using fake extractors.

This drives the actual OpenCV decode path, speaker selection, metrics, report
bundle, and annotated-video rendering end to end — without MediaPipe — so the
orchestration plumbing is covered deterministically.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from oratory_analyzer.config import PipelineConfig  # noqa: E402
from oratory_analyzer.landmarks import (  # noqa: E402
    ScriptedFaceExtractor,
    ScriptedPoseExtractor,
)
from oratory_analyzer.pipeline import OratoryAnalysisPipeline  # noqa: E402

from ..conftest import make_face, make_pose  # noqa: E402


def _write_clip(path, n, fps=15, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n):
        writer.write(np.full((h, w, 3), (i * 7) % 255, dtype=np.uint8))
    writer.release()


@pytest.mark.integration
def test_full_run_with_annotated_video(tmp_path):
    clip = str(tmp_path / "speech.mp4")
    n = 30
    _write_clip(clip, n=n, fps=15)

    # Enough scripted landmarks to cover every decoded frame.
    faces = [make_face(mouth_open=(0.04 if i % 2 else 0.0)) for i in range(n + 5)]
    poses = [make_pose() for _ in range(n + 5)]

    config = PipelineConfig(
        sample_fps=15,
        output_dir=str(tmp_path / "report"),
        write_annotated_video=True,
        with_plots=True,
    )
    pipe = OratoryAnalysisPipeline(
        ScriptedFaceExtractor(faces), ScriptedPoseExtractor(poses), config=config
    )
    written = pipe.run(clip)
    pipe.close()

    assert os.path.exists(written["html"])
    assert os.path.exists(written["json"])
    assert os.path.exists(written["md"])
    assert os.path.exists(written["annotated_video"])
    # annotated video should be a readable, non-empty file
    assert os.path.getsize(written["annotated_video"]) > 0


@pytest.mark.integration
def test_analyze_video_returns_assessment(tmp_path):
    clip = str(tmp_path / "speech.mp4")
    _write_clip(clip, n=20, fps=10)
    faces = [make_face() for _ in range(30)]
    pipe = OratoryAnalysisPipeline(
        ScriptedFaceExtractor(faces), None,
        config=PipelineConfig(sample_fps=10, analyze_pose=False),
    )
    result = pipe.analyze_video(clip)
    assert result.metadata is not None
    assert result.assessment.frames_analyzed >= 15
    assert "eye_contact" in result.assessment.metrics
