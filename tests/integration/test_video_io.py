"""Round-trip tests for the OpenCV video reader/writer/annotator.

Uses real OpenCV encode/decode (no MediaPipe). Generates a tiny synthetic clip,
reads it back at a subsampled rate, annotates frames, and re-encodes.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from oratory_analyzer.video import (  # noqa: E402
    FrameAnnotator,
    OpenCVVideoReader,
    OpenCVVideoWriter,
)
from oratory_analyzer.video.reader import VideoReadError  # noqa: E402

from ..conftest import make_frame  # noqa: E402


def _write_clip(path, n=30, fps=30, w=160, h=120):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.mark.integration
class TestVideoReader:
    def test_reads_metadata_and_frames(self, tmp_path):
        path = str(tmp_path / "clip.mp4")
        _write_clip(path, n=30, fps=30, w=160, h=120)
        with OpenCVVideoReader(path) as reader:
            assert reader.metadata.width == 160
            assert reader.metadata.height == 120
            assert reader.metadata.fps == pytest.approx(30, abs=1)
            frames = list(reader.frames())
        assert len(frames) >= 25  # ~30 decoded
        idx, ts, frame = frames[0]
        assert idx == 0 and ts == pytest.approx(0.0)
        assert frame.shape == (120, 160, 3)

    def test_subsampling_reduces_frame_count(self, tmp_path):
        path = str(tmp_path / "clip.mp4")
        _write_clip(path, n=30, fps=30)
        with OpenCVVideoReader(path, sample_fps=10) as reader:
            assert reader.stride == 3
            frames = list(reader.frames())
        assert 8 <= len(frames) <= 12  # ~ every 3rd frame

    def test_missing_file_raises(self):
        with pytest.raises(VideoReadError):
            OpenCVVideoReader("nonexistent_file_98765.mp4")


@pytest.mark.integration
class TestWriterAndAnnotator:
    def test_annotate_and_write_roundtrip(self, tmp_path):
        src = str(tmp_path / "src.mp4")
        out = str(tmp_path / "out.mp4")
        _write_clip(src, n=10, fps=15, w=160, h=120)

        annotator = FrameAnnotator()
        fl = make_frame()  # neutral face+pose; coordinates normalized
        with OpenCVVideoReader(src) as reader:
            meta = reader.metadata
            with OpenCVVideoWriter(out, fps=15, width=meta.width, height=meta.height) as writer:
                for _idx, _ts, frame in reader.frames():
                    annotated = annotator.annotate(frame, fl)
                    assert annotated.shape == frame.shape
                    writer.write(annotated)

        # The annotated file should be readable and non-trivial.
        with OpenCVVideoReader(out) as reader:
            assert len(list(reader.frames())) >= 8
