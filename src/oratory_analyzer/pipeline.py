"""End-to-end orchestration: video -> landmarks -> speaker -> metrics -> report.

The pipeline is engine-agnostic: it accepts any :class:`FaceExtractor` /
:class:`PoseExtractor` (real MediaPipe or scripted fakes), which keeps it fully
testable without pixels or ML models.

Flow
----
1. Decode frames (optionally subsampled to ``sample_fps``).
2. Per frame, extract every candidate face + the pose.
3. Track faces across frames and select the primary **speaker**.
4. For each frame, pick the speaker's face among the candidates.
5. Run metrics over the speaker's landmark series and aggregate to a grade.
6. Render the report bundle; optionally render an annotated video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .analysis.aggregator import Analyzer, OverallAssessment
from .config import PipelineConfig
from .detection.speaker import SpeakerSelector
from .domain.frame import VideoMetadata
from .domain.landmarks import BoundingBox, FrameLandmarks
from .heartbeat import NullProgress, ProgressReporter
from .landmarks.base import FaceExtractor, PoseExtractor
from .metrics.registry import MetricRegistry


@dataclass
class _FrameRecord:
    index: int
    timestamp: float
    faces: List = field(default_factory=list)         # List[FaceLandmarks]
    boxes: List[BoundingBox] = field(default_factory=list)
    pose: Optional[object] = None                       # Optional[PoseLandmarks]
    hands: tuple = ()                                    # Tuple[HandLandmarks, ...]


@dataclass
class AnalysisResult:
    """Bundle returned by the pipeline."""

    assessment: OverallAssessment
    frames: List[FrameLandmarks]
    metadata: Optional[VideoMetadata] = None


class OratoryAnalysisPipeline:
    def __init__(
        self,
        face_extractor: Optional[FaceExtractor] = None,
        pose_extractor: Optional[PoseExtractor] = None,
        hand_extractor=None,
        *,
        selector: Optional[SpeakerSelector] = None,
        registry: Optional[MetricRegistry] = None,
        analyzer: Optional[Analyzer] = None,
        config: Optional[PipelineConfig] = None,
        progress: Optional[ProgressReporter] = None,
    ) -> None:
        if face_extractor is None and pose_extractor is None and hand_extractor is None:
            raise ValueError("pipeline needs at least one extractor")
        self.face_extractor = face_extractor
        self.pose_extractor = pose_extractor
        self.hand_extractor = hand_extractor
        self.selector = selector or SpeakerSelector()
        self.registry = registry or MetricRegistry.default()
        self.analyzer = analyzer or Analyzer()
        self.config = config or PipelineConfig()
        self.progress = progress or NullProgress()

    # --- construction helpers ------------------------------------------

    @classmethod
    def with_mediapipe(
        cls, config: Optional[PipelineConfig] = None, **kwargs
    ) -> "OratoryAnalysisPipeline":
        from .landmarks import get_mediapipe_extractors

        config = config or PipelineConfig()
        ex = get_mediapipe_extractors(
            extract_face=config.analyze_face,
            extract_pose=config.analyze_pose,
            extract_hands=config.analyze_hands,
            max_num_faces=config.max_num_faces,
            max_num_hands=config.max_num_hands,
        )
        return cls(ex.face, ex.pose, ex.hands, config=config, **kwargs)

    # --- extraction primitives -----------------------------------------

    def _extract_faces(self, frame_bgr: np.ndarray) -> List:
        if self.face_extractor is None:
            return []
        # Prefer multi-face detection when the extractor supports it.
        extract_all = getattr(self.face_extractor, "extract_all", None)
        if callable(extract_all):
            return list(extract_all(frame_bgr))
        face = self.face_extractor.extract(frame_bgr)
        return [face] if face is not None else []

    def _build_speaker_frames(self, records: List[_FrameRecord]) -> List[FrameLandmarks]:
        per_frame_boxes = [r.boxes for r in records]
        localization = self.selector.localize(per_frame_boxes)
        self.progress.log(
            f"Speaker localization: {localization.num_tracks} face track(s); "
            f"speaker present in {localization.presence_fraction * 100:.0f}% of frames."
        )

        frames: List[FrameLandmarks] = []
        for pos, rec in enumerate(records):
            target = localization.box_for(pos)
            face = None
            face_box = None
            if rec.boxes:
                idx = self.selector.best_match(rec.boxes, target)
                if idx is not None:
                    face = rec.faces[idx]
                    face_box = rec.boxes[idx]
            frames.append(
                FrameLandmarks(
                    index=rec.index,
                    timestamp=rec.timestamp,
                    face=face,
                    pose=rec.pose,
                    hands=rec.hands,
                    face_box=face_box,
                )
            )
        return frames

    # --- main entry points ---------------------------------------------

    def analyze_frames_source(
        self, source, *, video_meta: Optional[VideoMetadata] = None
    ) -> AnalysisResult:
        """Analyze an iterable of ``(index, timestamp, frame_bgr)`` tuples.

        Decoupled from any specific reader so tests can feed synthetic frames.
        """
        records: List[_FrameRecord] = []
        for index, timestamp, frame_bgr in source:
            faces = self._extract_faces(frame_bgr)
            boxes = [BoundingBox.from_points(f.points, padding=0.02) for f in faces]
            pose = (
                self.pose_extractor.extract(frame_bgr)
                if self.pose_extractor is not None
                else None
            )
            hands = (
                tuple(self.hand_extractor.extract(frame_bgr))
                if self.hand_extractor is not None
                else ()
            )
            records.append(
                _FrameRecord(
                    index=index, timestamp=timestamp, faces=faces, boxes=boxes,
                    pose=pose, hands=hands,
                )
            )
            if len(records) % 50 == 0:
                self.progress.log(f"Processed {len(records)} frames…")

        if not records:
            raise ValueError("No frames were decoded from the source")

        frames = self._build_speaker_frames(records)
        frames_with_speaker = sum(
            1 for f in frames if f.has_face or f.has_pose or f.has_hands
        )

        results = self.registry.evaluate(frames)
        if not results:
            raise ValueError(
                "No metrics could be computed — no face or pose landmarks were detected."
            )
        assessment = self.analyzer.analyze(
            results,
            frames_analyzed=len(frames),
            frames_with_speaker=frames_with_speaker,
            video_meta=video_meta,
        )
        return AnalysisResult(assessment=assessment, frames=frames, metadata=video_meta)

    def analyze_video(self, video_path: str) -> AnalysisResult:
        """Decode ``video_path`` and run the full analysis."""
        from .video.reader import OpenCVVideoReader

        self.progress.log(f"Opening video: {video_path}")
        with OpenCVVideoReader(video_path, sample_fps=self.config.sample_fps) as reader:
            self.progress.log(
                f"{reader.metadata.width}x{reader.metadata.height} @ "
                f"{reader.metadata.fps:.1f}fps, "
                f"{reader.metadata.duration_seconds:.1f}s; stride={reader.stride}"
            )
            result = self.analyze_frames_source(
                reader.frames(), video_meta=reader.metadata
            )
        return result

    def run(self, video_path: str) -> Dict[str, str]:
        """Full run: analyze, write the report bundle, optional annotated video."""
        from .report.builder import ReportBuilder

        result = self.analyze_video(video_path)
        self.progress.log(
            f"Overall score {result.assessment.overall_score}/100 "
            f"(grade {result.assessment.grade})."
        )

        builder = ReportBuilder(with_plots=self.config.with_plots)
        written = builder.build(result.assessment, self.config.output_dir)
        self.progress.log(f"Report written to {self.config.output_dir}/")

        if self.config.write_annotated_video:
            out_path = self._render_annotated(video_path, result.frames)
            if out_path:
                written["annotated_video"] = out_path
        return written

    def _render_annotated(
        self, video_path: str, frames: List[FrameLandmarks]
    ) -> Optional[str]:
        import os

        from .video.annotator import FrameAnnotator
        from .video.reader import OpenCVVideoReader
        from .video.writer import OpenCVVideoWriter

        by_index = {f.index: f for f in frames}
        annotator = FrameAnnotator(
            draw_face_mesh=self.config.analyze_face,
            draw_pose=self.config.analyze_pose,
            draw_hands=self.config.analyze_hands,
        )
        os.makedirs(self.config.output_dir, exist_ok=True)
        out_path = os.path.join(self.config.output_dir, "annotated.mp4")

        with OpenCVVideoReader(video_path, sample_fps=self.config.sample_fps) as reader:
            meta = reader.metadata
            writer = OpenCVVideoWriter(
                out_path, fps=self.config.sample_fps, width=meta.width, height=meta.height
            )
            try:
                for index, _ts, frame in reader.frames():
                    fl = by_index.get(index)
                    if fl is not None:
                        frame = annotator.annotate(frame, fl)
                    writer.write(frame)
            finally:
                writer.close()
        self.progress.log(f"Annotated video written to {out_path}")
        return out_path

    def close(self) -> None:
        if self.face_extractor is not None:
            self.face_extractor.close()
        if self.pose_extractor is not None:
            self.pose_extractor.close()
        if self.hand_extractor is not None:
            self.hand_extractor.close()
