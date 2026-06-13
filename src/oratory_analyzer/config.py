"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Tunable knobs for an analysis run.

    Attributes
    ----------
    sample_fps:
        Target frames-per-second to analyze. Lower values run faster; 8-15 is
        plenty for delivery analysis. Set to a large number to analyze every
        frame.
    max_num_faces:
        Maximum faces MediaPipe detects per frame (enables speaker selection in
        multi-person clips).
    analyze_face / analyze_pose:
        Toggle face-mesh and pose extraction independently.
    write_annotated_video:
        If true, render an annotated mp4 alongside the report.
    output_dir:
        Directory to write the report bundle into.
    with_plots:
        Generate summary charts (requires matplotlib).
    """

    sample_fps: float = 12.0
    max_num_faces: int = 3
    analyze_face: bool = True
    analyze_pose: bool = True
    write_annotated_video: bool = False
    output_dir: str = "oratory_report"
    with_plots: bool = True

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.max_num_faces < 1:
            raise ValueError("max_num_faces must be >= 1")
        if not (self.analyze_face or self.analyze_pose):
            raise ValueError("at least one of analyze_face/analyze_pose must be true")
