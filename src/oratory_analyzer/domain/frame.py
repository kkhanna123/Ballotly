"""Video-level value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMetadata:
    """Static properties of a decoded video stream."""

    width: int
    height: int
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video dimensions must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.frame_count < 0:
            raise ValueError("frame_count must be non-negative")

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def timestamp_for(self, frame_index: int) -> float:
        """Seconds from start for a 0-based frame index."""
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        return frame_index / self.fps
