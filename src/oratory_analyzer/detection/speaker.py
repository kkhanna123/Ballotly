"""Primary-speaker identification.

Runs the IoU tracker over per-frame face boxes, scores the resulting tracks, and
designates the dominant one as *the speaker*. The intuition: across a clip, the
speaker is the face that is **present most often**, **largest** (closest to
camera), and **most central** in frame. The selector then emits, for each frame,
the speaker's bounding box (or ``None`` if the speaker isn't visible that frame),
which downstream extraction uses to pick the right face among several.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..domain.landmarks import BoundingBox
from .tracker import IoUTracker, Track


@dataclass
class SpeakerLocalization:
    """Result of locating the speaker across a clip."""

    speaker_track_id: Optional[int]
    boxes_by_frame: Dict[int, BoundingBox] = field(default_factory=dict)
    num_tracks: int = 0
    presence_fraction: float = 0.0

    def box_for(self, frame_index: int) -> Optional[BoundingBox]:
        return self.boxes_by_frame.get(frame_index)

    @property
    def found(self) -> bool:
        return self.speaker_track_id is not None


class SpeakerSelector:
    """Selects the dominant speaker track and localizes it per frame."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        w_presence: float = 1.0,
        w_area: float = 1.0,
        w_centrality: float = 0.5,
        min_track_length: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.w_presence = w_presence
        self.w_area = w_area
        self.w_centrality = w_centrality
        self.min_track_length = min_track_length

    def score_track(self, track: Track, total_frames: int) -> float:
        presence = track.length / total_frames if total_frames else 0.0
        # mean_area is in [0,1] (normalized box area); centrality in [0,1].
        return (
            self.w_presence * presence
            + self.w_area * track.mean_area()
            + self.w_centrality * track.mean_centrality()
        )

    def select(self, tracks: Sequence[Track], total_frames: int) -> Optional[Track]:
        eligible = [t for t in tracks if t.length >= self.min_track_length]
        if not eligible:
            return None
        return max(eligible, key=lambda t: self.score_track(t, total_frames))

    def localize(
        self, per_frame_boxes: Sequence[Sequence[BoundingBox]]
    ) -> SpeakerLocalization:
        """Track across frames and return the speaker's per-frame boxes.

        ``per_frame_boxes[i]`` is the list of face boxes detected in frame ``i``.
        """
        tracker = IoUTracker(iou_threshold=self.iou_threshold, max_age=self.max_age)
        total_frames = len(per_frame_boxes)
        for i, boxes in enumerate(per_frame_boxes):
            tracker.update(i, list(boxes))

        tracks = tracker.tracks
        speaker = self.select(tracks, total_frames)
        if speaker is None:
            return SpeakerLocalization(
                speaker_track_id=None, num_tracks=len(tracks)
            )

        presence = speaker.length / total_frames if total_frames else 0.0
        return SpeakerLocalization(
            speaker_track_id=speaker.track_id,
            boxes_by_frame=dict(speaker.boxes),
            num_tracks=len(tracks),
            presence_fraction=presence,
        )

    @staticmethod
    def best_match(
        candidates: Sequence[BoundingBox], target: Optional[BoundingBox]
    ) -> Optional[int]:
        """Index of the candidate box best matching ``target`` (by IoU, then center).

        Used in the extraction pass to pick the speaker's face among several
        detected faces. Returns ``None`` when there are no candidates.
        """
        if not candidates:
            return None
        if target is None:
            # No speaker region known: fall back to the largest face.
            return max(range(len(candidates)), key=lambda i: candidates[i].area)

        best_idx = None
        best_iou = -1.0
        for i, box in enumerate(candidates):
            iou = box.iou(target)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou > 0.0:
            return best_idx
        # Disjoint from target: choose the nearest center instead.
        tcx, tcy = target.center

        def center_dist(i: int) -> float:
            cx, cy = candidates[i].center
            return (cx - tcx) ** 2 + (cy - tcy) ** 2

        return min(range(len(candidates)), key=center_dist)
