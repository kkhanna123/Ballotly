"""A lightweight IoU-based multi-object tracker.

Given per-frame face bounding boxes, it assigns stable track IDs so that the
same person keeps the same ID across frames. This is the basis for identifying
*the speaker* in clips that contain more than one detected face (audience,
opponents, a moderator). It is pure and deterministic — no ML, no pixels — so it
is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..domain.landmarks import BoundingBox


@dataclass
class Track:
    """A sequence of boxes believed to belong to one person."""

    track_id: int
    boxes: Dict[int, BoundingBox] = field(default_factory=dict)

    def add(self, frame_index: int, box: BoundingBox) -> None:
        self.boxes[frame_index] = box

    @property
    def last_box(self) -> BoundingBox:
        last_frame = max(self.boxes)
        return self.boxes[last_frame]

    @property
    def first_frame(self) -> int:
        return min(self.boxes)

    @property
    def last_frame(self) -> int:
        return max(self.boxes)

    @property
    def length(self) -> int:
        return len(self.boxes)

    def mean_area(self) -> float:
        return sum(b.area for b in self.boxes.values()) / len(self.boxes)

    def mean_centrality(self) -> float:
        """Mean closeness of the box center to the frame center (1 == centered)."""
        total = 0.0
        for b in self.boxes.values():
            cx, cy = b.center
            # distance from (0.5, 0.5), max ~0.707; convert to a 0..1 closeness.
            dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            total += max(0.0, 1.0 - dist / 0.7071)
        return total / len(self.boxes)


class IoUTracker:
    """Greedy IoU tracker with track expiry.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU for a detection to continue an existing track.
    max_age:
        Number of consecutive frames a track may go unmatched before it expires.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: Dict[int, Track] = {}
        self._last_seen: Dict[int, int] = {}
        self._next_id = 0

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks.values())

    def update(self, frame_index: int, boxes: List[BoundingBox]) -> List[Tuple[BoundingBox, int]]:
        """Assign ``boxes`` in this frame to tracks; return (box, track_id) pairs."""
        # Expire stale tracks.
        for tid in list(self._tracks):
            if frame_index - self._last_seen[tid] > self.max_age:
                # Keep the historical track for later analysis but stop matching.
                # We move it out of the "active" matching set by leaving it in
                # _tracks but its last_seen is old; we simply won't match it.
                pass

        active = {
            tid: self._tracks[tid]
            for tid in self._tracks
            if frame_index - self._last_seen[tid] <= self.max_age
        }

        # Build all candidate (iou, box_idx, track_id) and match greedily.
        candidates = []
        for bi, box in enumerate(boxes):
            for tid, track in active.items():
                iou = box.iou(track.last_box)
                if iou >= self.iou_threshold:
                    candidates.append((iou, bi, tid))
        candidates.sort(reverse=True)

        assigned_boxes: Dict[int, int] = {}  # box_idx -> track_id
        used_tracks = set()
        for iou, bi, tid in candidates:
            if bi in assigned_boxes or tid in used_tracks:
                continue
            assigned_boxes[bi] = tid
            used_tracks.add(tid)

        results: List[Tuple[BoundingBox, int]] = []
        for bi, box in enumerate(boxes):
            if bi in assigned_boxes:
                tid = assigned_boxes[bi]
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = Track(track_id=tid)
            self._tracks[tid].add(frame_index, box)
            self._last_seen[tid] = frame_index
            results.append((box, tid))
        return results
