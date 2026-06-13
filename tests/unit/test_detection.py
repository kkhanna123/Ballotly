"""Tests for the IoU tracker and speaker selector."""

from __future__ import annotations

import pytest

from oratory_analyzer.detection import IoUTracker, SpeakerSelector
from oratory_analyzer.domain.landmarks import BoundingBox


def box(cx, cy, size=0.2):
    h = size / 2.0
    return BoundingBox(cx - h, cy - h, cx + h, cy + h)


class TestIoUTracker:
    def test_single_object_keeps_id(self):
        tracker = IoUTracker(iou_threshold=0.3)
        ids = []
        for i in range(5):
            # drift slightly each frame but heavily overlapping
            res = tracker.update(i, [box(0.5 + i * 0.01, 0.5)])
            ids.append(res[0][1])
        assert len(set(ids)) == 1  # same track id throughout

    def test_two_objects_get_two_ids(self):
        tracker = IoUTracker(iou_threshold=0.3)
        res = tracker.update(0, [box(0.25, 0.5), box(0.75, 0.5)])
        ids = {tid for _, tid in res}
        assert len(ids) == 2
        # next frame: same two persons keep their ids
        res2 = tracker.update(1, [box(0.26, 0.5), box(0.74, 0.5)])
        assert {tid for _, tid in res2} == ids

    def test_new_track_after_disappearance_beyond_max_age(self):
        tracker = IoUTracker(iou_threshold=0.3, max_age=2)
        first = tracker.update(0, [box(0.5, 0.5)])[0][1]
        # absent for 3 frames (> max_age)
        for i in range(1, 4):
            tracker.update(i, [])
        reappeared = tracker.update(5, [box(0.5, 0.5)])[0][1]
        assert reappeared != first

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            IoUTracker(iou_threshold=0.0)

    def test_track_stats(self):
        tracker = IoUTracker()
        for i in range(4):
            tracker.update(i, [box(0.5, 0.5, size=0.4)])
        track = tracker.tracks[0]
        assert track.length == 4
        assert track.mean_area() == pytest.approx(0.16, abs=1e-6)
        assert track.mean_centrality() > 0.9  # centered


class TestSpeakerSelector:
    def test_picks_present_and_large_face(self):
        # speaker: large, central, present every frame.
        # bystander: small, off to the side, present only twice.
        per_frame = []
        for i in range(10):
            frame = [box(0.5, 0.5, size=0.4)]  # speaker
            if i < 2:
                frame.append(box(0.1, 0.1, size=0.1))  # transient bystander
            per_frame.append(frame)
        loc = SpeakerSelector().localize(per_frame)
        assert loc.found
        assert loc.presence_fraction == pytest.approx(1.0)
        # speaker box present for every frame
        assert all(loc.box_for(i) is not None for i in range(10))
        assert loc.num_tracks == 2

    def test_no_faces_returns_not_found(self):
        loc = SpeakerSelector().localize([[], [], []])
        assert not loc.found
        assert loc.speaker_track_id is None

    def test_larger_face_beats_more_central_small_face(self):
        # A big off-center face should beat a tiny perfectly-centered one.
        per_frame = [[box(0.35, 0.4, size=0.5), box(0.5, 0.5, size=0.05)] for _ in range(8)]
        loc = SpeakerSelector().localize(per_frame)
        # The speaker's boxes should be the large ones (area ~0.25).
        assert loc.box_for(0).area > 0.1

    def test_best_match_by_iou(self):
        target = box(0.5, 0.5, size=0.3)
        candidates = [box(0.1, 0.1, size=0.2), box(0.5, 0.5, size=0.3)]
        assert SpeakerSelector.best_match(candidates, target) == 1

    def test_best_match_no_target_uses_largest(self):
        candidates = [box(0.1, 0.1, size=0.1), box(0.8, 0.8, size=0.4)]
        assert SpeakerSelector.best_match(candidates, None) == 1

    def test_best_match_disjoint_uses_nearest_center(self):
        target = box(0.5, 0.5, size=0.1)
        candidates = [box(0.9, 0.9, size=0.1), box(0.55, 0.55, size=0.1)]
        assert SpeakerSelector.best_match(candidates, target) == 1

    def test_best_match_empty(self):
        assert SpeakerSelector.best_match([], box(0.5, 0.5)) is None
