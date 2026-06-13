"""Tests for the core landmark value objects."""

from __future__ import annotations

import math

import pytest

from oratory_analyzer.domain.landmarks import (
    BoundingBox,
    FaceLandmarks,
    FrameLandmarks,
    Point3D,
    PoseLandmarks,
)


class TestPoint3D:
    def test_distance_2d(self):
        a = Point3D(0.0, 0.0)
        b = Point3D(3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_3d(self):
        a = Point3D(0.0, 0.0, 0.0)
        b = Point3D(0.0, 0.0, 2.0)
        assert a.distance_to(b, use_z=True) == pytest.approx(2.0)
        assert a.distance_to(b, use_z=False) == pytest.approx(0.0)

    def test_midpoint(self):
        mid = Point3D(0.0, 0.0).midpoint(Point3D(1.0, 1.0))
        assert (mid.x, mid.y) == pytest.approx((0.5, 0.5))

    def test_midpoint_visibility_is_min(self):
        mid = Point3D(0, 0, visibility=0.9).midpoint(Point3D(1, 1, visibility=0.4))
        assert mid.visibility == pytest.approx(0.4)

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            Point3D(float("nan"), 0.0)

    def test_is_frozen(self):
        p = Point3D(0.0, 0.0)
        with pytest.raises(Exception):
            p.x = 1.0  # type: ignore[misc]

    def test_as_tuple(self):
        p = Point3D(0.1, 0.2, 0.3)
        assert p.as_tuple() == (0.1, 0.2)
        assert p.as_tuple(include_z=True) == (0.1, 0.2, 0.3)


class TestFaceLandmarks:
    def test_len_and_index(self):
        face = FaceLandmarks(points=tuple(Point3D(0.0, 0.0) for _ in range(468)))
        assert len(face) == 468
        assert face[0] == Point3D(0.0, 0.0)

    def test_has_irises(self):
        assert not FaceLandmarks(tuple(Point3D(0, 0) for _ in range(468))).has_irises
        assert FaceLandmarks(tuple(Point3D(0, 0) for _ in range(478))).has_irises

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            FaceLandmarks(points=())


class TestPoseLandmarks:
    def test_len(self):
        pose = PoseLandmarks(points=tuple(Point3D(0, 0) for _ in range(33)))
        assert len(pose) == 33

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            PoseLandmarks(points=())


class TestBoundingBox:
    def test_geometry(self):
        box = BoundingBox(0.0, 0.0, 0.5, 0.4)
        assert box.width == pytest.approx(0.5)
        assert box.height == pytest.approx(0.4)
        assert box.area == pytest.approx(0.2)
        assert box.center == pytest.approx((0.25, 0.2))

    def test_invalid_box(self):
        with pytest.raises(ValueError):
            BoundingBox(0.5, 0.0, 0.1, 0.4)

    def test_iou_identical(self):
        box = BoundingBox(0.0, 0.0, 1.0, 1.0)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_disjoint(self):
        a = BoundingBox(0.0, 0.0, 0.1, 0.1)
        b = BoundingBox(0.9, 0.9, 1.0, 1.0)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_half_overlap(self):
        a = BoundingBox(0.0, 0.0, 0.2, 0.1)
        b = BoundingBox(0.1, 0.0, 0.3, 0.1)
        # intersection 0.1x0.1=0.01; union 0.02+0.02-0.01=0.03
        assert a.iou(b) == pytest.approx(1.0 / 3.0)

    def test_from_points_with_padding_clamped(self):
        pts = [Point3D(0.0, 0.0), Point3D(0.4, 0.6)]
        box = BoundingBox.from_points(pts, padding=0.1)
        assert box.x_min == 0.0  # clamped, not negative
        assert box.x_max == pytest.approx(0.5)
        assert box.y_max == pytest.approx(0.7)

    def test_from_points_empty(self):
        with pytest.raises(ValueError):
            BoundingBox.from_points([])


class TestFrameLandmarks:
    def test_flags(self):
        face = FaceLandmarks(tuple(Point3D(0, 0) for _ in range(468)))
        frame = FrameLandmarks(index=3, timestamp=0.1, face=face)
        assert frame.has_face and not frame.has_pose
        assert not frame.is_empty

    def test_empty_frame(self):
        frame = FrameLandmarks(index=0, timestamp=0.0)
        assert frame.is_empty
        assert not frame.has_face and not frame.has_pose
