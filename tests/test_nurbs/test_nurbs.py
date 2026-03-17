"""Tests for NURBS curve and surface utilities."""

import numpy as np
import pytest

from astraturbo.nurbs import (
    interpolate_3d,
    evaluate_curve,
    evaluate_curve_array,
    curve_length,
    find_u_from_point,
    interpolate_surface,
    evaluate_surface,
    xyz_to_rpz,
    rpz_to_xyz,
    norm,
    distance,
    curve_to_points,
)


class TestCurves:
    def test_interpolate_straight_line(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        crv = interpolate_3d(pts, degree=2)
        # Midpoint should be at (1, 0, 0)
        mid = evaluate_curve(crv, 0.5)
        assert mid[0] == pytest.approx(1.0, abs=0.01)
        assert mid[1] == pytest.approx(0.0, abs=0.01)

    def test_evaluate_array(self):
        pts = np.array(
            [[0, 0, 0], [0.5, 1, 0], [1, 0, 0]], dtype=np.float64
        )
        crv = interpolate_3d(pts, degree=2)
        arr = evaluate_curve_array(crv, 50)
        assert arr.shape == (50, 3)
        assert arr[0, 0] == pytest.approx(0.0, abs=0.01)
        assert arr[-1, 0] == pytest.approx(1.0, abs=0.01)

    def test_curve_length(self):
        # Straight line from (0,0,0) to (3,0,0)
        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64
        )
        crv = interpolate_3d(pts, degree=2)
        length = curve_length(crv)
        assert length == pytest.approx(3.0, abs=0.05)

    def test_find_u_from_point(self):
        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64
        )
        crv = interpolate_3d(pts, degree=2)
        target = np.array([1.0, 0.0, 0.0])
        u = find_u_from_point(crv, target)
        assert u == pytest.approx(0.5, abs=0.02)

    def test_curve_to_points(self):
        pts = np.array(
            [[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float64
        )
        crv = interpolate_3d(pts, degree=2)
        result = curve_to_points(crv, 100)
        assert result.shape == (100, 3)


class TestSurfaces:
    def test_interpolate_flat_surface(self):
        # 3x3 flat grid
        pts = np.zeros((3, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                pts[i, j] = [float(i), float(j), 0.0]

        srf = interpolate_surface(pts, degree_u=2, degree_v=2)
        # Center point should be (1, 1, 0)
        center = evaluate_surface(srf, 0.5, 0.5)
        assert center[0] == pytest.approx(1.0, abs=0.1)
        assert center[1] == pytest.approx(1.0, abs=0.1)
        assert center[2] == pytest.approx(0.0, abs=0.1)


class TestOperations:
    def test_xyz_to_rpz_and_back(self):
        xyz = np.array([1.0, 0.0, 5.0])
        rpz = xyz_to_rpz(xyz)
        assert rpz[0] == pytest.approx(1.0, abs=1e-10)  # r
        assert rpz[1] == pytest.approx(0.0, abs=1e-10)  # phi
        assert rpz[2] == pytest.approx(5.0, abs=1e-10)  # z

        back = rpz_to_xyz(rpz)
        np.testing.assert_allclose(back, xyz, atol=1e-10)

    def test_norm(self):
        assert norm(np.array([3, 4, 0])) == pytest.approx(5.0)

    def test_distance(self):
        p1 = np.array([0, 0, 0], dtype=np.float64)
        p2 = np.array([1, 0, 0], dtype=np.float64)
        assert distance(p1, p2) == pytest.approx(1.0)
