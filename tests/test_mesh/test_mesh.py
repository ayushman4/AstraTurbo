"""Tests for mesh generation modules."""

import numpy as np
import pytest

from astraturbo.mesh.transfinite import (
    tfi_2d,
    tfi_2d_vectorized,
    tfi_2d_graded,
    apply_grading,
)
from astraturbo.mesh.scm_mesher import SCMMesher, SCMMeshConfig
from astraturbo.mesh.ogrid import OGridGenerator, OGridMeshConfig
from astraturbo.mesh.quality import (
    compute_aspect_ratio,
    compute_skewness,
    estimate_yplus,
    first_cell_height_for_yplus,
    mesh_quality_report,
)


class TestTransfiniteInterpolation:
    def test_rectangular_mesh(self):
        # Unit square boundaries
        n = 11
        bottom = np.column_stack((np.linspace(0, 1, n), np.zeros(n)))
        top = np.column_stack((np.linspace(0, 1, n), np.ones(n)))
        left = np.column_stack((np.zeros(n), np.linspace(0, 1, n)))
        right = np.column_stack((np.ones(n), np.linspace(0, 1, n)))

        mesh = tfi_2d(bottom, top, left, right)
        assert mesh.shape == (n, n, 2)

        # Center point should be (0.5, 0.5)
        mid = n // 2
        assert mesh[mid, mid, 0] == pytest.approx(0.5, abs=0.01)
        assert mesh[mid, mid, 1] == pytest.approx(0.5, abs=0.01)

    def test_vectorized_matches_loop(self):
        n = 11
        bottom = np.column_stack((np.linspace(0, 1, n), np.zeros(n)))
        top = np.column_stack((np.linspace(0, 1, n), np.ones(n)))
        left = np.column_stack((np.zeros(n), np.linspace(0, 1, n)))
        right = np.column_stack((np.ones(n), np.linspace(0, 1, n)))

        mesh_loop = tfi_2d(bottom, top, left, right)
        mesh_vec = tfi_2d_vectorized(bottom, top, left, right)
        np.testing.assert_allclose(mesh_loop, mesh_vec, atol=1e-12)

    def test_graded_mesh(self):
        n = 11
        bottom = np.column_stack((np.linspace(0, 1, n), np.zeros(n)))
        top = np.column_stack((np.linspace(0, 1, n), np.ones(n)))
        left = np.column_stack((np.zeros(n), np.linspace(0, 1, n)))
        right = np.column_stack((np.ones(n), np.linspace(0, 1, n)))

        mesh = tfi_2d_graded(bottom, top, left, right, grading_s=2.0)
        assert mesh.shape[0] == n
        assert mesh.shape[1] == n

    def test_apply_grading_uniform(self):
        params = apply_grading(10, 1.0)
        assert len(params) == 11
        assert params[0] == pytest.approx(0.0)
        assert params[-1] == pytest.approx(1.0)
        # Uniform spacing
        diffs = np.diff(params)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-10)

    def test_apply_grading_nonuniform(self):
        params = apply_grading(10, 3.0)
        assert len(params) == 11
        diffs = np.diff(params)
        # Last cell should be larger than first
        assert diffs[-1] > diffs[0]


class TestSCMMesher:
    def test_generate_3_blocks(self):
        hub = np.array([[0.0, 0.1], [0.05, 0.1], [0.1, 0.1]])
        shroud = np.array([[0.0, 0.2], [0.05, 0.2], [0.1, 0.2]])

        mesher = SCMMesher(SCMMeshConfig(
            n_inlet_axial=5, n_blade_axial=10, n_outlet_axial=5, n_radial=5,
        ))
        blocks = mesher.generate(hub, shroud, le_z=0.03, te_z=0.07)

        assert len(blocks) == 3
        for block in blocks:
            assert block.points is not None
            assert block.points.shape[2] == 2  # z, r coordinates

    def test_get_all_points(self):
        hub = np.array([[0.0, 0.1], [0.1, 0.1]])
        shroud = np.array([[0.0, 0.2], [0.1, 0.2]])

        mesher = SCMMesher(SCMMeshConfig(
            n_inlet_axial=3, n_blade_axial=5, n_outlet_axial=3, n_radial=3,
        ))
        mesher.generate(hub, shroud, le_z=0.03, te_z=0.07)
        all_pts = mesher.get_all_points()
        assert all_pts.shape[1] == 2
        assert len(all_pts) > 0


class TestOGridGenerator:
    def test_generate_mesh(self):
        # Simple airfoil-like profile
        t = np.linspace(0, 2 * np.pi, 100)
        profile = np.column_stack((
            0.5 + 0.5 * np.cos(t),
            0.1 * np.sin(t),
        ))

        gen = OGridGenerator(OGridMeshConfig(
            n_ogrid_normal=3,
            n_blade_wrap=20,
            n_inlet=5,
            n_outlet=5,
            n_passage=6,
        ))
        mesh = gen.generate(profile, pitch=0.5)
        assert mesh.n_blocks > 0
        all_pts = mesh.get_all_points()
        assert all_pts.shape[0] > 0


class TestMeshQuality:
    def _make_uniform_block(self):
        x = np.linspace(0, 1, 6)
        y = np.linspace(0, 1, 6)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return np.stack((xx, yy), axis=-1)

    def test_aspect_ratio_uniform(self):
        block = self._make_uniform_block()
        ar = compute_aspect_ratio(block)
        assert ar.shape == (5, 5)
        np.testing.assert_allclose(ar, 1.0, atol=0.01)

    def test_skewness_rectangular(self):
        block = self._make_uniform_block()
        skew = compute_skewness(block)
        assert skew.shape == (5, 5)
        # Rectangular grid should have zero skewness
        np.testing.assert_allclose(skew, 0.0, atol=0.01)

    def test_yplus_estimate(self):
        yp = estimate_yplus(
            first_cell_height=1e-5,
            density=1.225,
            velocity=100.0,
            dynamic_viscosity=1.8e-5,
            chord=0.1,
        )
        assert yp > 0
        assert yp < 100  # Should be reasonable for this config

    def test_yplus_inverse(self):
        dy = first_cell_height_for_yplus(
            target_yplus=1.0,
            density=1.225,
            velocity=100.0,
            dynamic_viscosity=1.8e-5,
            chord=0.1,
        )
        assert dy > 0
        assert dy < 1e-3  # Should be sub-mm for y+=1

    def test_quality_report(self):
        block = self._make_uniform_block()
        report = mesh_quality_report(block)
        assert "aspect_ratio_max" in report
        assert "skewness_max" in report
        assert report["n_cells"] == 25
        assert report["n_points"] == 36
