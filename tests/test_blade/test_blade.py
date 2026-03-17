"""Tests for 3D blade geometry."""

import numpy as np
import pytest

from astraturbo.camberline import NACA2Digit
from astraturbo.thickness import NACA4Digit
from astraturbo.profile import Superposition
from astraturbo.blade import BladeRow
from astraturbo.blade.stacking import axial_stacking
from astraturbo.blade.camber_surface import extract_camber_surface
from astraturbo.blade.hub_shroud import MeridionalContour, compute_stacking_line


class TestStacking:
    def test_axial_stacking_shape(self):
        # Two profiles at different radii
        profile = np.column_stack((
            np.linspace(0, 1, 50),
            0.05 * np.sin(np.linspace(0, np.pi, 50)),
        ))
        profiles = [profile, profile]
        radii = np.array([0.1, 0.2])
        stagger = np.array([0.0, 0.0])
        chords = np.array([0.05, 0.06])

        result = axial_stacking(profiles, radii, stagger, chords)
        assert len(result) == 2
        assert result[0].shape == (50, 3)
        assert result[1].shape == (50, 3)


class TestMeridionalContour:
    def test_radius_at_z(self):
        pts = np.array([[0.0, 0.1], [0.05, 0.1], [0.1, 0.1]])
        contour = MeridionalContour(pts)
        assert contour.radius_at_z(0.05) == pytest.approx(0.1, abs=1e-6)

    def test_stacking_line(self):
        hub = MeridionalContour(np.array([[0.0, 0.1], [0.1, 0.1]]))
        shroud = MeridionalContour(np.array([[0.0, 0.2], [0.1, 0.2]]))
        radii = compute_stacking_line(hub, shroud, 5)
        assert len(radii) == 5
        assert radii[0] == pytest.approx(0.1, abs=0.01)
        assert radii[-1] == pytest.approx(0.2, abs=0.01)


class TestBladeRow:
    def _make_blade_row(self, n_profiles=3):
        row = BladeRow(
            hub_points=np.array([[0.0, 0.1], [0.1, 0.1]]),
            shroud_points=np.array([[0.0, 0.2], [0.1, 0.2]]),
            stacking_mode=0,
        )
        for _ in range(n_profiles):
            row.add_profile(Superposition.default())
        return row

    def test_create_blade_row(self):
        row = self._make_blade_row()
        assert len(row.profiles) == 3
        assert row.stacking_mode == 0

    def test_compute(self):
        row = self._make_blade_row(3)
        stagger = np.zeros(3)
        chords = np.ones(3) * 0.05
        row.compute(stagger_angles=stagger, chord_lengths=chords)
        assert row.blade_surface is not None
        assert row.profiles_3d is not None
        assert len(row.profiles_3d) == 3

    def test_leading_trailing_edges(self):
        row = self._make_blade_row(3)
        row.compute(
            stagger_angles=np.zeros(3),
            chord_lengths=np.ones(3) * 0.05,
        )
        le = row.leading_edge
        te = row.trailing_edge
        assert le is not None
        assert te is not None
        assert le.shape == (3, 3)
        assert te.shape == (3, 3)


class TestCamberSurface:
    def test_extract_camber(self):
        p = Superposition.default()
        arr = p.as_array()
        # Simulate 3 span positions
        profiles_3d = [
            np.column_stack((arr, np.full(len(arr), z)))
            for z in [0.0, 0.5, 1.0]
        ]
        camber = extract_camber_surface(profiles_3d)
        assert camber.shape[0] == 3  # 3 spans
        assert camber.shape[2] == 3  # xyz
