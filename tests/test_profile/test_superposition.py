"""Tests for profile superposition."""

import numpy as np
import pytest

from astraturbo.camberline import NACA2Digit, Joukowski
from astraturbo.thickness import NACA4Digit, Elliptic
from astraturbo.profile import Superposition


class TestSuperposition:
    def test_default(self):
        p = Superposition.default()
        arr = p.as_array()
        # Closed profile: 2*N - 1 points
        assert arr.shape[1] == 2
        assert arr.shape[0] == 2 * 200 - 1

    def test_closed_contour(self):
        p = Superposition.default()
        arr = p.as_array()
        # First and last points should be at (or near) trailing edge
        assert arr[0, 0] == pytest.approx(arr[-1, 0], abs=1e-6)

    def test_upper_lower_surfaces(self):
        p = Superposition.default()
        upper = p.upper_surface()
        lower = p.lower_surface()
        assert upper.shape == (200, 2)
        assert lower.shape == (200, 2)

    def test_upper_above_lower(self):
        p = Superposition(
            camber_line=Joukowski(max_camber=0.05),
            thickness_distribution=NACA4Digit(max_thickness=0.12),
        )
        upper = p.upper_surface()
        lower = p.lower_surface()
        # In the middle of the chord, upper y should be > lower y
        mid = len(upper) // 2
        assert upper[mid, 1] > lower[mid, 1]

    def test_custom_camber_and_thickness(self):
        p = Superposition(
            camber_line=NACA2Digit(max_camber=0.04, max_camber_position=0.3),
            thickness_distribution=Elliptic(max_thickness=0.15),
        )
        arr = p.as_array()
        assert arr.shape[0] == 2 * 200 - 1

    def test_centroid_reasonable(self):
        p = Superposition.default()
        c = p.centroid
        # Centroid should be somewhere near the middle of the chord
        assert 0.2 < c[0] < 0.8
