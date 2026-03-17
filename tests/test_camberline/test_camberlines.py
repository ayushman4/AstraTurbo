"""Tests for camber line generators."""

import math

import numpy as np
import pytest

from astraturbo.camberline import (
    CircularArc,
    CubicPolynomial,
    Joukowski,
    NACA2Digit,
    NACA65,
    QuadraticPolynomial,
    QuarticPolynomial,
    create_camberline,
)


class TestCircularArc:
    def test_shape(self):
        cl = CircularArc(angle_of_inflow=100)
        arr = cl.as_array()
        assert arr.shape == (200, 2)

    def test_endpoints(self):
        cl = CircularArc(angle_of_inflow=100)
        arr = cl.as_array()
        assert arr[0, 0] == pytest.approx(0, abs=1e-10)
        assert arr[-1, 0] == pytest.approx(1, abs=1e-10)

    def test_camber_starts_at_zero(self):
        cl = CircularArc(angle_of_inflow=100)
        arr = cl.as_array()
        assert arr[0, 1] == pytest.approx(0, abs=1e-6)
        assert arr[-1, 1] == pytest.approx(0, abs=1e-6)

    def test_derivations_shape(self):
        cl = CircularArc(angle_of_inflow=100)
        d = cl.get_derivations()
        assert d.shape == (200,)

    def test_default_factory(self):
        cl = CircularArc.default()
        assert cl.angle_of_inflow == 100


class TestPolynomials:
    def test_quadratic_shape(self):
        cl = QuadraticPolynomial(angle_of_inflow=100)
        arr = cl.as_array()
        assert arr.shape == (200, 2)
        assert arr[0, 0] == pytest.approx(0, abs=1e-10)
        assert arr[-1, 0] == pytest.approx(1, abs=1e-10)

    def test_cubic_shape(self):
        cl = CubicPolynomial(angle_of_inflow=100, angle_of_outflow=90)
        arr = cl.as_array()
        assert arr.shape == (200, 2)

    def test_quartic_shape(self):
        cl = QuarticPolynomial(
            angle_of_inflow=100, angle_of_outflow=90, max_camber_position=0.4
        )
        arr = cl.as_array()
        assert arr.shape == (200, 2)

    def test_quadratic_zero_camber_at_ends(self):
        cl = QuadraticPolynomial(angle_of_inflow=100)
        arr = cl.as_array()
        assert arr[0, 1] == pytest.approx(0, abs=1e-10)


class TestJoukowski:
    def test_shape(self):
        cl = Joukowski(max_camber=0.12)
        arr = cl.as_array()
        assert arr.shape == (200, 2)

    def test_parabolic(self):
        cl = Joukowski(max_camber=1.0)
        arr = cl.as_array()
        # Maximum should be at x=0.5: y = 1.0 * 0.5 * 0.5 = 0.25
        mid_idx = len(arr) // 2
        # Check peak is near 0.25
        assert np.max(arr[:, 1]) == pytest.approx(0.25, abs=0.01)

    def test_endpoints_zero(self):
        cl = Joukowski(max_camber=0.12)
        arr = cl.as_array()
        assert arr[0, 1] == pytest.approx(0, abs=1e-10)
        assert arr[-1, 1] == pytest.approx(0, abs=1e-10)


class TestNACA2Digit:
    def test_shape(self):
        cl = NACA2Digit(max_camber=0.02, max_camber_position=0.4)
        arr = cl.as_array()
        assert arr.shape == (200, 2)

    def test_zero_camber(self):
        cl = NACA2Digit(max_camber=0.0, max_camber_position=0.4)
        arr = cl.as_array()
        assert np.allclose(arr[:, 1], 0, atol=1e-12)


class TestNACA65:
    def test_shape(self):
        cl = NACA65(cl0=1.0)
        arr = cl.as_array()
        assert arr.shape == (200, 2)

    def test_symmetric(self):
        cl = NACA65(cl0=1.0)
        arr = cl.as_array()
        # NACA65 camber is symmetric about x=0.5
        n = len(arr)
        mid = n // 2
        # Values near the middle should be near maximum
        assert arr[mid, 1] > arr[0, 1]


class TestFactory:
    def test_create_circular_arc(self):
        cl = create_camberline("circular_arc", angle_of_inflow=100)
        assert isinstance(cl, CircularArc)

    def test_create_default(self):
        cl = create_camberline("joukowski")
        assert isinstance(cl, Joukowski)

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_camberline("nonexistent")
