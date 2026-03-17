"""Tests for thickness distributions."""

import numpy as np
import pytest

from astraturbo.thickness import (
    NACA4Digit,
    NACA65Series,
    JoukowskiThickness,
    Elliptic,
    create_thickness,
)


class TestNACA4Digit:
    def test_shape(self):
        td = NACA4Digit(max_thickness=0.12)
        arr = td.as_array()
        assert arr.shape == (200, 2)

    def test_positive_thickness(self):
        td = NACA4Digit(max_thickness=0.12)
        arr = td.as_array()
        # Thickness should be non-negative everywhere
        assert np.all(arr[1:-1, 1] > 0)

    def test_zero_at_leading_edge(self):
        td = NACA4Digit(max_thickness=0.12)
        arr = td.as_array()
        assert arr[0, 1] == pytest.approx(0, abs=1e-6)


class TestNACA65Series:
    def test_shape(self):
        td = NACA65Series(max_thickness=0.1)
        arr = td.as_array()
        assert arr.shape == (200, 2)

    def test_positive_thickness(self):
        td = NACA65Series(max_thickness=0.1)
        arr = td.as_array()
        # Interior points should have positive thickness
        assert np.all(arr[1:-1, 1] >= 0)


class TestJoukowski:
    def test_shape(self):
        td = JoukowskiThickness(max_thickness=0.1)
        arr = td.as_array()
        assert arr.shape == (200, 2)


class TestElliptic:
    def test_shape(self):
        td = Elliptic(max_thickness=0.1)
        arr = td.as_array()
        assert arr.shape == (200, 2)

    def test_max_at_midchord(self):
        td = Elliptic(max_thickness=0.2)
        arr = td.as_array()
        # Maximum should be at approximately x=0.5
        max_idx = np.argmax(arr[:, 1])
        assert arr[max_idx, 0] == pytest.approx(0.5, abs=0.05)
        assert arr[max_idx, 1] == pytest.approx(0.1, abs=0.01)

    def test_zero_at_endpoints(self):
        td = Elliptic(max_thickness=0.1)
        arr = td.as_array()
        assert arr[0, 1] == pytest.approx(0, abs=1e-6)
        assert arr[-1, 1] == pytest.approx(0, abs=1e-6)


class TestFactory:
    def test_create_naca4digit(self):
        td = create_thickness("naca4digit", max_thickness=0.12)
        assert isinstance(td, NACA4Digit)

    def test_create_default(self):
        td = create_thickness("elliptic")
        assert isinstance(td, Elliptic)

    def test_unknown(self):
        with pytest.raises(ValueError):
            create_thickness("nonexistent")
