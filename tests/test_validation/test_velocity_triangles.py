"""Validation test: velocity triangle calculations.

Tests velocity triangle computations against textbook examples with
known exact answers. These are fundamental turbomachinery relations
that must be exactly correct.

Velocity triangles relate:
  - C: Absolute velocity
  - W: Relative velocity
  - U: Blade speed
  - alpha: Absolute flow angle (from axial)
  - beta: Relative flow angle (from axial)
  - C_axial = C * cos(alpha) = W * cos(beta) (axial component)
  - C_theta = C * sin(alpha) (absolute tangential)
  - W_theta = W * sin(beta) (relative tangential)
  - C_theta = U + W_theta (for rotors)
"""

from __future__ import annotations

import numpy as np
import pytest


def compute_velocity_triangle(
    U: float,
    C_axial: float,
    alpha: float,
) -> dict[str, float]:
    """Compute full velocity triangle from blade speed, axial velocity, and abs angle.

    Args:
        U: Blade speed (m/s).
        C_axial: Axial velocity component (m/s).
        alpha: Absolute flow angle from axial (degrees).

    Returns:
        Dictionary with C, W, beta, C_theta, W_theta, C_axial.
    """
    alpha_rad = np.radians(alpha)

    # Absolute velocity components
    C_theta = C_axial * np.tan(alpha_rad)
    C = C_axial / np.cos(alpha_rad)

    # Relative velocity components
    W_theta = C_theta - U
    W = np.sqrt(C_axial**2 + W_theta**2)
    beta = np.degrees(np.arctan2(W_theta, C_axial))

    return {
        "C": float(C),
        "W": float(W),
        "beta": float(beta),
        "C_theta": float(C_theta),
        "W_theta": float(W_theta),
        "C_axial": float(C_axial),
        "alpha": float(alpha),
        "U": float(U),
    }


class TestVelocityTriangles:
    """Validate velocity triangle calculations against exact solutions."""

    def test_zero_swirl_inlet(self) -> None:
        """Test with zero inlet swirl (alpha=0).

        When alpha=0: C = C_axial, C_theta = 0, W_theta = -U
        """
        U = 300.0
        C_axial = 150.0
        alpha = 0.0

        tri = compute_velocity_triangle(U, C_axial, alpha)

        assert tri["C"] == pytest.approx(150.0, rel=1e-10)
        assert tri["C_theta"] == pytest.approx(0.0, abs=1e-10)
        assert tri["W_theta"] == pytest.approx(-300.0, rel=1e-10)
        assert tri["W"] == pytest.approx(np.sqrt(150**2 + 300**2), rel=1e-10)
        assert tri["beta"] == pytest.approx(
            np.degrees(np.arctan2(-300, 150)), rel=1e-10
        )

    def test_30_degree_inlet_swirl(self) -> None:
        """Test with alpha=30 degrees.

        U=300, C_axial=150, alpha=30:
          C_theta = 150 * tan(30) = 150/sqrt(3) = 86.60
          C = 150 / cos(30) = 150 / (sqrt(3)/2) = 173.21
          W_theta = C_theta - U = 86.60 - 300 = -213.40
          W = sqrt(150^2 + 213.40^2) = 260.84
          beta = atan2(-213.40, 150) = -54.89 deg
        """
        U = 300.0
        C_axial = 150.0
        alpha = 30.0

        tri = compute_velocity_triangle(U, C_axial, alpha)

        C_theta_expected = 150.0 * np.tan(np.radians(30))
        C_expected = 150.0 / np.cos(np.radians(30))
        W_theta_expected = C_theta_expected - 300.0
        W_expected = np.sqrt(150.0**2 + W_theta_expected**2)
        beta_expected = np.degrees(np.arctan2(W_theta_expected, 150.0))

        assert tri["C"] == pytest.approx(C_expected, rel=1e-10)
        assert tri["C_theta"] == pytest.approx(C_theta_expected, rel=1e-10)
        assert tri["W_theta"] == pytest.approx(W_theta_expected, rel=1e-10)
        assert tri["W"] == pytest.approx(W_expected, rel=1e-10)
        assert tri["beta"] == pytest.approx(beta_expected, rel=1e-10)

    def test_45_degree_inlet_swirl(self) -> None:
        """Test with alpha=45 degrees.

        At 45 degrees: C_theta = C_axial, C = C_axial * sqrt(2).
        """
        U = 200.0
        C_axial = 200.0
        alpha = 45.0

        tri = compute_velocity_triangle(U, C_axial, alpha)

        assert tri["C_theta"] == pytest.approx(200.0, rel=1e-10)
        assert tri["C"] == pytest.approx(200.0 * np.sqrt(2), rel=1e-10)
        # W_theta = 200 - 200 = 0 -> beta = 0
        assert tri["W_theta"] == pytest.approx(0.0, abs=1e-10)
        assert tri["beta"] == pytest.approx(0.0, abs=1e-10)
        assert tri["W"] == pytest.approx(200.0, rel=1e-10)

    def test_negative_swirl(self) -> None:
        """Test with negative swirl (counter-rotation)."""
        U = 200.0
        C_axial = 150.0
        alpha = -20.0

        tri = compute_velocity_triangle(U, C_axial, alpha)

        alpha_rad = np.radians(-20)
        C_theta_expected = 150.0 * np.tan(alpha_rad)
        assert tri["C_theta"] == pytest.approx(C_theta_expected, rel=1e-10)
        assert tri["C_theta"] < 0, "Negative swirl should give negative C_theta"

    def test_axial_velocity_conservation(self) -> None:
        """Axial velocity should be the same in both frames.

        C_axial = C * cos(alpha) = W * cos(beta)
        """
        U = 250.0
        C_axial = 180.0
        alpha = 35.0

        tri = compute_velocity_triangle(U, C_axial, alpha)

        C_axial_from_C = tri["C"] * np.cos(np.radians(tri["alpha"]))
        C_axial_from_W = tri["W"] * np.cos(np.radians(tri["beta"]))

        assert C_axial_from_C == pytest.approx(C_axial, rel=1e-10)
        assert C_axial_from_W == pytest.approx(C_axial, rel=1e-10)

    def test_euler_work(self) -> None:
        """Verify Euler turbomachinery equation: work = U * delta_C_theta.

        For a compressor stage with inlet alpha1=0 and exit alpha2=40:
          work = U * (C_theta2 - C_theta1)
        """
        U = 300.0
        C_axial = 150.0

        tri_in = compute_velocity_triangle(U, C_axial, alpha=0.0)
        tri_out = compute_velocity_triangle(U, C_axial, alpha=40.0)

        delta_C_theta = tri_out["C_theta"] - tri_in["C_theta"]
        work = U * delta_C_theta

        # This should equal the change in total enthalpy: delta_h0 = cp * delta_T0
        # Just verify the multiplication is correct
        assert work == pytest.approx(
            U * (C_axial * np.tan(np.radians(40)) - 0.0), rel=1e-10
        )
        assert work > 0, "Compressor should add work (positive)"

    def test_stator_no_work(self) -> None:
        """Stator (U=0) should do no work."""
        U = 0.0
        C_axial = 150.0
        alpha = 30.0

        tri = compute_velocity_triangle(U, C_axial, alpha)

        # With U=0, absolute and relative frames are the same
        assert tri["C"] == pytest.approx(tri["W"], rel=1e-10)
        assert tri["C_theta"] == pytest.approx(tri["W_theta"], rel=1e-10)
        assert tri["alpha"] == pytest.approx(tri["beta"], rel=1e-10)

    def test_known_textbook_example_1(self) -> None:
        """Textbook example: axial compressor stage.

        Given:
          U = 350 m/s
          C_axial = 175 m/s
          alpha_1 = 15 degrees (inlet)

        Known answers:
          C1 = 175 / cos(15) = 181.17
          C_theta1 = 175 * tan(15) = 46.89
          W_theta1 = 46.89 - 350 = -303.11
          W1 = sqrt(175^2 + 303.11^2) = 349.97
        """
        tri = compute_velocity_triangle(U=350.0, C_axial=175.0, alpha=15.0)

        assert tri["C"] == pytest.approx(181.17, abs=0.01)
        assert tri["C_theta"] == pytest.approx(46.89, abs=0.01)
        assert tri["W_theta"] == pytest.approx(-303.11, abs=0.01)
        assert tri["W"] == pytest.approx(
            np.sqrt(175.0**2 + 303.11**2), abs=0.1
        )

    def test_known_textbook_example_2(self) -> None:
        """Another textbook case: turbine stage.

        Given: U=400, C_axial=200, alpha=60 degrees (high swirl exit)

        C_theta = 200 * tan(60) = 346.41
        C = 200 / cos(60) = 400.0
        W_theta = 346.41 - 400 = -53.59
        W = sqrt(200^2 + 53.59^2) = 207.06
        """
        tri = compute_velocity_triangle(U=400.0, C_axial=200.0, alpha=60.0)

        assert tri["C"] == pytest.approx(400.0, abs=0.01)
        assert tri["C_theta"] == pytest.approx(346.41, abs=0.01)
        assert tri["W_theta"] == pytest.approx(-53.59, abs=0.01)
        assert tri["W"] == pytest.approx(207.06, abs=0.1)
