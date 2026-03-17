"""Velocity triangle calculations for turbomachinery.

Computes the fundamental velocity relationships at the inlet and outlet
of each blade row:
    U = blade speed (radius x omega)
    C = absolute velocity (stationary frame)
    W = relative velocity (rotating frame)
    V_axial = axial component (same in both frames)
    alpha = absolute flow angle
    beta = relative flow angle

Vector relationship: C = W + U  (vectorially)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityTriangle:
    """Velocity triangle at a single station (inlet or outlet of a blade row).

    All angles in radians. Positive angles measured from axial direction
    toward the direction of blade rotation.
    """

    U: float           # Blade speed (m/s)
    C_axial: float     # Axial velocity component (m/s)
    C_theta: float     # Tangential (swirl) component of absolute velocity (m/s)

    @property
    def C(self) -> float:
        """Absolute velocity magnitude."""
        return math.sqrt(self.C_axial**2 + self.C_theta**2)

    @property
    def W_theta(self) -> float:
        """Tangential component of relative velocity."""
        return self.C_theta - self.U

    @property
    def W_axial(self) -> float:
        """Axial component of relative velocity (same as C_axial)."""
        return self.C_axial

    @property
    def W(self) -> float:
        """Relative velocity magnitude."""
        return math.sqrt(self.W_axial**2 + self.W_theta**2)

    @property
    def alpha(self) -> float:
        """Absolute flow angle (radians from axial)."""
        return math.atan2(self.C_theta, self.C_axial)

    @property
    def alpha_deg(self) -> float:
        """Absolute flow angle in degrees."""
        return math.degrees(self.alpha)

    @property
    def beta(self) -> float:
        """Relative flow angle (radians from axial)."""
        return math.atan2(self.W_theta, self.W_axial)

    @property
    def beta_deg(self) -> float:
        """Relative flow angle in degrees."""
        return math.degrees(self.beta)

    @property
    def C_mach(self) -> float:
        """Absolute Mach number (requires set_gas_properties first)."""
        return 0.0  # Requires speed of sound

    def to_dict(self) -> dict:
        """Return all quantities as a dictionary."""
        return {
            "U": self.U,
            "C_axial": self.C_axial,
            "C_theta": self.C_theta,
            "C": self.C,
            "W_theta": self.W_theta,
            "W_axial": self.W_axial,
            "W": self.W,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
        }


def compute_triangle_from_angles(
    U: float,
    C_axial: float,
    alpha: float,
) -> VelocityTriangle:
    """Compute velocity triangle from blade speed, axial velocity, and absolute angle.

    Args:
        U: Blade speed (m/s).
        C_axial: Axial velocity (m/s).
        alpha: Absolute flow angle (radians from axial).

    Returns:
        Complete VelocityTriangle.
    """
    C_theta = C_axial * math.tan(alpha)
    return VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta)


def compute_triangle_from_beta(
    U: float,
    C_axial: float,
    beta: float,
) -> VelocityTriangle:
    """Compute velocity triangle from blade speed, axial velocity, and relative angle.

    Args:
        U: Blade speed (m/s).
        C_axial: Axial velocity (m/s).
        beta: Relative flow angle (radians from axial).

    Returns:
        Complete VelocityTriangle.
    """
    W_theta = C_axial * math.tan(beta)
    C_theta = W_theta + U
    return VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta)


@dataclass
class BladeRowTriangles:
    """Inlet and outlet velocity triangles for a blade row."""

    inlet: VelocityTriangle
    outlet: VelocityTriangle

    @property
    def delta_C_theta(self) -> float:
        """Change in tangential velocity (swirl) across the row."""
        return self.outlet.C_theta - self.inlet.C_theta

    @property
    def work_per_unit_mass(self) -> float:
        """Specific work (J/kg) via Euler equation: w = U * delta_C_theta."""
        U = (self.inlet.U + self.outlet.U) / 2.0
        return U * self.delta_C_theta

    @property
    def flow_turning(self) -> float:
        """Flow turning angle in the relative frame (radians)."""
        return self.inlet.beta - self.outlet.beta

    @property
    def flow_turning_deg(self) -> float:
        """Flow turning in degrees."""
        return math.degrees(self.flow_turning)

    @property
    def de_haller_ratio(self) -> float:
        """De Haller number: W_out/W_in. Should be > 0.72 to avoid separation."""
        if self.inlet.W < 1e-10:
            return 0.0
        return self.outlet.W / self.inlet.W

    def summary(self) -> str:
        """Return a text summary of the velocity triangles."""
        lines = [
            "Inlet:",
            f"  U = {self.inlet.U:.1f} m/s",
            f"  C = {self.inlet.C:.1f} m/s  (alpha = {self.inlet.alpha_deg:.1f} deg)",
            f"  W = {self.inlet.W:.1f} m/s  (beta  = {self.inlet.beta_deg:.1f} deg)",
            "Outlet:",
            f"  U = {self.outlet.U:.1f} m/s",
            f"  C = {self.outlet.C:.1f} m/s  (alpha = {self.outlet.alpha_deg:.1f} deg)",
            f"  W = {self.outlet.W:.1f} m/s  (beta  = {self.outlet.beta_deg:.1f} deg)",
            "Performance:",
            f"  Work = {self.work_per_unit_mass:.0f} J/kg",
            f"  Turning = {self.flow_turning_deg:.1f} deg",
            f"  De Haller = {self.de_haller_ratio:.3f}",
        ]
        return "\n".join(lines)
