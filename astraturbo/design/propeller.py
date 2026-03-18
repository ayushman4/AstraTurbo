"""Propeller / rotor design for drones and eVTOL.

Uses actuator disk momentum theory with empirical profile drag corrections
to compute thrust, power, efficiency, and blade geometry.

Note: This is a preliminary design tool using momentum theory, NOT a full
blade-element momentum (BEM) solver. Profile drag is accounted for via
empirical correction factors rather than computed from airfoil polars.

Supports both hover (V=0) and forward flight conditions.

Physics:
    Actuator disk:  T = 2 ρ A V_i (V + V_i)
    Hover induced velocity:  V_i = sqrt(T / (2 ρ A))
    Figure of Merit:  FM = T^{3/2} / (sqrt(2 ρ A) × P)
    Advance ratio:  J = V / (n D)
    CT = T / (ρ n² D⁴),  CP = P / (ρ n³ D⁵)

References:
    Leishman, J.G., "Principles of Helicopter Aerodynamics", 2nd ed., 2006.
    Johnson, W., "Helicopter Theory", Dover, 1994.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── Empirical constants (documented, configurable via function args) ──

# Profile drag correction factors applied to ideal induced power.
# These account for blade profile drag, tip losses, and swirl losses
# that are not captured by simple momentum theory.
# Ref: Leishman Ch. 2 — typical rotors have FM 0.7-0.85, implying
# P_actual / P_ideal ≈ 1.12-1.20.
HOVER_PROFILE_DRAG_FACTOR = 1.15    # ~15% extra power in hover (Leishman, Table 2.2)
FORWARD_PROFILE_DRAG_FACTOR = 1.12  # ~12% in forward flight (lower due to reduced induced losses)

# Blade geometry estimation constants for preliminary sizing.
# chord_75 / D ≈ 0.10 is a typical value for small UAV propellers.
# alpha_design ≈ 4° gives Cl ≈ 0.5-0.7 for typical airfoils at low Re.
CHORD_TO_DIAMETER_RATIO = 0.10   # c/D at 75% radius
DESIGN_ANGLE_OF_ATTACK = 4.0     # degrees, for Cl ≈ 0.6


@dataclass
class PropellerResult:
    """Result of propeller/rotor design."""

    thrust: float = 0.0            # N
    power: float = 0.0             # W
    efficiency: float = 0.0        # propulsive efficiency (forward flight)
    figure_of_merit: float = 0.0   # hover FM
    diameter: float = 0.0          # m
    rpm: float = 0.0
    n_blades: int = 2
    advance_ratio: float = 0.0    # J
    CT: float = 0.0               # thrust coefficient
    CP: float = 0.0               # power coefficient
    blade_angle_75: float = 0.0   # blade pitch at 75% radius (deg)
    solidity: float = 0.0
    disk_loading: float = 0.0     # N/m²
    tip_speed: float = 0.0        # m/s
    tip_mach: float = 0.0

    def summary(self) -> str:
        """Human-readable propeller summary."""
        lines = [
            "Propeller / Rotor Design",
            "=" * 50,
            f"  Thrust:         {self.thrust:.1f} N",
            f"  Power:          {self.power:.0f} W ({self.power / 1000:.1f} kW)",
            f"  Diameter:       {self.diameter:.3f} m",
            f"  RPM:            {self.rpm:.0f}",
            f"  Blades:         {self.n_blades}",
            f"  Advance Ratio:  {self.advance_ratio:.4f}",
            f"  CT:             {self.CT:.6f}",
            f"  CP:             {self.CP:.6f}",
            f"  Disk Loading:   {self.disk_loading:.1f} N/m²",
            f"  Tip Speed:      {self.tip_speed:.1f} m/s",
            f"  Tip Mach:       {self.tip_mach:.3f}",
            f"  Blade Angle @75%: {self.blade_angle_75:.1f}°",
            f"  Solidity:       {self.solidity:.4f}",
        ]
        if self.advance_ratio < 0.01:
            lines.append(f"  Figure of Merit: {self.figure_of_merit:.4f} (hover)")
        else:
            lines.append(f"  Propulsive Eff: {self.efficiency:.4f}")
        return "\n".join(lines)


def propeller_design(
    thrust_required: float,
    n_blades: int,
    diameter: float,
    rpm: float,
    V_flight: float = 0.0,
    altitude: float = 0.0,
) -> PropellerResult:
    """Design a propeller/rotor from thrust requirements.

    Args:
        thrust_required: Required thrust (N).
        n_blades: Number of blades.
        diameter: Propeller diameter (m).
        rpm: Rotational speed (RPM).
        V_flight: Forward flight speed (m/s). 0 = hover.
        altitude: Altitude (m) for atmospheric conditions.

    Returns:
        PropellerResult with all performance and geometry parameters.

    Raises:
        ValueError: On invalid inputs.
    """
    if thrust_required <= 0:
        raise ValueError(f"thrust_required must be positive, got {thrust_required}")
    if n_blades < 1:
        raise ValueError(f"n_blades must be >= 1, got {n_blades}")
    if diameter <= 0:
        raise ValueError(f"diameter must be positive, got {diameter}")
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    if V_flight < 0:
        raise ValueError(f"V_flight must be >= 0, got {V_flight}")

    # Atmospheric conditions
    from .engine_cycle import standard_atmosphere
    T_amb, P_amb, rho = standard_atmosphere(altitude)

    # Geometry
    R = diameter / 2.0
    A_disk = math.pi * R ** 2
    n_rps = rpm / 60.0           # revolutions per second
    omega = rpm * 2.0 * math.pi / 60.0

    # Tip speed
    tip_speed = omega * R
    a_sound = math.sqrt(1.4 * 287.0 * T_amb)
    # Effective tip Mach includes forward flight component
    V_tip_eff = math.sqrt(tip_speed ** 2 + V_flight ** 2)
    tip_mach = V_tip_eff / a_sound

    # Disk loading
    disk_loading = thrust_required / A_disk

    # Actuator disk theory
    T = thrust_required
    if V_flight < 0.1:
        # Hover: T = 2 ρ A Vi²
        Vi = math.sqrt(T / (2.0 * rho * A_disk))
        P_ideal = T * Vi
        # Blade element correction: ~15% extra power for profile drag
        P = P_ideal * HOVER_PROFILE_DRAG_FACTOR
        efficiency = 0.0  # not defined in hover
        fm = T * Vi / P  # figure of merit
    else:
        # Forward flight: momentum theory
        # T = 2 ρ A Vi (V + Vi)
        # Solve quadratic for Vi: 2 ρ A Vi² + 2 ρ A V Vi - T = 0
        a_coeff = 2.0 * rho * A_disk
        b_coeff = 2.0 * rho * A_disk * V_flight
        c_coeff = -T
        disc = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
        Vi = (-b_coeff + math.sqrt(max(disc, 0.0))) / (2.0 * a_coeff)
        P_ideal = T * (V_flight + Vi)
        P = P_ideal * FORWARD_PROFILE_DRAG_FACTOR
        efficiency = T * V_flight / P if P > 0 else 0.0
        fm = 0.0  # FM not used in forward flight

    # Non-dimensional coefficients
    CT = T / (rho * n_rps ** 2 * diameter ** 4) if n_rps > 0 else 0.0
    CP = P / (rho * n_rps ** 3 * diameter ** 5) if n_rps > 0 else 0.0

    # Advance ratio
    J = V_flight / (n_rps * diameter) if n_rps * diameter > 0 else 0.0

    # Blade angle at 75% radius (simple estimate from inflow angle)
    r75 = 0.75 * R
    V_rot_75 = omega * r75
    V_axial = V_flight + Vi if V_flight >= 0.1 else Vi
    phi_75 = math.degrees(math.atan2(V_axial, V_rot_75))
    # Add design angle of attack for typical operating Cl
    blade_angle_75 = phi_75 + DESIGN_ANGLE_OF_ATTACK

    # Solidity: chord estimated from chord-to-diameter ratio at 75% radius
    chord_75 = CHORD_TO_DIAMETER_RATIO * diameter
    solidity = n_blades * chord_75 / (math.pi * r75) if r75 > 0 else 0.0

    return PropellerResult(
        thrust=T,
        power=P,
        efficiency=efficiency,
        figure_of_merit=fm,
        diameter=diameter,
        rpm=rpm,
        n_blades=n_blades,
        advance_ratio=J,
        CT=CT,
        CP=CP,
        blade_angle_75=blade_angle_75,
        solidity=solidity,
        disk_loading=disk_loading,
        tip_speed=tip_speed,
        tip_mach=tip_mach,
    )
