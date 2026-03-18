"""Centrifugal pump design for rocket turbopump and industrial applications.

Models an incompressible centrifugal pump using Euler turbomachinery
equations, specific speed correlations for efficiency, and NPSH
estimation from the Thoma cavitation parameter.

Physics:
    Head = P / (ρ g)
    Euler work:  w = U₂ C_θ₂ − U₁ C_θ₁  (with Wiesner slip)
    Specific speed:  Ns = ω √Q / (g H)^{3/4}
    NPSH_required from Thoma parameter:  σ = NPSH / H

References:
    Gülich, J.F., "Centrifugal Pumps", 3rd ed., Springer, 2014.
    Lakshminarayana, B., "Fluid Dynamics and Heat Transfer of Turbomachinery", 1996.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── Fluid property database ─────────────────────────

FLUIDS: dict[str, float] = {
    "LOX": 1141.0,
    "RP-1": 810.0,
    "LH2": 71.0,
    "water": 998.0,
    "kerosene": 810.0,
    "hydrazine": 1004.0,
    "MMH": 878.0,
}


# ── Design constants ─────────────────────────────────

# Head coefficient: U₂ = Ku × √(2gH/η). Ku ≈ 1.0 for centrifugal pumps
# (Gülich, Table 3.2). Range 0.9-1.1 depending on specific speed.
DEFAULT_HEAD_COEFFICIENT_KU = 1.0

# Default number of impeller blades for pumps (5-9 typical, 7 is standard).
DEFAULT_PUMP_BLADES = 7

# Default backsweep angle for pump impellers (degrees).
# Pumps typically use -20° to -30° backsweep.
DEFAULT_BACKSWEEP_DEG = -25.0


@dataclass
class PumpResult:
    """Result of centrifugal pump design."""

    head: float = 0.0              # m
    flow_rate: float = 0.0         # m³/s
    rpm: float = 0.0
    power_kW: float = 0.0
    efficiency: float = 0.0
    specific_speed: float = 0.0    # dimensionless (rad-based)
    impeller_diameter: float = 0.0 # m
    tip_speed: float = 0.0        # m/s
    npsh_required: float = 0.0    # m
    fluid_density: float = 998.0  # kg/m³
    fluid_name: str = "water"
    n_blades: int = 7

    def summary(self) -> str:
        """Human-readable pump summary."""
        lines = [
            "Centrifugal Pump Design",
            "=" * 50,
            f"  Fluid:              {self.fluid_name} (ρ={self.fluid_density:.0f} kg/m³)",
            f"  Head:               {self.head:.1f} m",
            f"  Flow Rate:          {self.flow_rate:.4f} m³/s ({self.flow_rate * 1000:.1f} L/s)",
            f"  RPM:                {self.rpm:.0f}",
            f"  Power:              {self.power_kW:.2f} kW",
            f"  Efficiency:         {self.efficiency:.4f} ({self.efficiency * 100:.1f}%)",
            f"  Specific Speed:     {self.specific_speed:.2f}",
            f"  Impeller Diameter:  {self.impeller_diameter:.4f} m ({self.impeller_diameter * 1000:.1f} mm)",
            f"  Tip Speed:          {self.tip_speed:.0f} m/s",
            f"  NPSH Required:      {self.npsh_required:.2f} m",
            f"  Number of Blades:   {self.n_blades}",
        ]
        return "\n".join(lines)


def centrifugal_pump(
    head: float,
    flow_rate: float,
    rpm: float,
    fluid_density: float = 998.0,
    fluid_name: str = "water",
) -> PumpResult:
    """Design a centrifugal pump from hydraulic requirements.

    Args:
        head: Required pump head (m).
        flow_rate: Volume flow rate (m³/s).
        rpm: Shaft speed (RPM).
        fluid_density: Fluid density (kg/m³). Overridden if fluid_name is in FLUIDS.
        fluid_name: Fluid name for lookup.

    Returns:
        PumpResult with all sizing parameters.

    Raises:
        ValueError: On invalid inputs.
    """
    if head <= 0:
        raise ValueError(f"head must be positive, got {head}")
    if flow_rate <= 0:
        raise ValueError(f"flow_rate must be positive, got {flow_rate}")
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")

    # Lookup fluid density
    if fluid_name in FLUIDS:
        fluid_density = FLUIDS[fluid_name]

    g = 9.80665
    omega = rpm * 2.0 * math.pi / 60.0

    # Specific speed (dimensionless, rad-based)
    Ns = omega * math.sqrt(flow_rate) / (g * head) ** 0.75

    # Efficiency from specific speed correlation (Gülich)
    # Peak efficiency near Ns ~ 0.8-1.2
    # eta = 1 - 0.2 / Ns^0.5 (simplified Cordier-based)
    if Ns > 0.01:
        eta = min(0.92, max(0.45, 1.0 - 0.2 / Ns ** 0.5))
    else:
        eta = 0.45

    # Shaft power
    P_hydraulic = fluid_density * g * head * flow_rate
    P_shaft = P_hydraulic / eta
    power_kW = P_shaft / 1000.0

    # Impeller sizing
    # Euler head: H = η_h × (U₂² / g) × (1 - φ₂ cot β₂) × σ_slip
    # Simplified: U₂ = Ku × sqrt(2 g H), Ku ≈ 1.0-1.1 for centrifugal pumps
    U2 = DEFAULT_HEAD_COEFFICIENT_KU * math.sqrt(2.0 * g * head / eta)
    D2 = 2.0 * U2 / omega if omega > 0 else 0.0  # impeller diameter

    # Slip factor (Wiesner)
    n_blades = DEFAULT_PUMP_BLADES
    try:
        from .centrifugal import wiesner_slip_factor
        sigma_slip = wiesner_slip_factor(n_blades, DEFAULT_BACKSWEEP_DEG)
    except ImportError:
        sigma_slip = 1.0 - math.sqrt(math.sin(math.radians(70))) / n_blades ** 0.7

    # NPSH required (Thoma parameter correlation)
    # sigma_Th = NPSH / H, sigma_Th ≈ 0.3 × (Ns)^(4/3) (Stepanoff)
    sigma_thoma = 0.3 * abs(Ns) ** (4.0 / 3.0) if Ns > 0 else 0.1
    sigma_thoma = max(sigma_thoma, 0.02)  # minimum
    npsh_required = sigma_thoma * head

    tip_speed = U2

    return PumpResult(
        head=head,
        flow_rate=flow_rate,
        rpm=rpm,
        power_kW=power_kW,
        efficiency=eta,
        specific_speed=Ns,
        impeller_diameter=D2,
        tip_speed=tip_speed,
        npsh_required=npsh_required,
        fluid_density=fluid_density,
        fluid_name=fluid_name,
        n_blades=n_blades,
    )
