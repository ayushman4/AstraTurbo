"""Electric motor sizing for eVTOL and drone propulsion.

Models BLDC and PMSM motors from shaft power, RPM, and voltage.
Computes torque, motor constant (Kv), weight, efficiency, and
thermal margin.

Physics:
    torque = P / omega
    Kv = RPM / V  (motor velocity constant)
    Efficiency: parabolic model  eta = eta_peak * (1 - (1 - load)^2 * 0.15)
    Weight from empirical power-to-weight: 7 kW/kg (BLDC), 5 kW/kg (PMSM)
    Thermal margin from I_max rating

References:
    Gundlach, J., "Designing Unmanned Aircraft Systems", AIAA, 2012.
    McDonald, R.A., "Electric Propulsion Modeling for Conceptual Aircraft Design",
    AIAA 2014-0536.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Power-to-weight ratios (kW/kg) for different motor types
_POWER_DENSITY: dict[str, float] = {
    "BLDC": 7.0,
    "PMSM": 5.0,
}


@dataclass
class ElectricMotorResult:
    """Result of electric motor sizing."""

    shaft_power: float = 0.0       # W
    torque: float = 0.0            # N·m
    rpm: float = 0.0
    voltage: float = 0.0           # V
    current: float = 0.0           # A
    efficiency: float = 0.0
    weight_kg: float = 0.0
    power_density: float = 0.0     # kW/kg
    motor_constant_kv: float = 0.0 # RPM/V
    thermal_margin: float = 0.0    # fraction of I_max remaining
    motor_type: str = "BLDC"

    def summary(self) -> str:
        """Human-readable motor summary."""
        lines = [
            f"Electric Motor Sizing — {self.motor_type}",
            "=" * 50,
            f"  Shaft Power:    {self.shaft_power:.0f} W ({self.shaft_power / 1000:.1f} kW)",
            f"  Torque:         {self.torque:.3f} N·m",
            f"  RPM:            {self.rpm:.0f}",
            f"  Voltage:        {self.voltage:.1f} V",
            f"  Current:        {self.current:.1f} A",
            f"  Efficiency:     {self.efficiency:.4f} ({self.efficiency * 100:.1f}%)",
            f"  Weight:         {self.weight_kg:.2f} kg",
            f"  Power Density:  {self.power_density:.1f} kW/kg",
            f"  Kv:             {self.motor_constant_kv:.1f} RPM/V",
            f"  Thermal Margin: {self.thermal_margin:.2f}",
        ]
        return "\n".join(lines)


def electric_motor(
    shaft_power: float,
    rpm: float,
    voltage: float,
    motor_type: str = "BLDC",
    eta_peak: float = 0.92,
    load_fraction: float = 1.0,
) -> ElectricMotorResult:
    """Size an electric motor from shaft power requirements.

    Args:
        shaft_power: Required shaft power (W).
        rpm: Motor speed (RPM).
        voltage: Supply voltage (V).
        motor_type: "BLDC" or "PMSM".
        eta_peak: Peak efficiency at full load (default 0.92).
        load_fraction: Operating point as fraction of rated power (0-1).

    Returns:
        ElectricMotorResult with all sizing parameters.

    Raises:
        ValueError: On invalid inputs.
    """
    if shaft_power <= 0:
        raise ValueError(f"shaft_power must be positive, got {shaft_power}")
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    if voltage <= 0:
        raise ValueError(f"voltage must be positive, got {voltage}")
    if motor_type not in _POWER_DENSITY:
        raise ValueError(
            f"motor_type must be one of {list(_POWER_DENSITY.keys())}, got '{motor_type}'"
        )
    if not (0.5 <= eta_peak <= 0.99):
        raise ValueError(f"eta_peak must be in [0.5, 0.99], got {eta_peak}")
    if not (0.0 < load_fraction <= 1.5):
        raise ValueError(f"load_fraction must be in (0, 1.5], got {load_fraction}")

    # Torque
    omega = rpm * 2.0 * math.pi / 60.0   # rad/s
    torque = shaft_power / omega

    # Motor constant
    kv = rpm / voltage  # RPM/V

    # Efficiency: parabolic model — drops at partial load
    eta = eta_peak * (1.0 - (1.0 - load_fraction) ** 2 * 0.15)
    eta = max(eta, 0.5)

    # Electrical power and current
    P_elec = shaft_power / eta
    current = P_elec / voltage

    # Weight from power density
    pw = _POWER_DENSITY[motor_type]  # kW/kg
    weight_kg = (shaft_power / 1000.0) / pw

    # Thermal margin: assume I_max = 1.5 × rated current at peak power
    I_max = 1.5 * (shaft_power / (eta_peak * voltage))
    thermal_margin = max(1.0 - current / I_max, 0.0)

    return ElectricMotorResult(
        shaft_power=shaft_power,
        torque=torque,
        rpm=rpm,
        voltage=voltage,
        current=current,
        efficiency=eta,
        weight_kg=weight_kg,
        power_density=pw,
        motor_constant_kv=kv,
        thermal_margin=thermal_margin,
        motor_type=motor_type,
    )
