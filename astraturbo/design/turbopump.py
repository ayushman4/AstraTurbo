"""Turbopump assembly for rocket engines.

Couples a centrifugal pump with an axial turbine through a common
shaft to model the integrated propellant feed system.

Physics:
    - Pump power from centrifugal_pump()
    - Turbine power from meanline_turbine()
    - Shaft power balance: P_turbine × η_mech ≥ P_pump
    - Power margin = (P_turbine × η_mech − P_pump) / P_pump

Cycle types:
    - gas_generator: Turbine driven by ~3% bleed from a separate GG.
      Low turbine mass flow, moderate expansion ratio.
      Examples: Merlin (SpaceX), F-1 (Saturn V).
    - staged_combustion: All propellant flows through turbine (preburner).
      High turbine mass flow (100% of pump flow), low expansion ratio,
      higher chamber pressure.
      Examples: RD-180, Raptor (partial), RS-25 (SSME).
    - expander: Turbine driven by heated propellant (no combustion).
      Lower turbine inlet temperature (~300-600 K), limited power.
      Examples: RL-10, Vinci.

References:
    Huzel, D.K. & Huang, D.H., "Modern Engineering for Design of
    Liquid-Propellant Rocket Engines", AIAA, 1992.
    Sutton, G.P., "Rocket Propulsion Elements", 9th ed., Wiley, 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .pump import PumpResult, centrifugal_pump
from .turbine import TurbineResult, meanline_turbine
from .meanline import GasProperties


# ── Cycle-type constants ────────────────────────────

VALID_CYCLE_TYPES = ("gas_generator", "staged_combustion", "expander")

# Turbine mass flow as fraction of pump propellant mass flow, by cycle type.
# GG: ~3% bleed (Huzel & Huang, Table 6-1)
# Staged combustion: 100% of propellant mass (all flow through preburner)
# Expander: ~100% of fuel-side flow (for fuel-side expander)
_TURBINE_MASS_FLOW_FRACTION = {
    "gas_generator": 0.03,
    "staged_combustion": 1.0,
    "expander": 1.0,
}

# Turbine inlet temperature caps by cycle type (K).
# GG: limited by turbine materials (~700-1000 K)
# Staged combustion: similar to GG (fuel-rich preburner ~700-900 K)
# Expander: limited by regenerative heating (~300-600 K)
_MAX_TURBINE_TEMP = {
    "gas_generator": None,        # user-specified
    "staged_combustion": None,    # user-specified
    "expander": 600.0,            # physical limit from regen heating
}

# Turbine polytropic efficiency estimate by cycle type.
_ETA_POLY_TURBINE = {
    "gas_generator": 0.85,
    "staged_combustion": 0.82,    # higher mass flow, more losses
    "expander": 0.80,             # lower pressure ratio, less optimal
}

# Expansion ratio limits by cycle type.
_ER_MIN = {"gas_generator": 1.2, "staged_combustion": 1.1, "expander": 1.1}
_ER_MAX = {"gas_generator": 20.0, "staged_combustion": 5.0, "expander": 3.0}

# Temp drop ratio physical limit (avoids unphysical turbine sizing).
TEMP_DROP_RATIO_LIMIT = 0.8


@dataclass
class TurbopumpResult:
    """Complete turbopump assembly result."""

    pump: PumpResult | None = None
    turbine: TurbineResult | None = None

    shaft_rpm: float = 0.0
    shaft_power: float = 0.0           # W (turbine output)
    pump_power: float = 0.0            # W
    turbine_power: float = 0.0         # W
    power_margin: float = 0.0          # fraction
    mechanical_efficiency: float = 0.97
    overall_efficiency: float = 0.0
    cycle_type: str = "gas_generator"

    def summary(self) -> str:
        """Human-readable turbopump summary."""
        lines = [
            f"Turbopump Assembly — {self.cycle_type.replace('_', ' ').title()}",
            "=" * 50,
            f"  Shaft RPM:            {self.shaft_rpm:.0f}",
            f"  Pump Power:           {self.pump_power / 1000:.2f} kW",
            f"  Turbine Power:        {self.turbine_power / 1000:.2f} kW",
            f"  Shaft Power:          {self.shaft_power / 1000:.2f} kW",
            f"  Power Margin:         {self.power_margin:.4f} ({self.power_margin * 100:.1f}%)",
            f"  Mechanical Eff:       {self.mechanical_efficiency:.4f}",
            f"  Overall Eff:          {self.overall_efficiency:.4f}",
        ]
        if self.pump:
            lines.append("")
            lines.append("  Pump:")
            lines.append(f"    Fluid:              {self.pump.fluid_name} (ρ={self.pump.fluid_density:.0f} kg/m³)")
            lines.append(f"    Head:               {self.pump.head:.1f} m")
            lines.append(f"    Flow Rate:          {self.pump.flow_rate:.4f} m³/s")
            lines.append(f"    Efficiency:         {self.pump.efficiency:.4f}")
            lines.append(f"    Impeller Diameter:  {self.pump.impeller_diameter * 1000:.1f} mm")
        if self.turbine:
            lines.append("")
            lines.append("  Turbine:")
            lines.append(f"    Expansion Ratio:    {self.turbine.overall_expansion_ratio:.2f}")
            lines.append(f"    Stages:             {self.turbine.n_stages}")
            lines.append(f"    Efficiency:         {self.turbine.overall_efficiency:.4f}")
            lines.append(f"    Work:               {self.turbine.total_work:.0f} J/kg")

        return "\n".join(lines)


def turbopump(
    pump_head: float,
    pump_flow_rate: float,
    fluid_density: float,
    turbine_inlet_temp: float,
    turbine_inlet_pressure: float,
    rpm: float = 30000.0,
    cycle_type: str = "gas_generator",
    eta_mech: float = 0.97,
    fluid_name: str = "LOX",
    # Turbine geometry
    turbine_r_hub: float = 0.03,
    turbine_r_tip: float = 0.06,
    turbine_mass_flow: float | None = None,
    gamma_hot: float = 1.25,
    cp_hot: float = 1500.0,
) -> TurbopumpResult:
    """Design a turbopump assembly coupling pump and turbine.

    Args:
        pump_head: Required pump head (m).
        pump_flow_rate: Pump volume flow rate (m³/s).
        fluid_density: Pump fluid density (kg/m³).
        turbine_inlet_temp: Turbine inlet total temperature (K).
        turbine_inlet_pressure: Turbine inlet total pressure (Pa).
        rpm: Common shaft speed (RPM).
        cycle_type: "gas_generator", "staged_combustion", or "expander".
        eta_mech: Mechanical efficiency of shaft/bearings.
        fluid_name: Pump fluid name.
        turbine_r_hub: Turbine hub radius (m).
        turbine_r_tip: Turbine tip radius (m).
        turbine_mass_flow: Turbine gas mass flow (kg/s).
            If None, estimated from pump flow as 3% of pump mass flow
            (typical gas generator bleed).
        gamma_hot: Turbine gas ratio of specific heats.
        cp_hot: Turbine gas specific heat (J/(kg·K)).

    Returns:
        TurbopumpResult with pump, turbine, and shaft balance.

    Raises:
        ValueError: On invalid inputs.
    """
    if pump_head <= 0:
        raise ValueError(f"pump_head must be positive, got {pump_head}")
    if pump_flow_rate <= 0:
        raise ValueError(f"pump_flow_rate must be positive, got {pump_flow_rate}")
    if fluid_density <= 0:
        raise ValueError(f"fluid_density must be positive, got {fluid_density}")
    if turbine_inlet_temp < 300:
        raise ValueError(f"turbine_inlet_temp must be >= 300 K, got {turbine_inlet_temp}")
    if turbine_inlet_pressure <= 0:
        raise ValueError(f"turbine_inlet_pressure must be positive, got {turbine_inlet_pressure}")
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    if cycle_type not in VALID_CYCLE_TYPES:
        raise ValueError(
            f"cycle_type must be one of {VALID_CYCLE_TYPES}, got '{cycle_type}'"
        )

    # Apply expander cycle temperature cap
    T_turb_in = turbine_inlet_temp
    max_temp = _MAX_TURBINE_TEMP.get(cycle_type)
    if max_temp is not None and T_turb_in > max_temp:
        T_turb_in = max_temp

    # 1. Design the pump
    pump_result = centrifugal_pump(
        head=pump_head,
        flow_rate=pump_flow_rate,
        rpm=rpm,
        fluid_density=fluid_density,
        fluid_name=fluid_name,
    )
    P_pump = pump_result.power_kW * 1000.0  # W

    # 2. Determine turbine mass flow based on cycle type
    if turbine_mass_flow is None:
        pump_mass_flow = pump_flow_rate * fluid_density
        bleed_fraction = _TURBINE_MASS_FLOW_FRACTION[cycle_type]
        turbine_mass_flow = max(pump_mass_flow * bleed_fraction, 0.1)

    # 3. Determine turbine expansion ratio from power requirement
    P_required = P_pump / eta_mech
    eta_turb_est = _ETA_POLY_TURBINE[cycle_type]
    work_specific_req = P_required / turbine_mass_flow
    temp_drop_ratio = work_specific_req / (cp_hot * T_turb_in * eta_turb_est)
    temp_drop_ratio = min(temp_drop_ratio, TEMP_DROP_RATIO_LIMIT)
    if temp_drop_ratio > 0.01:
        ER = (1.0 - temp_drop_ratio) ** (-gamma_hot / (gamma_hot - 1.0))
    else:
        ER = _ER_MIN[cycle_type]

    ER = max(ER, _ER_MIN[cycle_type])
    ER = min(ER, _ER_MAX[cycle_type])

    # 4. Design the turbine
    gas_hot = GasProperties(
        gamma=gamma_hot,
        cp=cp_hot,
        R=cp_hot * (gamma_hot - 1.0) / gamma_hot,
    )
    turbine_result = meanline_turbine(
        overall_expansion_ratio=ER,
        mass_flow=turbine_mass_flow,
        rpm=rpm,
        r_hub=turbine_r_hub,
        r_tip=turbine_r_tip,
        eta_poly=eta_turb_est,
        gas=gas_hot,
        T_inlet=T_turb_in,
        P_inlet=turbine_inlet_pressure,
    )

    # 5. Compute shaft balance
    P_turbine = turbine_result.total_work * turbine_mass_flow  # W
    shaft_power = P_turbine * eta_mech
    power_margin = (shaft_power - P_pump) / P_pump if P_pump > 0 else 0.0

    # Overall efficiency: pump hydraulic power / turbine gas power
    P_hydraulic = fluid_density * 9.80665 * pump_head * pump_flow_rate
    overall_eff = P_hydraulic / P_turbine if P_turbine > 0 else 0.0

    return TurbopumpResult(
        pump=pump_result,
        turbine=turbine_result,
        shaft_rpm=rpm,
        shaft_power=shaft_power,
        pump_power=P_pump,
        turbine_power=P_turbine,
        power_margin=power_margin,
        mechanical_efficiency=eta_mech,
        overall_efficiency=overall_eff,
        cycle_type=cycle_type,
    )
