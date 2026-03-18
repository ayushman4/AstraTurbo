"""Engine cycle solver for complete gas turbine analysis.

Connects compressor → combustor → turbine → nozzle to model a full engine.
Supports turbojet (thrust) and turboshaft (shaft power) configurations.

Usage:
    from astraturbo.design.engine_cycle import engine_cycle

    result = engine_cycle(
        engine_type="turbojet",
        overall_pressure_ratio=20.0,
        turbine_inlet_temp=1700.0,
        mass_flow=20.0,
        rpm=15000,
        r_hub=0.15,
        r_tip=0.30,
    )
    print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

from .meanline import (
    GasProperties,
    StationConditions,
    MeanlineResult,
    meanline_compressor,
)
from .centrifugal import CentrifugalResult, centrifugal_compressor
from .turbine import TurbineResult, meanline_turbine


# ── ISA Standard Atmosphere ──────────────────────────


def standard_atmosphere(altitude: float) -> tuple[float, float, float]:
    """ISA standard atmosphere model.

    Args:
        altitude: Geometric altitude in meters (0 to 47000 m).

    Returns:
        (T, P, rho) — temperature (K), pressure (Pa), density (kg/m³).
    """
    T0 = 288.15        # Sea-level temperature (K)
    P0 = 101325.0       # Sea-level pressure (Pa)
    g = 9.80665         # Gravity (m/s²)
    R = 287.0528        # Gas constant for air (J/(kg·K))
    lapse = 0.0065      # Lapse rate (K/m) in troposphere

    if altitude < 0:
        altitude = 0.0

    if altitude <= 11000.0:
        # Troposphere: linear temperature decrease
        T = T0 - lapse * altitude
        P = P0 * (T / T0) ** (g / (R * lapse))
    else:
        # Stratosphere: isothermal layer (11 km – 20 km)
        T_trop = T0 - lapse * 11000.0   # 216.65 K
        P_trop = P0 * (T_trop / T0) ** (g / (R * lapse))
        T = T_trop
        P = P_trop * math.exp(-g * (altitude - 11000.0) / (R * T_trop))

    rho = P / (R * T)
    return T, P, rho


# ── Component Result Dataclasses ─────────────────────


@dataclass
class InletResult:
    """Result from ram inlet model."""
    P_total_out: float = 0.0
    T_total_out: float = 0.0
    eta_inlet: float = 0.97
    mach_flight: float = 0.0


@dataclass
class CombustorResult:
    """Result from combustor model."""
    T_out: float = 0.0
    P_out: float = 0.0
    fuel_air_ratio: float = 0.0
    fuel_flow: float = 0.0
    eta_combustor: float = 0.99
    dp_fraction: float = 0.04


@dataclass
class NozzleResult:
    """Result from convergent nozzle model."""
    V_exit: float = 0.0
    P_exit: float = 0.0
    T_exit: float = 0.0
    A_exit: float = 0.0
    is_choked: bool = False
    gross_thrust: float = 0.0
    mach_exit: float = 0.0


# ── Component Physics Functions ──────────────────────


def inlet_model(
    T_ambient: float,
    P_ambient: float,
    mach_flight: float,
    eta_inlet: float = 0.97,
    gamma: float = 1.4,
) -> InletResult:
    """Ram inlet: convert flight Mach to stagnation conditions.

    Args:
        T_ambient: Static ambient temperature (K).
        P_ambient: Static ambient pressure (Pa).
        mach_flight: Flight Mach number.
        eta_inlet: Inlet total pressure recovery.
        gamma: Ratio of specific heats.

    Returns:
        InletResult with total conditions at compressor face.
    """
    M2 = mach_flight ** 2
    gm1_half = (gamma - 1.0) / 2.0
    T02 = T_ambient * (1.0 + gm1_half * M2)
    P02 = P_ambient * eta_inlet * (1.0 + gm1_half * M2) ** (gamma / (gamma - 1.0))
    return InletResult(
        P_total_out=P02,
        T_total_out=T02,
        eta_inlet=eta_inlet,
        mach_flight=mach_flight,
    )


def combustor_model(
    P_in: float,
    T_in: float,
    T_target_out: float,
    mass_flow_air: float,
    eta_combustor: float = 0.99,
    dp_fraction: float = 0.04,
    Q_fuel: float = 43e6,
    gamma_hot: float = 1.33,
    cp_hot: float = 1150.0,
) -> CombustorResult:
    """Simple combustor: pressure drop + fuel-air ratio from energy balance.

    Args:
        P_in: Inlet total pressure (Pa).
        T_in: Inlet total temperature (K).
        T_target_out: Target turbine inlet temperature (K).
        mass_flow_air: Air mass flow (kg/s).
        eta_combustor: Combustion efficiency.
        dp_fraction: Fractional total pressure loss.
        Q_fuel: Fuel lower heating value (J/kg).
        gamma_hot: Ratio of specific heats for hot gas.
        cp_hot: Specific heat for hot gas (J/(kg·K)).

    Returns:
        CombustorResult with outlet conditions and fuel flow.
    """
    P_out = P_in * (1.0 - dp_fraction)
    f = cp_hot * (T_target_out - T_in) / (Q_fuel * eta_combustor - cp_hot * T_target_out)
    f = max(f, 0.0)
    fuel_flow = f * mass_flow_air
    return CombustorResult(
        T_out=T_target_out,
        P_out=P_out,
        fuel_air_ratio=f,
        fuel_flow=fuel_flow,
        eta_combustor=eta_combustor,
        dp_fraction=dp_fraction,
    )


def nozzle_model(
    P_in: float,
    T_in: float,
    P_ambient: float,
    mass_flow_total: float,
    gamma: float = 1.33,
    cp: float = 1150.0,
) -> NozzleResult:
    """Convergent nozzle with choke check.

    Args:
        P_in: Inlet total pressure (Pa).
        T_in: Inlet total temperature (K).
        P_ambient: Ambient static pressure (Pa).
        mass_flow_total: Total mass flow (air + fuel) (kg/s).
        gamma: Ratio of specific heats.
        cp: Specific heat at constant pressure (J/(kg·K)).

    Returns:
        NozzleResult with exit conditions and gross thrust.
    """
    R_gas = cp * (gamma - 1.0) / gamma
    critical_pr = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    npr = P_in / P_ambient

    if npr > critical_pr:
        # Choked nozzle: exit at sonic conditions
        T_exit = T_in * 2.0 / (gamma + 1.0)
        P_exit = P_in / critical_pr
        V_exit = math.sqrt(gamma * R_gas * T_exit)
        rho_exit = P_exit / (R_gas * T_exit)
        A_exit = mass_flow_total / (rho_exit * V_exit) if rho_exit * V_exit > 0 else 0.0
        gross_thrust = mass_flow_total * V_exit + (P_exit - P_ambient) * A_exit
        mach_exit = 1.0
        is_choked = True
    else:
        # Unchoked nozzle: full expansion to P_ambient
        P_exit = P_ambient
        T_exit = T_in * (P_ambient / P_in) ** ((gamma - 1.0) / gamma)
        V_exit = math.sqrt(2.0 * cp * (T_in - T_exit))
        rho_exit = P_exit / (R_gas * T_exit)
        A_exit = mass_flow_total / (rho_exit * V_exit) if rho_exit * V_exit > 0 else 0.0
        gross_thrust = mass_flow_total * V_exit
        a_exit = math.sqrt(gamma * R_gas * T_exit)
        mach_exit = V_exit / a_exit if a_exit > 0 else 0.0
        is_choked = False

    return NozzleResult(
        V_exit=V_exit,
        P_exit=P_exit,
        T_exit=T_exit,
        A_exit=A_exit,
        is_choked=is_choked,
        gross_thrust=gross_thrust,
        mach_exit=mach_exit,
    )


def turbine_expansion_ratio_from_power_balance(
    compressor_work: float,
    T_turbine_inlet: float,
    eta_turbine: float = 0.90,
    eta_mech: float = 0.99,
    gamma_hot: float = 1.33,
    cp_hot: float = 1150.0,
) -> float:
    """Compute turbine expansion ratio needed to drive the compressor.

    Args:
        compressor_work: Specific compressor work (J/kg).
        T_turbine_inlet: Turbine inlet total temperature (K).
        eta_turbine: Turbine isentropic efficiency.
        eta_mech: Mechanical efficiency (shaft losses).
        gamma_hot: Ratio of specific heats for hot gas.
        cp_hot: Specific heat for hot gas (J/(kg·K)).

    Returns:
        Expansion ratio P_in / P_out for the turbine.
    """
    work_req = compressor_work / eta_mech
    temp_ratio = 1.0 - work_req / (cp_hot * T_turbine_inlet * eta_turbine)
    temp_ratio = max(temp_ratio, 0.05)
    expansion_ratio = temp_ratio ** (-gamma_hot / (gamma_hot - 1.0))
    return expansion_ratio


# ── Input Validation ─────────────────────────────────


def _validate_cycle_inputs(
    engine_type: str,
    overall_pressure_ratio: float,
    turbine_inlet_temp: float,
    mass_flow: float,
    rpm: float,
    altitude: float,
    mach_flight: float,
) -> None:
    """Validate engine cycle inputs; raise ValueError on bad data."""
    if engine_type not in ("turbojet", "turboshaft"):
        raise ValueError(
            f"engine_type must be 'turbojet' or 'turboshaft', got '{engine_type}'"
        )
    if not (1.5 <= overall_pressure_ratio <= 60.0):
        raise ValueError(
            f"overall_pressure_ratio={overall_pressure_ratio} out of range [1.5, 60]"
        )
    if not (800.0 <= turbine_inlet_temp <= 2200.0):
        raise ValueError(
            f"turbine_inlet_temp={turbine_inlet_temp} out of range [800, 2200] K"
        )
    if not (0.1 <= mass_flow <= 1000.0):
        raise ValueError(f"mass_flow={mass_flow} out of range [0.1, 1000] kg/s")
    if not (100 <= rpm <= 200000):
        raise ValueError(f"rpm={rpm} out of range [100, 200000]")
    if not (0.0 <= altitude <= 47000.0):
        raise ValueError(f"altitude={altitude} out of range [0, 47000] m")
    if not (0.0 <= mach_flight <= 3.5):
        raise ValueError(f"mach_flight={mach_flight} out of range [0, 3.5]")


# ── Engine Cycle Result ──────────────────────────────


@dataclass
class EngineCycleResult:
    """Complete engine cycle analysis result."""

    engine_type: str = "turbojet"

    # Component results
    inlet: InletResult = field(default_factory=InletResult)
    compressor: Union[MeanlineResult, CentrifugalResult, None] = None
    combustor: CombustorResult = field(default_factory=CombustorResult)
    turbine: TurbineResult | None = None
    nozzle: NozzleResult | None = None

    # Station conditions (P_total, T_total at each station)
    stations: dict = field(default_factory=dict)

    # Overall performance
    net_thrust: float = 0.0                  # N
    specific_fuel_consumption: float = 0.0   # kg/(N·s)
    thermal_efficiency: float = 0.0
    propulsive_efficiency: float = 0.0
    overall_efficiency: float = 0.0

    # Power and flow
    shaft_power: float = 0.0          # W (turboshaft only)
    mass_flow: float = 0.0            # kg/s air
    fuel_flow: float = 0.0            # kg/s fuel
    compressor_work: float = 0.0      # J/kg
    turbine_work: float = 0.0         # J/kg
    mechanical_efficiency: float = 0.99

    def summary(self) -> str:
        """Human-readable engine cycle summary."""
        lines = [
            f"Engine Cycle Analysis — {self.engine_type.upper()}",
            "=" * 50,
        ]

        if self.engine_type == "turbojet":
            lines.append(f"  Net Thrust:      {self.net_thrust:.1f} N ({self.net_thrust / 1000:.2f} kN)")
            lines.append(f"  SFC:             {self.specific_fuel_consumption * 3600:.4f} kg/(N·h)")
        else:
            lines.append(f"  Shaft Power:     {self.shaft_power:.0f} W ({self.shaft_power / 1000:.1f} kW)")
            if self.shaft_power > 0:
                bsfc = self.fuel_flow / self.shaft_power * 1e6
                lines.append(f"  BSFC:            {bsfc:.1f} g/(kW·h)")

        lines.append(f"  Mass Flow (air): {self.mass_flow:.2f} kg/s")
        lines.append(f"  Fuel Flow:       {self.fuel_flow:.4f} kg/s")
        lines.append(f"  Fuel/Air Ratio:  {self.combustor.fuel_air_ratio:.4f}")
        lines.append("")

        lines.append("  Efficiencies:")
        lines.append(f"    Thermal:       {self.thermal_efficiency:.4f}")
        if self.engine_type == "turbojet":
            lines.append(f"    Propulsive:    {self.propulsive_efficiency:.4f}")
        lines.append(f"    Overall:       {self.overall_efficiency:.4f}")
        lines.append("")

        lines.append("  Power Balance:")
        lines.append(f"    Compressor:    {self.compressor_work:.0f} J/kg")
        lines.append(f"    Turbine:       {self.turbine_work:.0f} J/kg")
        lines.append("")

        lines.append("  Station Conditions:")
        lines.append(f"    {'Station':<20s} {'P_total (kPa)':>14s} {'T_total (K)':>12s}")
        lines.append(f"    {'-'*20} {'-'*14} {'-'*12}")
        for name, st in self.stations.items():
            lines.append(f"    {name:<20s} {st.P_total / 1000:>14.2f} {st.T_total:>12.1f}")

        return "\n".join(lines)


# ── Main Engine Cycle Solver ─────────────────────────


def engine_cycle(
    engine_type: str = "turbojet",
    altitude: float = 0.0,
    mach_flight: float = 0.0,
    overall_pressure_ratio: float = 8.0,
    turbine_inlet_temp: float = 1400.0,
    mass_flow: float = 20.0,
    rpm: float = 15000.0,
    r_hub: float = 0.15,
    r_tip: float = 0.30,
    compressor_type: str = "axial",
    eta_inlet: float = 0.97,
    eta_combustor: float = 0.99,
    dp_combustor: float = 0.04,
    eta_mech: float = 0.99,
    Q_fuel: float = 43e6,
    gamma_cold: float = 1.4,
    cp_cold: float = 1005.0,
    gamma_hot: float = 1.33,
    cp_hot: float = 1150.0,
    n_compressor_stages: int | None = None,
    n_turbine_stages: int | None = None,
    eta_poly_compressor: float = 0.90,
    eta_poly_turbine: float = 0.90,
    compressor_reaction: float = 0.5,
    turbine_reaction: float = 0.5,
) -> EngineCycleResult:
    """Run a complete gas turbine cycle analysis.

    Pipeline:
        1. Standard atmosphere → ambient conditions
        2. Ram inlet → station 2 (compressor face)
        3. Compressor (axial or centrifugal) → station 3
        4. Combustor → station 4 (turbine inlet)
        5. Turbine (sized by power balance) → station 5
        6. Turbojet: nozzle → station 8, compute thrust/SFC
           Turboshaft: compute shaft power

    Args:
        engine_type: "turbojet" or "turboshaft".
        altitude: Flight altitude (m).
        mach_flight: Flight Mach number.
        overall_pressure_ratio: Compressor total pressure ratio.
        turbine_inlet_temp: Combustor exit / turbine inlet temperature (K).
        mass_flow: Air mass flow rate (kg/s).
        rpm: Compressor/turbine shaft speed (RPM).
        r_hub: Blade hub radius (m).
        r_tip: Blade tip radius (m).
        compressor_type: "axial" or "centrifugal".
        eta_inlet: Inlet pressure recovery factor.
        eta_combustor: Combustion efficiency.
        dp_combustor: Combustor fractional pressure loss.
        eta_mech: Mechanical efficiency (shaft losses).
        Q_fuel: Fuel lower heating value (J/kg).
        gamma_cold: Ratio of specific heats for cold air.
        cp_cold: Specific heat for cold air (J/(kg·K)).
        gamma_hot: Ratio of specific heats for hot gas.
        cp_hot: Specific heat for hot gas (J/(kg·K)).
        n_compressor_stages: Number of compressor stages (auto if None).
        n_turbine_stages: Number of turbine stages (auto if None).
        eta_poly_compressor: Compressor polytropic efficiency.
        eta_poly_turbine: Turbine polytropic efficiency.
        compressor_reaction: Compressor stage reaction.
        turbine_reaction: Turbine stage reaction.

    Returns:
        EngineCycleResult with full station-by-station breakdown.
    """
    # 1. Validate
    _validate_cycle_inputs(
        engine_type, overall_pressure_ratio, turbine_inlet_temp,
        mass_flow, rpm, altitude, mach_flight,
    )

    # 2. Ambient conditions
    T_amb, P_amb, _rho = standard_atmosphere(altitude)
    stations = {}
    stations["ambient"] = StationConditions(P_total=P_amb, T_total=T_amb)

    # 3. Inlet
    inlet_res = inlet_model(T_amb, P_amb, mach_flight, eta_inlet, gamma_cold)
    stations["inlet_exit"] = StationConditions(
        P_total=inlet_res.P_total_out,
        T_total=inlet_res.T_total_out,
    )

    # 4. Compressor
    gas_cold = GasProperties(gamma=gamma_cold, cp=cp_cold)
    if compressor_type == "centrifugal":
        comp_result = centrifugal_compressor(
            pressure_ratio=overall_pressure_ratio,
            mass_flow=mass_flow,
            rpm=rpm,
            gas=gas_cold,
            T_inlet=inlet_res.T_total_out,
            P_inlet=inlet_res.P_total_out,
        )
        P3 = comp_result.P_outlet
        T3 = comp_result.T_outlet
        compressor_work = comp_result.power_kW * 1000.0 / mass_flow  # J/kg
    else:
        comp_result = meanline_compressor(
            overall_pressure_ratio=overall_pressure_ratio,
            mass_flow=mass_flow,
            rpm=rpm,
            r_hub=r_hub,
            r_tip=r_tip,
            n_stages=n_compressor_stages,
            eta_poly=eta_poly_compressor,
            reaction=compressor_reaction,
            gas=gas_cold,
            T_inlet=inlet_res.T_total_out,
            P_inlet=inlet_res.P_total_out,
        )
        last_station = comp_result.stations[-1]
        P3 = last_station.P_total
        T3 = last_station.T_total
        compressor_work = comp_result.total_work  # J/kg

    stations["compressor_exit"] = StationConditions(P_total=P3, T_total=T3)

    # 5. Combustor
    comb_result = combustor_model(
        P_in=P3,
        T_in=T3,
        T_target_out=turbine_inlet_temp,
        mass_flow_air=mass_flow,
        eta_combustor=eta_combustor,
        dp_fraction=dp_combustor,
        Q_fuel=Q_fuel,
        gamma_hot=gamma_hot,
        cp_hot=cp_hot,
    )
    P4 = comb_result.P_out
    T4 = comb_result.T_out
    fuel_flow = comb_result.fuel_flow
    mass_flow_total = mass_flow + fuel_flow
    stations["combustor_exit"] = StationConditions(P_total=P4, T_total=T4)

    # 6. Turbine expansion ratio
    # For turbojet: turbine only drives compressor, remaining energy goes to nozzle
    # For turboshaft: turbine expands fully (to near-ambient), all work to shaft
    turb_er = turbine_expansion_ratio_from_power_balance(
        compressor_work=compressor_work,
        T_turbine_inlet=T4,
        eta_turbine=eta_poly_turbine,
        eta_mech=eta_mech,
        gamma_hot=gamma_hot,
        cp_hot=cp_hot,
    )

    if engine_type == "turboshaft":
        # Expand turbine to exhaust pressure (slightly above ambient for losses)
        exhaust_loss_factor = 1.05  # 5% exhaust pressure loss
        max_turb_er = P4 / (P_amb * exhaust_loss_factor)
        turb_er = min(max_turb_er, P4 / P_amb)  # Don't expand below ambient

    # 7. Turbine meanline design
    gas_hot = GasProperties(gamma=gamma_hot, cp=cp_hot,
                            R=cp_hot * (gamma_hot - 1.0) / gamma_hot)
    turb_result = meanline_turbine(
        overall_expansion_ratio=turb_er,
        mass_flow=mass_flow_total,
        rpm=rpm,
        r_hub=r_hub,
        r_tip=r_tip,
        n_stages=n_turbine_stages,
        eta_poly=eta_poly_turbine,
        reaction=turbine_reaction,
        gas=gas_hot,
        T_inlet=T4,
        P_inlet=P4,
    )
    last_turb_station = turb_result.stations[-1]
    P5 = last_turb_station.P_total
    T5 = last_turb_station.T_total
    turbine_work = turb_result.total_work  # J/kg (positive)
    stations["turbine_exit"] = StationConditions(P_total=P5, T_total=T5)

    # 8. Nozzle or shaft power
    nozzle_res = None
    net_thrust = 0.0
    sfc = 0.0
    shaft_power = 0.0
    V_flight = mach_flight * math.sqrt(gamma_cold * (cp_cold * (gamma_cold - 1.0) / gamma_cold) * T_amb)

    if engine_type == "turbojet":
        nozzle_res = nozzle_model(
            P_in=P5,
            T_in=T5,
            P_ambient=P_amb,
            mass_flow_total=mass_flow_total,
            gamma=gamma_hot,
            cp=cp_hot,
        )
        stations["nozzle_exit"] = StationConditions(
            P_total=nozzle_res.P_exit,  # static at nozzle exit
            T_total=nozzle_res.T_exit,
        )
        # Net thrust = gross thrust - ram drag
        ram_drag = mass_flow * V_flight
        net_thrust = nozzle_res.gross_thrust - ram_drag
        if net_thrust > 0 and fuel_flow > 0:
            sfc = fuel_flow / net_thrust
    else:
        # Turboshaft: total turbine work minus compressor drive = shaft output
        shaft_work_specific = turbine_work * eta_mech - compressor_work
        shaft_power = shaft_work_specific * mass_flow_total  # Watts

    # 9. Efficiencies
    Q_in = fuel_flow * Q_fuel
    thermal_eff = 0.0
    propulsive_eff = 0.0
    overall_eff = 0.0

    if engine_type == "turbojet" and Q_in > 0:
        kinetic_gain = (0.5 * mass_flow_total * nozzle_res.V_exit ** 2
                        - 0.5 * mass_flow * V_flight ** 2)
        thermal_eff = kinetic_gain / Q_in if Q_in > 0 else 0.0
        thrust_power = net_thrust * V_flight
        propulsive_eff = thrust_power / kinetic_gain if kinetic_gain > 0 else 0.0
        overall_eff = thermal_eff * propulsive_eff
        # For static case (V_flight=0), use thermal_eff as overall
        if mach_flight < 0.01:
            overall_eff = thermal_eff
    elif engine_type == "turboshaft" and Q_in > 0:
        thermal_eff = shaft_power / Q_in
        overall_eff = thermal_eff

    return EngineCycleResult(
        engine_type=engine_type,
        inlet=inlet_res,
        compressor=comp_result,
        combustor=comb_result,
        turbine=turb_result,
        nozzle=nozzle_res,
        stations=stations,
        net_thrust=net_thrust,
        specific_fuel_consumption=sfc,
        thermal_efficiency=thermal_eff,
        propulsive_efficiency=propulsive_eff,
        overall_efficiency=overall_eff,
        shaft_power=shaft_power,
        mass_flow=mass_flow,
        fuel_flow=fuel_flow,
        compressor_work=compressor_work,
        turbine_work=turbine_work,
        mechanical_efficiency=eta_mech,
    )
