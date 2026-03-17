"""Meanline analysis for turbomachinery stages.

Performs 1D thermodynamic and aerodynamic analysis at the mean radius
of each blade row. Given high-level requirements (pressure ratio, mass flow,
RPM), computes velocity triangles, blade angles, and performance at each
station through the machine.

This is the bridge between cycle design and blade geometry:
    Cycle requirements → meanline → blade angles → AstraTurbo geometry → mesh
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .velocity_triangle import (
    VelocityTriangle,
    BladeRowTriangles,
    compute_triangle_from_angles,
    compute_triangle_from_beta,
)


@dataclass
class GasProperties:
    """Thermodynamic properties of the working fluid."""

    gamma: float = 1.4          # Ratio of specific heats (air)
    cp: float = 1005.0          # Specific heat at constant pressure (J/(kg·K))
    R: float = 287.0            # Specific gas constant (J/(kg·K))

    @property
    def cv(self) -> float:
        return self.cp / self.gamma

    def speed_of_sound(self, T: float) -> float:
        """Speed of sound at temperature T (K)."""
        return math.sqrt(self.gamma * self.R * T)

    def mach_number(self, V: float, T: float) -> float:
        """Mach number for velocity V at temperature T."""
        a = self.speed_of_sound(T)
        return V / a if a > 0 else 0.0


@dataclass
class StationConditions:
    """Thermodynamic conditions at a station between blade rows."""

    P_total: float          # Total pressure (Pa)
    T_total: float          # Total temperature (K)
    alpha: float = 0.0      # Absolute flow angle (radians)
    C_axial: float = 0.0    # Axial velocity (m/s)
    r_mean: float = 0.0     # Mean radius at this station (m)
    area: float = 0.0       # Annulus area (m²)

    @property
    def C_theta(self) -> float:
        return self.C_axial * math.tan(self.alpha)

    @property
    def C(self) -> float:
        return math.sqrt(self.C_axial**2 + self.C_theta**2)

    @property
    def T_static(self) -> float:
        """Static temperature from total temperature and velocity."""
        gas = GasProperties()
        return self.T_total - self.C**2 / (2 * gas.cp)

    @property
    def P_static(self) -> float:
        """Static pressure from isentropic relation."""
        gas = GasProperties()
        T_ratio = self.T_static / self.T_total
        return self.P_total * T_ratio ** (gas.gamma / (gas.gamma - 1))


@dataclass
class StageResult:
    """Results from meanline analysis of a single stage."""

    stage_number: int
    rotor_triangles: BladeRowTriangles
    stator_triangles: BladeRowTriangles | None

    # Thermodynamic results
    pressure_ratio: float = 0.0
    temperature_ratio: float = 0.0
    isentropic_efficiency: float = 0.0
    polytropic_efficiency: float = 0.0
    work_input: float = 0.0             # J/kg

    # Aerodynamic parameters
    flow_coefficient: float = 0.0       # phi = C_axial / U
    loading_coefficient: float = 0.0    # psi = delta_h / U²
    degree_of_reaction: float = 0.0     # R

    # Blade geometry outputs (feed into blade/ module)
    rotor_inlet_beta: float = 0.0       # Relative inlet angle (rad)
    rotor_outlet_beta: float = 0.0      # Relative outlet angle (rad)
    stator_inlet_alpha: float = 0.0     # Absolute inlet angle (rad)
    stator_outlet_alpha: float = 0.0    # Absolute outlet angle (rad)

    def blade_angles_deg(self) -> dict:
        """Return all blade metal angles in degrees for geometry input."""
        return {
            "rotor_inlet_beta": math.degrees(self.rotor_inlet_beta),
            "rotor_outlet_beta": math.degrees(self.rotor_outlet_beta),
            "stator_inlet_alpha": math.degrees(self.stator_inlet_alpha),
            "stator_outlet_alpha": math.degrees(self.stator_outlet_alpha),
        }


@dataclass
class MeanlineResult:
    """Complete meanline analysis result for a multi-stage machine."""

    stages: list[StageResult] = field(default_factory=list)
    stations: list[StationConditions] = field(default_factory=list)

    # Overall performance
    overall_pressure_ratio: float = 0.0
    overall_temperature_ratio: float = 0.0
    overall_efficiency: float = 0.0
    total_work: float = 0.0             # J/kg
    n_stages: int = 0

    def summary(self) -> str:
        lines = [
            f"Meanline Analysis: {self.n_stages} stages",
            f"  Overall PR:   {self.overall_pressure_ratio:.3f}",
            f"  Overall TR:   {self.overall_temperature_ratio:.3f}",
            f"  Overall eta:  {self.overall_efficiency:.4f}",
            f"  Total work:   {self.total_work:.0f} J/kg",
            "",
        ]
        for s in self.stages:
            angles = s.blade_angles_deg()
            lines.append(f"  Stage {s.stage_number}:")
            lines.append(f"    PR = {s.pressure_ratio:.3f}, eta = {s.isentropic_efficiency:.4f}")
            lines.append(f"    phi = {s.flow_coefficient:.3f}, psi = {s.loading_coefficient:.3f}, R = {s.degree_of_reaction:.3f}")
            lines.append(f"    Rotor beta:  {angles['rotor_inlet_beta']:.1f} → {angles['rotor_outlet_beta']:.1f} deg")
            lines.append(f"    Stator alpha: {angles['stator_inlet_alpha']:.1f} → {angles['stator_outlet_alpha']:.1f} deg")
            lines.append(f"    De Haller: {s.rotor_triangles.de_haller_ratio:.3f}")
        return "\n".join(lines)


def meanline_compressor_stage(
    U: float,
    C_axial: float,
    alpha_in: float,
    stage_pressure_ratio: float,
    eta_stage: float = 0.88,
    reaction: float = 0.5,
    gas: GasProperties | None = None,
    T_in: float = 288.15,
    P_in: float = 101325.0,
) -> StageResult:
    """Design a single compressor stage at the meanline.

    Uses specified reaction and loading to compute velocity triangles,
    then derives blade angles.

    Args:
        U: Blade speed at mean radius (m/s).
        C_axial: Axial velocity (m/s), assumed constant through stage.
        alpha_in: Absolute inlet flow angle (radians).
        stage_pressure_ratio: Target pressure ratio for this stage.
        eta_stage: Isentropic stage efficiency (0-1).
        reaction: Degree of reaction (0-1). 0.5 = symmetric stage.
        gas: Gas properties (default: air).
        T_in: Total temperature at inlet (K).
        P_in: Total pressure at inlet (Pa).

    Returns:
        StageResult with velocity triangles and blade angles.
    """
    if gas is None:
        gas = GasProperties()

    # Isentropic temperature rise
    T_out_ideal = T_in * stage_pressure_ratio ** ((gas.gamma - 1) / gas.gamma)
    delta_T_ideal = T_out_ideal - T_in
    delta_T_actual = delta_T_ideal / eta_stage

    # Work input from Euler equation: w = cp * delta_T
    work = gas.cp * delta_T_actual

    # Loading coefficient: psi = w / U²
    psi = work / U**2

    # Flow coefficient: phi = C_axial / U
    phi = C_axial / U

    # From reaction and loading, compute swirl change
    # R = 1 - (C_theta_in + C_theta_out) / (2U)
    # psi = (C_theta_out - C_theta_in) / U
    # Solving: C_theta_in = U * (1 - R - psi/2)
    #          C_theta_out = U * (1 - R + psi/2)
    C_theta_in = U * (1 - reaction - psi / 2)
    C_theta_out = U * (1 - reaction + psi / 2)

    # Build velocity triangles
    rotor_inlet = VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta_in)
    rotor_outlet = VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta_out)

    rotor_triangles = BladeRowTriangles(inlet=rotor_inlet, outlet=rotor_outlet)

    # Stator: removes swirl, outlet alpha ≈ 0 (or repeating stage: alpha_out = alpha_in)
    stator_inlet = VelocityTriangle(U=0, C_axial=C_axial, C_theta=C_theta_out)
    stator_outlet = VelocityTriangle(U=0, C_axial=C_axial, C_theta=C_theta_in)
    stator_triangles = BladeRowTriangles(inlet=stator_inlet, outlet=stator_outlet)

    # Temperature ratio
    T_out = T_in + delta_T_actual
    T_ratio = T_out / T_in

    # Polytropic efficiency
    if stage_pressure_ratio > 1.0:
        eta_poly = (
            (gas.gamma - 1) / gas.gamma
            * math.log(stage_pressure_ratio)
            / math.log(T_ratio)
        )
    else:
        eta_poly = eta_stage

    return StageResult(
        stage_number=0,
        rotor_triangles=rotor_triangles,
        stator_triangles=stator_triangles,
        pressure_ratio=stage_pressure_ratio,
        temperature_ratio=T_ratio,
        isentropic_efficiency=eta_stage,
        polytropic_efficiency=eta_poly,
        work_input=work,
        flow_coefficient=phi,
        loading_coefficient=psi,
        degree_of_reaction=reaction,
        rotor_inlet_beta=rotor_inlet.beta,
        rotor_outlet_beta=rotor_outlet.beta,
        stator_inlet_alpha=stator_inlet.alpha,
        stator_outlet_alpha=stator_outlet.alpha,
    )


def meanline_compressor(
    overall_pressure_ratio: float,
    mass_flow: float,
    rpm: float,
    r_hub: float,
    r_tip: float,
    n_stages: int | None = None,
    eta_poly: float = 0.90,
    reaction: float = 0.5,
    max_psi: float = 0.45,
    gas: GasProperties | None = None,
    T_inlet: float = 288.15,
    P_inlet: float = 101325.0,
) -> MeanlineResult:
    """Design a multi-stage axial compressor from top-level requirements.

    Automatically determines number of stages if not specified (based on
    maximum loading coefficient), then designs each stage.

    Args:
        overall_pressure_ratio: Total-to-total pressure ratio (e.g. 8.0).
        mass_flow: Mass flow rate (kg/s).
        rpm: Rotational speed (rev/min).
        r_hub: Hub radius at mean station (m).
        r_tip: Tip radius at mean station (m).
        n_stages: Number of stages (None = auto-calculate).
        eta_poly: Polytropic efficiency (0-1).
        reaction: Degree of reaction for each stage.
        max_psi: Maximum loading coefficient per stage.
        gas: Gas properties.
        T_inlet: Inlet total temperature (K).
        P_inlet: Inlet total pressure (Pa).

    Returns:
        MeanlineResult with all stages, stations, and performance.
    """
    if gas is None:
        gas = GasProperties()

    r_mean = (r_hub + r_tip) / 2.0
    omega = rpm * 2 * math.pi / 60.0
    U = omega * r_mean

    # Annulus area and axial velocity
    A = math.pi * (r_tip**2 - r_hub**2)
    rho_inlet = P_inlet / (gas.R * T_inlet)
    C_axial = mass_flow / (rho_inlet * A)

    # Total work required
    T_outlet_ideal = T_inlet * overall_pressure_ratio ** ((gas.gamma - 1) / gas.gamma)
    T_outlet = T_inlet * overall_pressure_ratio ** ((gas.gamma - 1) / (gas.gamma * eta_poly))
    total_work = gas.cp * (T_outlet - T_inlet)

    # Auto-calculate number of stages from loading limit
    if n_stages is None:
        work_per_stage_max = max_psi * U**2
        n_stages = max(1, math.ceil(total_work / work_per_stage_max))

    # Distribute work and pressure ratio across stages
    # Use equal loading (equal work split)
    work_per_stage = total_work / n_stages

    # Stage pressure ratios (from polytropic relation)
    stage_results = []
    stations = []

    T_current = T_inlet
    P_current = P_inlet
    alpha_current = 0.0  # No inlet swirl

    stations.append(StationConditions(
        P_total=P_current, T_total=T_current,
        alpha=alpha_current, C_axial=C_axial,
        r_mean=r_mean, area=A,
    ))

    for i in range(n_stages):
        delta_T = work_per_stage / gas.cp
        T_next = T_current + delta_T
        stage_pr = (T_next / T_current) ** (gas.gamma * eta_poly / (gas.gamma - 1))

        # Isentropic efficiency for this stage
        T_next_ideal = T_current * stage_pr ** ((gas.gamma - 1) / gas.gamma)
        eta_isen = (T_next_ideal - T_current) / (T_next - T_current) if delta_T > 0 else eta_poly

        stage = meanline_compressor_stage(
            U=U, C_axial=C_axial, alpha_in=alpha_current,
            stage_pressure_ratio=stage_pr, eta_stage=eta_isen,
            reaction=reaction, gas=gas, T_in=T_current, P_in=P_current,
        )
        stage.stage_number = i + 1
        stage_results.append(stage)

        # Update conditions for next stage
        P_current *= stage_pr
        T_current = T_next
        alpha_current = stage.stator_outlet_alpha

        stations.append(StationConditions(
            P_total=P_current, T_total=T_current,
            alpha=alpha_current, C_axial=C_axial,
            r_mean=r_mean, area=A,
        ))

    result = MeanlineResult(
        stages=stage_results,
        stations=stations,
        overall_pressure_ratio=P_current / P_inlet,
        overall_temperature_ratio=T_current / T_inlet,
        overall_efficiency=eta_poly,
        total_work=total_work,
        n_stages=n_stages,
    )
    return result


def meanline_to_blade_parameters(result: MeanlineResult) -> list[dict]:
    """Convert meanline results to blade geometry parameters.

    For each stage, produces a dict with the blade angles, chord estimates,
    and stagger angles that can be fed directly into AstraTurbo's blade/ module.

    Returns:
        List of dicts, one per stage, with keys:
            rotor_stagger, rotor_camber, rotor_solidity,
            stator_stagger, stator_camber, stator_solidity
    """
    params = []
    for stage in result.stages:
        beta_in = stage.rotor_inlet_beta
        beta_out = stage.rotor_outlet_beta
        alpha_in = stage.stator_inlet_alpha
        alpha_out = stage.stator_outlet_alpha

        # Stagger ≈ average of inlet and outlet angles
        rotor_stagger = (beta_in + beta_out) / 2.0
        stator_stagger = (alpha_in + alpha_out) / 2.0

        # Camber ≈ flow turning
        rotor_camber = abs(beta_in - beta_out)
        stator_camber = abs(alpha_in - alpha_out)

        # Solidity estimate (Lieblein diffusion factor based)
        # sigma ≈ 1.0 for moderate loading
        rotor_solidity = 1.0 + 0.5 * stage.loading_coefficient
        stator_solidity = 1.0 + 0.5 * stage.loading_coefficient

        params.append({
            "stage": stage.stage_number,
            "rotor_stagger_deg": math.degrees(rotor_stagger),
            "rotor_camber_deg": math.degrees(rotor_camber),
            "rotor_solidity": rotor_solidity,
            "rotor_inlet_beta_deg": math.degrees(beta_in),
            "rotor_outlet_beta_deg": math.degrees(beta_out),
            "stator_stagger_deg": math.degrees(stator_stagger),
            "stator_camber_deg": math.degrees(stator_camber),
            "stator_solidity": stator_solidity,
            "stator_inlet_alpha_deg": math.degrees(alpha_in),
            "stator_outlet_alpha_deg": math.degrees(alpha_out),
            "flow_coefficient": stage.flow_coefficient,
            "loading_coefficient": stage.loading_coefficient,
            "reaction": stage.degree_of_reaction,
            "de_haller": stage.rotor_triangles.de_haller_ratio,
        })

    return params
