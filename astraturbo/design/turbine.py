"""Meanline analysis for axial turbine stages.

Performs 1D thermodynamic and aerodynamic analysis at the mean radius
of each blade row for an axial turbine.  Key differences from compressor:
  - Expanding flow (pressure drops through each stage)
  - NGV (nozzle guide vane / stator) comes first, accelerates flow
  - Rotor extracts work (positive work output)
  - Soderberg loss model (simpler, standard for turbine prelim design)
  - Zweifel loading coefficient (turbine-specific blade loading measure)

Usage:
    from astraturbo.design.turbine import meanline_turbine

    result = meanline_turbine(
        overall_expansion_ratio=2.5, mass_flow=20.0,
        rpm=17189, r_hub=0.25, r_tip=0.35, inlet_temp=1500.0,
    )
    print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .velocity_triangle import VelocityTriangle, BladeRowTriangles
from .meanline import GasProperties, StationConditions


# ── Loss and loading models ──────────────────────────


def soderberg_loss(deflection_deg: float, Re: float = 1e5,
                   aspect_ratio: float = 3.0) -> float:
    """Soderberg total-pressure loss coefficient for a turbine blade row.

    Args:
        deflection_deg: Flow deflection (turning) in degrees.
        Re: Reynolds number based on chord and exit velocity.
        aspect_ratio: Blade span / axial chord.

    Returns:
        zeta — total-pressure loss coefficient (delta_P0 / q_exit).
    """
    Re_ref = 1.0e5
    eps = deflection_deg
    zeta = (0.04 * (1.0 + 1.5 * (eps / 100.0) ** 2)
            * (Re_ref / max(Re, 1e3)) ** 0.2
            / max(aspect_ratio, 0.5))
    return zeta


def zweifel_loading(alpha_in_deg: float, alpha_out_deg: float,
                    pitch: float, axial_chord: float) -> float:
    """Zweifel blade-loading coefficient.

    Optimal range is 0.8 – 1.0; values > 1.2 indicate overloaded blading.

    Args:
        alpha_in_deg: Inlet flow angle (degrees from axial).
        alpha_out_deg: Outlet flow angle (degrees from axial).
        pitch: Blade pitch (m).
        axial_chord: Blade axial chord (m).

    Returns:
        Z — Zweifel loading coefficient.
    """
    a_in = math.radians(alpha_in_deg)
    a_out = math.radians(alpha_out_deg)
    s_cx = pitch / max(axial_chord, 1e-6)
    Z = 2.0 * s_cx * math.cos(a_out) ** 2 * (
        math.tan(a_in) + math.tan(a_out)
    )
    return abs(Z)


# ── Dataclasses ──────────────────────────────────────


@dataclass
class TurbineStageResult:
    """Results from meanline analysis of a single turbine stage."""

    stage_number: int
    nozzle_triangles: BladeRowTriangles   # NGV (stator first)
    rotor_triangles: BladeRowTriangles

    # Thermodynamic
    expansion_ratio: float = 0.0          # P_in / P_out  (> 1)
    temperature_ratio: float = 0.0        # T_out / T_in  (< 1)
    isentropic_efficiency: float = 0.0
    work_output: float = 0.0              # J/kg, positive = extracted

    # Aerodynamic
    flow_coefficient: float = 0.0         # phi = C_axial / U
    loading_coefficient: float = 0.0      # psi = w / U^2
    degree_of_reaction: float = 0.0
    zweifel_coefficient: float = 0.0

    # NGV exit conditions
    nozzle_exit_mach: float = 0.0

    # Angles (radians)
    nozzle_inlet_alpha: float = 0.0
    nozzle_outlet_alpha: float = 0.0
    rotor_inlet_beta: float = 0.0
    rotor_outlet_beta: float = 0.0

    # Radial distribution
    radial_blade_angles: list[dict] = field(default_factory=list)

    def blade_angles_deg(self) -> dict:
        """Return all blade metal angles in degrees."""
        return {
            "nozzle_inlet_alpha": math.degrees(self.nozzle_inlet_alpha),
            "nozzle_outlet_alpha": math.degrees(self.nozzle_outlet_alpha),
            "rotor_inlet_beta": math.degrees(self.rotor_inlet_beta),
            "rotor_outlet_beta": math.degrees(self.rotor_outlet_beta),
        }


@dataclass
class TurbineResult:
    """Complete meanline result for a multi-stage axial turbine."""

    stages: list[TurbineStageResult] = field(default_factory=list)
    stations: list[StationConditions] = field(default_factory=list)

    overall_expansion_ratio: float = 0.0
    overall_temperature_ratio: float = 0.0
    overall_efficiency: float = 0.0
    total_work: float = 0.0               # J/kg (positive)
    n_stages: int = 0

    def summary(self) -> str:
        lines = [
            f"Turbine Meanline Analysis: {self.n_stages} stage(s)",
            f"  Overall ER:   {self.overall_expansion_ratio:.3f}",
            f"  Overall TR:   {self.overall_temperature_ratio:.4f}",
            f"  Overall eta:  {self.overall_efficiency:.4f}",
            f"  Total work:   {self.total_work:.0f} J/kg",
            "",
        ]
        for s in self.stages:
            a = s.blade_angles_deg()
            lines.append(f"  Stage {s.stage_number}:")
            lines.append(
                f"    ER = {s.expansion_ratio:.3f}, "
                f"eta = {s.isentropic_efficiency:.4f}"
            )
            lines.append(
                f"    phi = {s.flow_coefficient:.3f}, "
                f"psi = {s.loading_coefficient:.3f}, "
                f"R = {s.degree_of_reaction:.3f}"
            )
            lines.append(
                f"    Zweifel = {s.zweifel_coefficient:.3f}, "
                f"Nozzle M = {s.nozzle_exit_mach:.3f}"
            )
            lines.append(
                f"    NGV alpha: {a['nozzle_inlet_alpha']:.1f}"
                f" -> {a['nozzle_outlet_alpha']:.1f} deg"
            )
            lines.append(
                f"    Rotor beta: {a['rotor_inlet_beta']:.1f}"
                f" -> {a['rotor_outlet_beta']:.1f} deg"
            )
        return "\n".join(lines)


# ── Single-stage design ──────────────────────────────


def meanline_turbine_stage(
    U: float,
    C_axial: float,
    alpha_in: float,
    stage_expansion_ratio: float,
    eta_stage: float = 0.90,
    reaction: float = 0.5,
    gas: GasProperties | None = None,
    T_in: float = 1500.0,
    P_in: float = 101325.0,
) -> TurbineStageResult:
    """Design a single axial turbine stage at the meanline.

    Args:
        U: Blade speed at mean radius (m/s).
        C_axial: Axial velocity (m/s), assumed constant through stage.
        alpha_in: Absolute inlet flow angle (radians).
        stage_expansion_ratio: P_in / P_out for this stage (> 1).
        eta_stage: Isentropic stage efficiency (0-1).
        reaction: Degree of reaction (0-1).
        gas: Gas properties.
        T_in: Total temperature at inlet (K).
        P_in: Total pressure at inlet (Pa).

    Returns:
        TurbineStageResult with velocity triangles and blade angles.
    """
    if gas is None:
        gas = GasProperties()

    ER = stage_expansion_ratio

    # 1. Isentropic temperature drop
    T_out_ideal = T_in * (1.0 / ER) ** ((gas.gamma - 1) / gas.gamma)
    delta_T_ideal = T_in - T_out_ideal          # positive

    # 2. Actual temperature drop (turbine convention)
    delta_T_actual = eta_stage * delta_T_ideal
    T_out = T_in - delta_T_actual

    # 3. Work extracted and loading
    work = gas.cp * delta_T_actual               # J/kg, positive
    psi = work / U ** 2                          # loading coefficient
    phi = C_axial / U                            # flow coefficient

    # 4. Swirl distribution from reaction + loading
    #    R = 1 - (C_theta2 + C_theta3) / (2U)   (turbine convention)
    #    psi = (C_theta2 - C_theta3) / U
    #    => C_theta2 = U * (psi/2 + 1 - R)   (NGV exit swirl)
    #    => C_theta3 = U * (1 - R - psi/2)    (rotor exit swirl)
    C_theta2 = U * (psi / 2.0 + 1.0 - reaction)
    C_theta3 = U * (1.0 - reaction - psi / 2.0)

    # 5. Build velocity triangles
    # NGV: no blade speed (stationary), converts axial flow to swirled flow
    ngv_inlet = VelocityTriangle(
        U=0.0, C_axial=C_axial,
        C_theta=C_axial * math.tan(alpha_in),
    )
    ngv_outlet = VelocityTriangle(U=0.0, C_axial=C_axial, C_theta=C_theta2)
    nozzle_triangles = BladeRowTriangles(inlet=ngv_inlet, outlet=ngv_outlet)

    # Rotor: moving blades
    rotor_inlet = VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta2)
    rotor_outlet = VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta3)
    rotor_triangles = BladeRowTriangles(inlet=rotor_inlet, outlet=rotor_outlet)

    # 6. Soderberg losses (informational)
    ngv_deflection = abs(math.degrees(ngv_outlet.alpha) - math.degrees(
        math.atan2(ngv_inlet.C_theta, C_axial)
    ))
    rotor_deflection = abs(math.degrees(rotor_outlet.beta)
                           - math.degrees(rotor_inlet.beta))
    _ngv_loss = soderberg_loss(ngv_deflection)
    _rotor_loss = soderberg_loss(rotor_deflection)

    # 7. Zweifel loading (use rotor relative angles as tangential measure)
    # Approximate pitch/axial_chord ~ 0.8 (typical turbine)
    pitch_cx = 0.8
    zweifel = zweifel_loading(
        abs(math.degrees(rotor_inlet.beta)),
        abs(math.degrees(rotor_outlet.beta)),
        pitch_cx, 1.0,
    )

    # 8. Nozzle exit Mach
    C_ngv_exit = math.sqrt(C_axial ** 2 + C_theta2 ** 2)
    # Static temperature after NGV (use total-to-static)
    T_static_ngv = T_in - C_ngv_exit ** 2 / (2.0 * gas.cp)
    T_static_ngv = max(T_static_ngv, 100.0)     # safety floor
    a_ngv = gas.speed_of_sound(T_static_ngv)
    nozzle_mach = C_ngv_exit / a_ngv if a_ngv > 0 else 0.0

    # Temperature and pressure ratios
    temperature_ratio = T_out / T_in

    return TurbineStageResult(
        stage_number=0,
        nozzle_triangles=nozzle_triangles,
        rotor_triangles=rotor_triangles,
        expansion_ratio=ER,
        temperature_ratio=temperature_ratio,
        isentropic_efficiency=eta_stage,
        work_output=work,
        flow_coefficient=phi,
        loading_coefficient=psi,
        degree_of_reaction=reaction,
        zweifel_coefficient=zweifel,
        nozzle_exit_mach=nozzle_mach,
        nozzle_inlet_alpha=math.atan2(ngv_inlet.C_theta, C_axial),
        nozzle_outlet_alpha=ngv_outlet.alpha,
        rotor_inlet_beta=rotor_inlet.beta,
        rotor_outlet_beta=rotor_outlet.beta,
    )


# ── Multi-stage design ───────────────────────────────


def meanline_turbine(
    overall_expansion_ratio: float,
    mass_flow: float,
    rpm: float,
    r_hub: float,
    r_tip: float,
    n_stages: int | None = None,
    eta_poly: float = 0.90,
    reaction: float = 0.5,
    max_psi: float = 2.0,
    gas: GasProperties | None = None,
    T_inlet: float = 1500.0,
    P_inlet: float = 101325.0,
    radial_stations: int = 3,
) -> TurbineResult:
    """Design a multi-stage axial turbine from top-level requirements.

    Automatically determines number of stages if not specified (based on
    maximum loading coefficient).

    Args:
        overall_expansion_ratio: P_inlet / P_outlet (e.g. 2.5).
        mass_flow: Mass flow rate (kg/s).
        rpm: Rotational speed (rev/min).
        r_hub: Hub radius at mean station (m).
        r_tip: Tip radius at mean station (m).
        n_stages: Number of stages (None = auto-calculate).
        eta_poly: Polytropic efficiency (0-1).
        reaction: Degree of reaction for each stage.
        max_psi: Maximum loading coefficient per stage (default 2.0).
        gas: Gas properties.
        T_inlet: Inlet total temperature (K).
        P_inlet: Inlet total pressure (Pa).
        radial_stations: Number of radial stations for free-vortex (default 3).

    Returns:
        TurbineResult with all stages, stations, and performance.
    """
    if gas is None:
        gas = GasProperties()

    r_mean = (r_hub + r_tip) / 2.0
    omega = rpm * 2.0 * math.pi / 60.0
    U = omega * r_mean

    # Annulus area and axial velocity
    A = math.pi * (r_tip ** 2 - r_hub ** 2)
    rho_inlet = P_inlet / (gas.R * T_inlet)
    C_axial = mass_flow / (rho_inlet * A)

    # Total work available from expansion
    T_outlet_ideal = T_inlet * (1.0 / overall_expansion_ratio) ** (
        (gas.gamma - 1) / gas.gamma
    )
    # Polytropic outlet temperature
    T_outlet = T_inlet * (1.0 / overall_expansion_ratio) ** (
        (gas.gamma - 1) * eta_poly / gas.gamma
    )
    total_work = gas.cp * (T_inlet - T_outlet)

    # Auto-calculate number of stages from loading limit
    if n_stages is None:
        work_per_stage_max = max_psi * U ** 2
        n_stages = max(1, math.ceil(total_work / work_per_stage_max))

    work_per_stage = total_work / n_stages

    stage_results: list[TurbineStageResult] = []
    stations: list[StationConditions] = []

    T_current = T_inlet
    P_current = P_inlet
    alpha_current = 0.0  # axial inlet

    stations.append(StationConditions(
        P_total=P_current, T_total=T_current,
        alpha=alpha_current, C_axial=C_axial,
        r_mean=r_mean, area=A,
    ))

    for i in range(n_stages):
        delta_T = work_per_stage / gas.cp
        T_next = T_current - delta_T               # temperature drops

        # Stage expansion ratio from polytropic relation
        stage_er = (T_current / T_next) ** (gas.gamma / ((gas.gamma - 1) * eta_poly))

        # Isentropic efficiency for this stage
        T_next_ideal = T_current * (1.0 / stage_er) ** ((gas.gamma - 1) / gas.gamma)
        delta_T_ideal = T_current - T_next_ideal
        eta_isen = delta_T / delta_T_ideal if delta_T_ideal > 0 else eta_poly

        stage = meanline_turbine_stage(
            U=U, C_axial=C_axial, alpha_in=alpha_current,
            stage_expansion_ratio=stage_er, eta_stage=eta_isen,
            reaction=reaction, gas=gas, T_in=T_current, P_in=P_current,
        )
        stage.stage_number = i + 1

        # Radial blade angles via free-vortex
        radial_positions = np.linspace(r_hub, r_tip, max(radial_stations, 2))
        C_theta2_mean = stage.nozzle_triangles.outlet.C_theta
        C_theta3_mean = stage.rotor_triangles.outlet.C_theta
        radial_angles = []
        for r in radial_positions:
            ratio = r_mean / r if abs(r) > 1e-10 else 1.0
            C_theta2_r = C_theta2_mean * ratio
            C_theta3_r = C_theta3_mean * ratio
            U_r = omega * r

            # NGV absolute angles
            alpha_ngv_out_r = math.atan2(C_theta2_r, C_axial)

            # Rotor relative angles
            W_theta_in_r = C_theta2_r - U_r
            W_theta_out_r = C_theta3_r - U_r
            beta_in_r = math.atan2(W_theta_in_r, C_axial)
            beta_out_r = math.atan2(W_theta_out_r, C_axial)

            radial_angles.append({
                "r": float(r),
                "alpha_ngv_out": alpha_ngv_out_r,
                "beta_rotor_in": beta_in_r,
                "beta_rotor_out": beta_out_r,
            })

        stage.radial_blade_angles = radial_angles
        stage_results.append(stage)

        # Update for next stage
        P_current /= stage_er
        T_current = T_next
        # Rotor exit swirl becomes next stage inlet
        alpha_current = math.atan2(
            stage.rotor_triangles.outlet.C_theta, C_axial
        )

        stations.append(StationConditions(
            P_total=P_current, T_total=T_current,
            alpha=alpha_current, C_axial=C_axial,
            r_mean=r_mean, area=A,
        ))

    return TurbineResult(
        stages=stage_results,
        stations=stations,
        overall_expansion_ratio=P_inlet / P_current,
        overall_temperature_ratio=T_current / T_inlet,
        overall_efficiency=eta_poly,
        total_work=total_work,
        n_stages=n_stages,
    )


# ── Blade parameter extraction ───────────────────────


def meanline_to_turbine_blade_parameters(result: TurbineResult) -> list[dict]:
    """Convert turbine meanline results to blade geometry parameters.

    Returns:
        List of dicts, one per stage, with NGV and rotor angles/geometry.
    """
    params = []
    for stage in result.stages:
        a = stage.blade_angles_deg()

        # NGV (stator)
        ngv_inlet = a["nozzle_inlet_alpha"]
        ngv_outlet = a["nozzle_outlet_alpha"]
        ngv_stagger = (ngv_inlet + ngv_outlet) / 2.0
        ngv_camber = abs(ngv_outlet - ngv_inlet)

        # Rotor
        rotor_inlet = a["rotor_inlet_beta"]
        rotor_outlet = a["rotor_outlet_beta"]
        rotor_stagger = (rotor_inlet + rotor_outlet) / 2.0
        rotor_camber = abs(rotor_outlet - rotor_inlet)

        # Solidity estimate (turbine blades: slightly lower than compressor)
        ngv_solidity = 1.0 + 0.3 * stage.loading_coefficient
        rotor_solidity = 1.0 + 0.3 * stage.loading_coefficient

        params.append({
            "stage": stage.stage_number,
            "ngv_stagger_deg": ngv_stagger,
            "ngv_camber_deg": ngv_camber,
            "ngv_solidity": ngv_solidity,
            "ngv_inlet_alpha_deg": ngv_inlet,
            "ngv_outlet_alpha_deg": ngv_outlet,
            "rotor_stagger_deg": rotor_stagger,
            "rotor_camber_deg": rotor_camber,
            "rotor_solidity": rotor_solidity,
            "rotor_inlet_beta_deg": rotor_inlet,
            "rotor_outlet_beta_deg": rotor_outlet,
            "flow_coefficient": stage.flow_coefficient,
            "loading_coefficient": stage.loading_coefficient,
            "reaction": stage.degree_of_reaction,
            "zweifel": stage.zweifel_coefficient,
        })

    return params
