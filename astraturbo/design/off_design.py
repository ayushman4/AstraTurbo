"""Off-design meanline analysis for axial compressors.

At off-design conditions (different RPM or mass flow), blade metal angles
are fixed but flow angles change, creating incidence. This module computes
the resulting changes in losses, efficiency, and pressure ratio.

Uses the same loss models as design-point analysis:
  - Lieblein diffusion factor and profile loss
  - Ainley-Mathieson secondary loss
  - Tip clearance loss
  - Carter's deviation rule
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .meanline import (
    GasProperties,
    MeanlineResult,
    meanline_to_blade_parameters,
)
from .velocity_triangle import VelocityTriangle, compute_triangle_from_beta
from ..solver.loss_models import (
    lieblein_diffusion_factor,
    lieblein_profile_loss,
    ainley_mathieson_secondary_loss,
    tip_clearance_loss,
    carter_deviation,
)


@dataclass
class OffDesignResult:
    """Result of an off-design compressor analysis."""

    stages: list[dict] = field(default_factory=list)
    overall_pr: float = 1.0
    overall_efficiency: float = 0.0
    overall_temperature_ratio: float = 1.0
    total_work: float = 0.0
    is_stalled: bool = False
    is_choked: bool = False
    surge_margin: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Off-Design Analysis: {len(self.stages)} stages",
            f"  Overall PR:   {self.overall_pr:.4f}",
            f"  Overall TR:   {self.overall_temperature_ratio:.4f}",
            f"  Overall eta:  {self.overall_efficiency:.4f}",
            f"  Total work:   {self.total_work:.0f} J/kg",
            f"  Stalled:      {self.is_stalled}",
            f"  Choked:       {self.is_choked}",
            f"  Surge margin: {self.surge_margin:.4f}",
            "",
        ]
        for s in self.stages:
            lines.append(f"  Stage {s.get('stage', '?')}:")
            lines.append(
                f"    PR={s['PR']:.4f}  eta={s['efficiency']:.4f}  "
                f"DF={s['DF']:.4f}  incidence={s['incidence_deg']:.2f} deg"
            )
        return "\n".join(lines)


def off_design_stage(
    beta_metal_in: float,
    beta_metal_out: float,
    camber_deg: float,
    stagger_deg: float,
    solidity: float,
    chord: float,
    span: float,
    tip_clearance: float,
    U: float,
    C_axial: float,
    Pt_in: float,
    Tt_in: float,
    alpha_in: float,
    gas: GasProperties | None = None,
) -> dict:
    """Analyse a single compressor stage at off-design conditions.

    Metal angles are fixed from design; flow angles change with conditions.

    Args:
        beta_metal_in: Design rotor-relative inlet metal angle (radians).
        beta_metal_out: Design rotor-relative outlet metal angle (radians).
        camber_deg: Blade camber angle (degrees).
        stagger_deg: Blade stagger angle (degrees).
        solidity: Blade solidity (chord / pitch).
        chord: Blade chord length (m).
        span: Blade span (m).
        tip_clearance: Tip clearance gap (m).
        U: Blade speed at mean radius (m/s).
        C_axial: Axial velocity (m/s).
        Pt_in: Inlet total pressure (Pa).
        Tt_in: Inlet total temperature (K).
        alpha_in: Absolute inlet flow angle (radians).
        gas: Gas properties.

    Returns:
        Dict with PR, efficiency, DF, incidence, deviation, losses, flags.
    """
    if gas is None:
        gas = GasProperties()

    # 1. Build inlet velocity triangle
    C_theta_in = C_axial * math.tan(alpha_in)
    inlet_tri = VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta_in)

    # Actual flow relative inlet angle
    beta_flow_in = inlet_tri.beta

    # 2. Incidence
    incidence_rad = beta_flow_in - beta_metal_in
    incidence_deg = math.degrees(incidence_rad)

    # 3. Carter deviation
    deviation_deg = carter_deviation(camber_deg, solidity, stagger_deg)

    # 4. Outlet relative angle = metal angle + deviation (in radians)
    beta_out_flow = beta_metal_out + math.radians(deviation_deg)

    # 5. Outlet velocity triangle
    outlet_tri = compute_triangle_from_beta(U, C_axial, beta_out_flow)

    # 6. Diffusion factor
    W_in = inlet_tri.W
    W_out = outlet_tri.W
    W_theta_in = inlet_tri.W_theta
    W_theta_out = outlet_tri.W_theta
    DF = lieblein_diffusion_factor(W_in, W_out, W_theta_in, W_theta_out, solidity)

    # 7. Losses
    Re = max(1.0, gas.R * Tt_in / 1.5e-5 * W_in * chord / (gas.R * Tt_in))
    # Approximate Re = rho * W * chord / mu
    rho_in = Pt_in / (gas.R * Tt_in)
    mu = 1.8e-5  # dynamic viscosity of air at ~300K
    Re = rho_in * W_in * chord / mu if chord > 0 else 2e5

    omega_profile = lieblein_profile_loss(DF, solidity, Re)

    beta_in_deg = math.degrees(beta_flow_in)
    beta_out_deg = math.degrees(beta_out_flow)
    omega_secondary = ainley_mathieson_secondary_loss(
        abs(beta_in_deg), abs(beta_out_deg), span, chord
    )

    # Loading parameter for tip clearance loss
    delta_C_theta = outlet_tri.C_theta - inlet_tri.C_theta
    loading_param = abs(delta_C_theta / U) if U > 0 else 0.5
    omega_tc = tip_clearance_loss(tip_clearance, span, loading_param)

    omega_total = omega_profile + omega_secondary + omega_tc

    # 8. Total pressure loss
    dynamic_pressure_in = 0.5 * rho_in * W_in**2
    delta_Pt_loss = omega_total * dynamic_pressure_in

    # 9. Ideal exit total pressure (from Euler work)
    work = U * delta_C_theta  # J/kg
    delta_T = work / gas.cp
    Tt_out = Tt_in + delta_T

    if Tt_out > Tt_in and Tt_in > 0:
        Pt_out_ideal = Pt_in * (Tt_out / Tt_in) ** (gas.gamma / (gas.gamma - 1))
    else:
        Pt_out_ideal = Pt_in

    Pt_out = Pt_out_ideal - delta_Pt_loss
    Pt_out = max(Pt_out, Pt_in * 0.5)  # prevent unphysical negatives

    # 10. Stage PR
    PR = Pt_out / Pt_in

    # 11. Isentropic efficiency
    if delta_T > 0:
        Tt_out_ideal = Tt_in * PR ** ((gas.gamma - 1) / gas.gamma)
        delta_T_ideal = Tt_out_ideal - Tt_in
        efficiency = delta_T_ideal / delta_T if delta_T > 0 else 0.0
        efficiency = max(0.0, min(efficiency, 1.0))
    else:
        efficiency = 0.0

    # 12. Stall flag
    stalled = DF > 0.6

    # 13. Choke flag — throat Mach from relative velocity and local sound speed
    T_static = Tt_in - (inlet_tri.C**2) / (2 * gas.cp)
    T_static = max(T_static, 100.0)
    a_local = gas.speed_of_sound(T_static)
    throat_mach = W_in / a_local if a_local > 0 else 0.0
    choked = throat_mach >= 1.0

    # Outlet absolute angle for propagation to next stage
    alpha_out = outlet_tri.alpha

    return {
        "PR": PR,
        "efficiency": efficiency,
        "DF": DF,
        "incidence_deg": incidence_deg,
        "deviation_deg": deviation_deg,
        "loss_profile": omega_profile,
        "loss_secondary": omega_secondary,
        "loss_tip_clearance": omega_tc,
        "loss_total": omega_total,
        "is_stalled": stalled,
        "is_choked": choked,
        "work": work,
        "Pt_out": Pt_out,
        "Tt_out": Tt_out,
        "alpha_out": alpha_out,
        "throat_mach": throat_mach,
    }


def off_design_compressor(
    design_result: MeanlineResult,
    mass_flow: float,
    rpm: float,
    gas: GasProperties | None = None,
    T_inlet: float = 288.15,
    P_inlet: float = 101325.0,
) -> OffDesignResult:
    """Run off-design analysis of a complete compressor.

    Uses the blade geometry fixed at design point and computes performance
    at different mass flow and RPM conditions.

    Args:
        design_result: MeanlineResult from the design-point solver.
        mass_flow: Off-design mass flow rate (kg/s).
        rpm: Off-design rotational speed (RPM).
        gas: Gas properties.
        T_inlet: Inlet total temperature (K).
        P_inlet: Inlet total pressure (Pa).

    Returns:
        OffDesignResult with per-stage and overall performance.
    """
    if gas is None:
        gas = GasProperties()

    # Extract blade geometry from design result
    blade_params = meanline_to_blade_parameters(design_result)

    # Geometry from design
    if design_result.stations:
        r_mean = design_result.stations[0].r_mean
        area = design_result.stations[0].area
    else:
        r_mean = 0.2
        area = 0.05

    # New operating conditions
    omega = rpm * 2 * math.pi / 60.0
    U = omega * r_mean
    rho_in = P_inlet / (gas.R * T_inlet)
    C_axial = mass_flow / (rho_in * area) if area > 0 else 150.0

    # Loop through stages
    Pt_current = P_inlet
    Tt_current = T_inlet
    # Initialize inlet swirl from the design-point velocity triangle.
    # The design solver may place nonzero C_theta at the first rotor inlet
    # (via the reaction equation), implying an IGV or repeating-stage swirl.
    if design_result.stages:
        alpha_current = design_result.stages[0].rotor_triangles.inlet.alpha
    else:
        alpha_current = 0.0

    stage_results = []
    any_stalled = False
    any_choked = False
    total_work = 0.0

    for i, bp in enumerate(blade_params):
        # Metal angles from design (convert back to radians)
        beta_metal_in = math.radians(bp["rotor_inlet_beta_deg"])
        beta_metal_out = math.radians(bp["rotor_outlet_beta_deg"])
        camber_deg = bp["rotor_camber_deg"]
        stagger_deg = bp["rotor_stagger_deg"]
        solidity = bp["rotor_solidity"]

        # Estimate chord and span from geometry
        span = math.sqrt(area / math.pi) if area > 0 else 0.05
        # Use aspect ratio ~2 for chord estimate
        chord = span / 2.0
        # Tip clearance ~ 1% of span
        tip_clearance = 0.01 * span

        stage_data = off_design_stage(
            beta_metal_in=beta_metal_in,
            beta_metal_out=beta_metal_out,
            camber_deg=camber_deg,
            stagger_deg=stagger_deg,
            solidity=solidity,
            chord=chord,
            span=span,
            tip_clearance=tip_clearance,
            U=U,
            C_axial=C_axial,
            Pt_in=Pt_current,
            Tt_in=Tt_current,
            alpha_in=alpha_current,
            gas=gas,
        )
        stage_data["stage"] = i + 1

        if stage_data["is_stalled"]:
            any_stalled = True
        if stage_data["is_choked"]:
            any_choked = True
        total_work += stage_data["work"]

        # Propagate to next stage
        Pt_current = stage_data["Pt_out"]
        Tt_current = stage_data["Tt_out"]
        alpha_current = stage_data["alpha_out"]
        stage_results.append(stage_data)

    # Overall performance
    overall_pr = Pt_current / P_inlet
    overall_tr = Tt_current / T_inlet

    # Overall efficiency
    if overall_tr > 1.0:
        T_out_ideal = T_inlet * overall_pr ** ((gas.gamma - 1) / gas.gamma)
        delta_T_ideal = T_out_ideal - T_inlet
        delta_T_actual = Tt_current - T_inlet
        overall_eff = delta_T_ideal / delta_T_actual if delta_T_actual > 0 else 0.0
        overall_eff = max(0.0, min(overall_eff, 1.0))
    else:
        overall_eff = 0.0

    return OffDesignResult(
        stages=stage_results,
        overall_pr=overall_pr,
        overall_efficiency=overall_eff,
        overall_temperature_ratio=overall_tr,
        total_work=total_work,
        is_stalled=any_stalled,
        is_choked=any_choked,
        surge_margin=0.0,  # computed by map generator
    )
