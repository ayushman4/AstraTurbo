"""Off-design performance and map generation for axial turbines.

At off-design conditions (different RPM or mass flow), blade metal angles
are fixed but flow angles change, creating incidence.  For turbines the key
phenomena are:
  - NGV: flow turning deviation, throat Mach and choking
  - Rotor: incidence loss via Soderberg correlation
  - Stage work changes with velocity triangle mismatch

Mirrors the compressor off-design / compressor map pattern but with
turbine-specific physics (expanding flow, choke limit instead of surge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .turbine import (
    GasProperties,
    TurbineResult,
    soderberg_loss,
    meanline_to_turbine_blade_parameters,
)
from .velocity_triangle import VelocityTriangle


# ── Single-stage off-design ─────────────────────────


def turbine_off_design_stage(
    ngv_alpha_metal_out: float,
    rotor_beta_metal_in: float,
    rotor_beta_metal_out: float,
    U: float,
    C_axial: float,
    Pt_in: float,
    Tt_in: float,
    alpha_in: float,
    gas: GasProperties | None = None,
) -> dict:
    """Analyse a single turbine stage at off-design conditions.

    Metal angles are fixed from the design point; flow angles change
    with operating conditions.

    Args:
        ngv_alpha_metal_out: NGV exit absolute metal angle (radians).
        rotor_beta_metal_in: Rotor inlet relative metal angle (radians).
        rotor_beta_metal_out: Rotor outlet relative metal angle (radians).
        U: Blade speed at mean radius (m/s).
        C_axial: Axial velocity (m/s).
        Pt_in: Inlet total pressure (Pa).
        Tt_in: Inlet total temperature (K).
        alpha_in: Absolute inlet flow angle (radians).
        gas: Gas properties.

    Returns:
        Dict with ER, efficiency, incidence, nozzle_mach, loss, work,
        is_choked, and outlet conditions for propagation to next stage.
    """
    if gas is None:
        gas = GasProperties()

    # ── 1. NGV (nozzle guide vane) ──
    # Apply a small deviation model: flow angle ≈ metal angle + deviation
    # For turbines, deviation is small (~1-2 deg); use a simple empirical
    deviation_nozzle_deg = 2.0  # degrees
    alpha_ngv_out = ngv_alpha_metal_out + math.radians(deviation_nozzle_deg)

    # NGV exit velocity
    C_theta2 = C_axial * math.tan(alpha_ngv_out)
    C_ngv_exit = math.sqrt(C_axial**2 + C_theta2**2)

    # NGV exit Mach (check for choking)
    T_static_ngv = Tt_in - C_ngv_exit**2 / (2.0 * gas.cp)
    T_static_ngv = max(T_static_ngv, 100.0)
    a_ngv = gas.speed_of_sound(T_static_ngv)
    nozzle_mach = C_ngv_exit / a_ngv if a_ngv > 0 else 0.0
    is_choked = nozzle_mach >= 1.0

    # NGV loss (Soderberg)
    ngv_deflection = abs(math.degrees(alpha_ngv_out) - math.degrees(alpha_in))
    ngv_loss = soderberg_loss(ngv_deflection)

    # ── 2. Rotor ──
    # Build rotor inlet triangle (C_theta2 from NGV, blade speed U)
    rotor_inlet = VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta2)

    # Incidence = actual relative flow angle - metal angle
    incidence_rad = rotor_inlet.beta - rotor_beta_metal_in
    incidence_deg = math.degrees(incidence_rad)

    # Rotor exit: flow follows metal angle + small deviation
    deviation_rotor_deg = 1.5
    beta_out_flow = rotor_beta_metal_out + math.radians(deviation_rotor_deg)

    # Rotor outlet triangle
    W_theta_out = C_axial * math.tan(beta_out_flow)
    C_theta3 = W_theta_out + U
    rotor_outlet = VelocityTriangle(U=U, C_axial=C_axial, C_theta=C_theta3)

    # Rotor deflection and loss
    rotor_deflection = abs(math.degrees(rotor_inlet.beta) - math.degrees(beta_out_flow))
    rotor_loss = soderberg_loss(rotor_deflection)

    # Extra incidence loss (quadratic incidence penalty)
    incidence_penalty = 0.0005 * incidence_deg**2
    total_loss = ngv_loss + rotor_loss + incidence_penalty

    # ── 3. Thermodynamics ──
    # Euler work = U * delta_C_theta (turbine: C_theta decreases, work positive)
    delta_C_theta = C_theta2 - C_theta3
    work = U * delta_C_theta

    # Temperature drop
    delta_T = work / gas.cp
    Tt_out = Tt_in - delta_T
    Tt_out = max(Tt_out, 200.0)  # safety floor

    # Expansion ratio from temperature and efficiency
    # Account for losses: actual eta = ideal - loss derating
    eta_ideal = 0.90
    eta_actual = max(0.30, eta_ideal - total_loss)

    if delta_T > 0 and Tt_in > 0:
        # From isentropic relation: ER = (T_in/T_out_ideal)^(gamma/(gamma-1))
        delta_T_ideal = delta_T / eta_actual if eta_actual > 0 else delta_T
        T_out_ideal = Tt_in - delta_T_ideal
        T_out_ideal = max(T_out_ideal, 200.0)
        ER = (Tt_in / T_out_ideal) ** (gas.gamma / (gas.gamma - 1))
        ER = max(ER, 1.0)
    else:
        ER = 1.0
        eta_actual = 0.0

    Pt_out = Pt_in / ER

    # Outlet absolute angle for propagation
    alpha_out = math.atan2(C_theta3, C_axial)

    return {
        "ER": ER,
        "efficiency": eta_actual,
        "incidence_deg": incidence_deg,
        "nozzle_mach": nozzle_mach,
        "ngv_loss": ngv_loss,
        "rotor_loss": rotor_loss,
        "incidence_loss": incidence_penalty,
        "loss_total": total_loss,
        "work": work,
        "is_choked": is_choked,
        "Pt_out": Pt_out,
        "Tt_out": Tt_out,
        "alpha_out": alpha_out,
    }


# ── Multi-stage off-design result ───────────────────


@dataclass
class TurbineOffDesignResult:
    """Result of an off-design turbine analysis."""

    stages: list[dict] = field(default_factory=list)
    overall_er: float = 1.0
    overall_efficiency: float = 0.0
    total_work: float = 0.0
    is_choked: bool = False
    choke_margin: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Turbine Off-Design Analysis: {len(self.stages)} stages",
            f"  Overall ER:     {self.overall_er:.4f}",
            f"  Overall eta:    {self.overall_efficiency:.4f}",
            f"  Total work:     {self.total_work:.0f} J/kg",
            f"  Choked:         {self.is_choked}",
            f"  Choke margin:   {self.choke_margin:.4f}",
            "",
        ]
        for s in self.stages:
            lines.append(f"  Stage {s.get('stage', '?')}:")
            lines.append(
                f"    ER={s['ER']:.4f}  eta={s['efficiency']:.4f}  "
                f"incidence={s['incidence_deg']:.2f} deg  "
                f"nozzle_M={s['nozzle_mach']:.3f}"
            )
        return "\n".join(lines)


# ── Multi-stage off-design function ─────────────────


def turbine_off_design(
    design_result: TurbineResult,
    mass_flow: float,
    rpm: float,
    gas: GasProperties | None = None,
    T_inlet: float | None = None,
    P_inlet: float | None = None,
) -> TurbineOffDesignResult:
    """Run off-design analysis of a complete axial turbine.

    Uses the blade geometry (metal angles) fixed at the design point and
    computes performance at different mass flow and RPM conditions.

    Args:
        design_result: TurbineResult from the design-point solver.
        mass_flow: Off-design mass flow rate (kg/s).
        rpm: Off-design rotational speed (RPM).
        gas: Gas properties (default air).
        T_inlet: Inlet total temperature (K). Defaults to design.
        P_inlet: Inlet total pressure (Pa). Defaults to design.

    Returns:
        TurbineOffDesignResult with per-stage and overall performance.
    """
    if gas is None:
        gas = GasProperties()

    # Defaults from design result
    if T_inlet is None:
        T_inlet = design_result.stations[0].T_total if design_result.stations else 1500.0
    if P_inlet is None:
        P_inlet = design_result.stations[0].P_total if design_result.stations else 101325.0

    # Extract metal angles from design
    blade_params = meanline_to_turbine_blade_parameters(design_result)

    # Geometry from design
    if design_result.stations:
        r_mean = design_result.stations[0].r_mean
        area = design_result.stations[0].area
    else:
        r_mean = 0.30
        area = 0.05

    # New operating conditions
    omega = rpm * 2 * math.pi / 60.0
    U = omega * r_mean
    rho_in = P_inlet / (gas.R * T_inlet)
    C_axial = mass_flow / (rho_in * area) if area > 0 else 200.0

    # Loop through stages
    Pt_current = P_inlet
    Tt_current = T_inlet
    alpha_current = 0.0

    stage_results = []
    any_choked = False
    total_work = 0.0
    max_nozzle_mach = 0.0

    for i, bp in enumerate(blade_params):
        # Metal angles from design (convert to radians)
        ngv_alpha_metal_out = math.radians(bp["ngv_outlet_alpha_deg"])
        rotor_beta_metal_in = math.radians(bp["rotor_inlet_beta_deg"])
        rotor_beta_metal_out = math.radians(bp["rotor_outlet_beta_deg"])

        stage_data = turbine_off_design_stage(
            ngv_alpha_metal_out=ngv_alpha_metal_out,
            rotor_beta_metal_in=rotor_beta_metal_in,
            rotor_beta_metal_out=rotor_beta_metal_out,
            U=U,
            C_axial=C_axial,
            Pt_in=Pt_current,
            Tt_in=Tt_current,
            alpha_in=alpha_current,
            gas=gas,
        )
        stage_data["stage"] = i + 1

        if stage_data["is_choked"]:
            any_choked = True
        total_work += stage_data["work"]
        max_nozzle_mach = max(max_nozzle_mach, stage_data["nozzle_mach"])

        # Propagate to next stage
        Pt_current = stage_data["Pt_out"]
        Tt_current = stage_data["Tt_out"]
        alpha_current = stage_data["alpha_out"]
        stage_results.append(stage_data)

    # Overall performance
    overall_er = P_inlet / Pt_current if Pt_current > 0 else 1.0

    # Overall efficiency
    if total_work > 0 and Tt_current < T_inlet:
        T_out_ideal = T_inlet * (1.0 / overall_er) ** ((gas.gamma - 1) / gas.gamma)
        delta_T_ideal = T_inlet - T_out_ideal
        delta_T_actual = T_inlet - Tt_current
        overall_eff = delta_T_actual / delta_T_ideal if delta_T_ideal > 0 else 0.0
        overall_eff = max(0.0, min(overall_eff, 1.0))
    else:
        overall_eff = 0.0

    # Choke margin: how far the maximum nozzle Mach is from 1.0
    choke_margin = 1.0 - max_nozzle_mach

    return TurbineOffDesignResult(
        stages=stage_results,
        overall_er=overall_er,
        overall_efficiency=overall_eff,
        total_work=total_work,
        is_choked=any_choked,
        choke_margin=choke_margin,
    )


# ── Turbine map dataclasses ─────────────────────────


@dataclass
class TurbineSpeedLine:
    """A single constant-speed operating line on the turbine map."""

    rpm_fraction: float = 1.0
    mass_flows: list[float] = field(default_factory=list)
    expansion_ratios: list[float] = field(default_factory=list)
    efficiencies: list[float] = field(default_factory=list)
    is_choked: list[bool] = field(default_factory=list)
    choke_point_index: int | None = None


@dataclass
class TurbineMap:
    """Complete turbine performance map."""

    speed_lines: list[TurbineSpeedLine] = field(default_factory=list)
    choke_line: list[tuple[float, float]] = field(default_factory=list)
    design_point: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Text table of all speed lines."""
        lines = [
            "Turbine Map Summary",
            "=" * 72,
        ]
        if self.design_point:
            dp = self.design_point
            lines.append(
                f"Design Point: m_dot={dp.get('mass_flow', 0):.2f} kg/s, "
                f"ER={dp.get('er', 0):.4f}, "
                f"eta={dp.get('efficiency', 0):.4f}, "
                f"RPM={dp.get('rpm', 0):.0f}"
            )
            lines.append("")

        for sl in self.speed_lines:
            lines.append(f"Speed Line: N/N_des = {sl.rpm_fraction:.2f}")
            lines.append(
                f"  {'m_dot (kg/s)':>14s}  {'ER':>8s}  {'eta':>8s}  "
                f"{'choked':>6s}"
            )
            lines.append("  " + "-" * 44)
            for j in range(len(sl.mass_flows)):
                choke_marker = " *CHOKE*" if sl.is_choked[j] else ""
                lines.append(
                    f"  {sl.mass_flows[j]:14.3f}  {sl.expansion_ratios[j]:8.4f}  "
                    f"{sl.efficiencies[j]:8.4f}  "
                    f"{'YES' if sl.is_choked[j] else 'no':>6s}"
                    f"{choke_marker}"
                )
            if sl.choke_point_index is not None:
                idx = sl.choke_point_index
                lines.append(
                    f"  Choke point: m_dot={sl.mass_flows[idx]:.3f}, "
                    f"ER={sl.expansion_ratios[idx]:.4f}"
                )
            lines.append("")

        if self.choke_line:
            lines.append("Choke Line:")
            for mf, er in self.choke_line:
                lines.append(f"  m_dot={mf:.3f} kg/s, ER={er:.4f}")

        return "\n".join(lines)


# ── Map generation ──────────────────────────────────


def generate_turbine_map(
    design_result: TurbineResult,
    rpm_fractions: list[float] | None = None,
    n_points: int = 15,
    gas: GasProperties | None = None,
    T_inlet: float | None = None,
    P_inlet: float | None = None,
) -> TurbineMap:
    """Generate a complete turbine performance map.

    Sweeps mass flow at several RPM fractions to produce speed lines,
    then identifies choke points and builds the choke line.

    Args:
        design_result: TurbineResult from design-point solver.
        rpm_fractions: List of N/N_design fractions to evaluate.
        n_points: Number of mass flow points per speed line.
        gas: Gas properties.
        T_inlet: Inlet total temperature (K). Defaults to design.
        P_inlet: Inlet total pressure (Pa). Defaults to design.

    Returns:
        TurbineMap with speed lines and choke line.
    """
    if gas is None:
        gas = GasProperties()

    if rpm_fractions is None:
        rpm_fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05]

    # Defaults from design
    if T_inlet is None:
        T_inlet = design_result.stations[0].T_total if design_result.stations else 1500.0
    if P_inlet is None:
        P_inlet = design_result.stations[0].P_total if design_result.stations else 101325.0

    # Extract design conditions
    if design_result.stations and len(design_result.stations) >= 2:
        station0 = design_result.stations[0]
        rho_design = station0.P_total / (gas.R * station0.T_total)
        design_mass_flow = rho_design * station0.area * station0.C_axial
        r_mean = station0.r_mean
    else:
        design_mass_flow = 20.0
        r_mean = 0.30

    # Design RPM from first stage rotor
    if design_result.stages:
        U_design = design_result.stages[0].rotor_triangles.inlet.U
        omega_design = U_design / r_mean if r_mean > 0 else 1000.0
        design_rpm = omega_design * 60.0 / (2 * math.pi)
    else:
        design_rpm = 17189.0

    # Run design point for reference
    design_od = turbine_off_design(
        design_result, design_mass_flow, design_rpm, gas, T_inlet, P_inlet
    )

    design_point_info = {
        "mass_flow": design_mass_flow,
        "er": design_od.overall_er,
        "efficiency": design_od.overall_efficiency,
        "rpm": design_rpm,
    }

    speed_lines = []
    choke_points = []

    for frac in sorted(rpm_fractions):
        actual_rpm = frac * design_rpm

        # Mass flow range: scale by RPM fraction for approximate similarity
        mf_min = 0.50 * design_mass_flow * frac
        mf_max = 1.30 * design_mass_flow * frac  # turbines can swallow more flow
        mass_flows_sweep = [
            mf_min + (mf_max - mf_min) * i / max(n_points - 1, 1)
            for i in range(n_points)
        ]

        sl = TurbineSpeedLine(rpm_fraction=frac)
        choke_idx = None

        for mf in mass_flows_sweep:
            if mf <= 0:
                continue
            try:
                result = turbine_off_design(
                    design_result, mf, actual_rpm, gas, T_inlet, P_inlet
                )
                sl.mass_flows.append(mf)
                sl.expansion_ratios.append(result.overall_er)
                sl.efficiencies.append(result.overall_efficiency)
                sl.is_choked.append(result.is_choked)

                # First choked point
                if result.is_choked and choke_idx is None:
                    choke_idx = len(sl.mass_flows) - 1
            except Exception:
                continue

        sl.choke_point_index = choke_idx
        speed_lines.append(sl)

        if choke_idx is not None and choke_idx < len(sl.mass_flows):
            choke_points.append(
                (sl.mass_flows[choke_idx], sl.expansion_ratios[choke_idx])
            )

    return TurbineMap(
        speed_lines=speed_lines,
        choke_line=choke_points,
        design_point=design_point_info,
    )
