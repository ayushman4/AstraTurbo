"""Centrifugal (radial) compressor meanline design.

Designs a single-stage centrifugal compressor from top-level requirements:
impeller + diffuser. Suitable for eVTOL, drone, turbocharger, and small
turboshaft applications.

Key differences from axial:
  - Flow turns from axial to radial inside the impeller
  - Work input via centrifugal effect: w = U2*C_theta2 - U1*C_theta1
  - Slip factor reduces ideal work (Wiesner correlation)
  - Diffuser recovers kinetic energy after impeller discharge

References:
    Aungier, R.H., "Centrifugal Compressors", ASME Press, 2000.
    Japikse, D., "Centrifugal Compressor Design and Performance", 1996.
    Dixon & Hall, "Fluid Mechanics and Thermodynamics of Turbomachinery", 7th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .meanline import GasProperties


@dataclass
class CentrifugalResult:
    """Result of centrifugal compressor meanline design."""

    # Overall
    pressure_ratio: float = 1.0
    isentropic_efficiency: float = 0.0
    polytropic_efficiency: float = 0.0
    mass_flow: float = 0.0
    rpm: float = 0.0
    power_kW: float = 0.0
    tip_speed: float = 0.0
    specific_speed: float = 0.0

    # Impeller details
    impeller: dict = field(default_factory=dict)

    # Diffuser details
    diffuser: dict = field(default_factory=dict)

    # Inlet/outlet conditions
    T_inlet: float = 288.15
    T_outlet: float = 0.0
    P_inlet: float = 101325.0
    P_outlet: float = 0.0

    def summary(self) -> str:
        lines = [
            "Centrifugal Compressor Design",
            f"  PR:         {self.pressure_ratio:.4f}",
            f"  Efficiency: {self.isentropic_efficiency:.4f} (isentropic)",
            f"  Mass flow:  {self.mass_flow:.3f} kg/s",
            f"  RPM:        {self.rpm:.0f}",
            f"  Power:      {self.power_kW:.1f} kW",
            f"  Tip speed:  {self.tip_speed:.0f} m/s",
            f"  Ns:         {self.specific_speed:.3f}",
            "",
            "  Impeller:",
        ]
        imp = self.impeller
        if imp:
            lines.append(f"    r1_tip:     {imp.get('r1_tip', 0):.4f} m")
            lines.append(f"    r2:         {imp.get('r2', 0):.4f} m")
            lines.append(f"    Backsweep:  {imp.get('beta2_blade_deg', 0):.1f} deg")
            lines.append(f"    Slip:       {imp.get('slip_factor', 0):.4f}")
            lines.append(f"    Blades:     {imp.get('n_blades', 0)}")
            lines.append(f"    M_w1:       {imp.get('M_w1', 0):.3f}")
            lines.append(f"    U2:         {imp.get('U2', 0):.0f} m/s")

        lines.append("")
        lines.append("  Diffuser:")
        diff = self.diffuser
        if diff:
            lines.append(f"    r3:         {diff.get('r3', 0):.4f} m")
            lines.append(f"    r4:         {diff.get('r4', 0):.4f} m")
            lines.append(f"    Type:       {diff.get('type', 'vaneless')}")
            lines.append(f"    Cp:         {diff.get('Cp', 0):.4f}")

        return "\n".join(lines)


def wiesner_slip_factor(n_blades: int, beta2_blade_deg: float) -> float:
    """Wiesner slip factor correlation.

    sigma = 1 - sqrt(cos(beta2)) / n_blades^0.7

    Args:
        n_blades: Number of impeller blades.
        beta2_blade_deg: Impeller exit blade angle (degrees from radial).
            0 = radial blades, negative = backsweep.

    Returns:
        Slip factor (0-1). Typical range 0.85-0.95.
    """
    beta2_rad = math.radians(abs(beta2_blade_deg))
    cos_beta2 = math.cos(beta2_rad)
    sigma = 1.0 - math.sqrt(cos_beta2) / max(n_blades, 1) ** 0.7
    return max(0.5, min(sigma, 0.99))


def centrifugal_compressor(
    pressure_ratio: float,
    mass_flow: float,
    rpm: float,
    r1_hub: float = 0.02,
    r1_tip: float = 0.05,
    r2: float | None = None,
    beta2_blade_deg: float = -30.0,
    n_blades: int = 17,
    diffuser_ratio: float = 1.6,
    eta_impeller: float = 0.88,
    eta_diffuser: float = 0.75,
    gas: GasProperties | None = None,
    T_inlet: float = 288.15,
    P_inlet: float = 101325.0,
) -> CentrifugalResult:
    """Design a single-stage centrifugal compressor.

    Args:
        pressure_ratio: Target total-to-total pressure ratio.
        mass_flow: Mass flow rate (kg/s).
        rpm: Rotational speed (rev/min).
        r1_hub: Impeller inlet hub radius (m).
        r1_tip: Impeller inlet tip radius (m).
        r2: Impeller exit radius (m). If None, auto-sized from tip speed.
        beta2_blade_deg: Impeller exit blade angle (degrees from radial).
            Negative = backsweep. Typical: -20 to -40 deg.
        n_blades: Number of impeller blades (typically 15-21).
        diffuser_ratio: Diffuser exit/inlet radius ratio (typically 1.4-1.8).
        eta_impeller: Impeller isentropic efficiency.
        eta_diffuser: Diffuser pressure recovery efficiency.
        gas: Gas properties.
        T_inlet: Inlet total temperature (K).
        P_inlet: Inlet total pressure (Pa).

    Returns:
        CentrifugalResult with impeller and diffuser details.
    """
    if gas is None:
        gas = GasProperties()

    omega = rpm * 2 * math.pi / 60.0
    gamma = gas.gamma
    cp = gas.cp

    # Isentropic temperature rise for target PR
    T2s = T_inlet * pressure_ratio ** ((gamma - 1) / gamma)
    delta_Ts = T2s - T_inlet

    # Overall isentropic efficiency (impeller dominates)
    eta_overall = eta_impeller * eta_diffuser / (
        eta_diffuser + eta_impeller * (1 - eta_diffuser) * 0.5
    )
    # Simplified: eta_overall ~ 0.80 * eta_impeller typically
    eta_overall = max(0.5, min(eta_overall, 0.95))

    # Actual temperature rise
    delta_T_actual = delta_Ts / eta_overall
    T_outlet = T_inlet + delta_T_actual

    # Required work
    work = cp * delta_T_actual  # J/kg
    power = work * mass_flow  # W

    # Tip speed required: U2^2 ~ work / (slip * (1 - C_theta1/U2))
    # First estimate without slip for sizing
    slip = wiesner_slip_factor(n_blades, beta2_blade_deg)

    # For backswept impeller: w = slip * U2^2 * (1 + tan(beta2)*C_r2/U2)
    # Simplified: w ~ slip * U2^2 (for small backsweep)
    # Include backsweep correction
    beta2_rad = math.radians(beta2_blade_deg)
    # work = slip * U2^2 + U2 * C_r2 * tan(beta2)
    # Approximate: work ~ slip * U2^2 * (1 - |tan(beta2)| * phi_2)
    # where phi_2 = C_r2 / U2 ~ 0.25-0.35
    phi2_est = 0.30
    work_coeff = slip * (1.0 + math.tan(beta2_rad) * phi2_est)
    U2_required = math.sqrt(work / max(work_coeff, 0.3))

    # Size impeller exit radius
    if r2 is None:
        r2 = U2_required / omega
        # Ensure r2 > r1_tip but don't over-size
        r2 = max(r2, r1_tip * 1.2)
        # Limit r2 to prevent over-speed
        r2 = min(r2, r1_tip * 4.0)

    U2 = omega * r2
    U1_tip = omega * r1_tip
    U1_hub = omega * r1_hub
    r1_mean = (r1_hub + r1_tip) / 2.0

    # Inlet conditions
    rho_inlet = P_inlet / (gas.R * T_inlet)
    A1 = math.pi * (r1_tip**2 - r1_hub**2)
    C_axial1 = mass_flow / (rho_inlet * A1) if A1 > 0 else 100.0

    # Inlet velocity triangle (axial entry, no pre-swirl)
    C1 = C_axial1
    W1_tip = math.sqrt(C_axial1**2 + U1_tip**2)
    T1_static = T_inlet - C1**2 / (2 * cp)
    T1_static = max(T1_static, 200.0)
    a1 = gas.speed_of_sound(T1_static)
    M_w1 = W1_tip / a1  # Inlet relative Mach at tip

    # Impeller exit
    # Radial velocity at exit (continuity)
    # Assume exit blade height b2 ~ 0.1 * r2 (typical)
    b2 = 0.10 * r2
    rho_2_est = rho_inlet * (1 + delta_T_actual / T_inlet) * pressure_ratio ** (1 / gamma)
    rho_2_est = max(rho_2_est, rho_inlet)  # density increases through compressor
    A2 = 2 * math.pi * r2 * b2
    C_r2 = mass_flow / (rho_2_est * A2) if A2 > 0 else 50.0

    # Tangential velocity at impeller exit (with slip)
    C_theta2_ideal = U2 + C_r2 * math.tan(beta2_rad)
    C_theta2 = slip * C_theta2_ideal

    # Actual work
    # Assuming no inlet swirl (C_theta1 = 0)
    work_actual = U2 * C_theta2
    delta_T_actual_2 = work_actual / cp
    T2_total = T_inlet + delta_T_actual_2

    # Impeller exit total pressure (with efficiency)
    T2s_imp = T_inlet + delta_T_actual_2 * eta_impeller
    PR_impeller = (T2s_imp / T_inlet) ** (gamma / (gamma - 1))

    # Impeller exit velocity
    C2 = math.sqrt(C_r2**2 + C_theta2**2)
    alpha2 = math.degrees(math.atan2(C_theta2, C_r2))

    # Diffuser
    r3 = r2 * 1.05  # Small vaneless gap
    r4 = r2 * diffuser_ratio

    # Vaneless diffuser: conservation of angular momentum
    # C_theta3 = C_theta2 * r2 / r3 (approximately)
    C_theta4 = C_theta2 * r2 / r4
    # Radial velocity decreases
    C_r4 = C_r2 * (r2 * b2) / (r4 * b2 * 1.2)  # assume b4 ~ 1.2 * b2
    C4 = math.sqrt(C_r4**2 + C_theta4**2)

    # Diffuser pressure recovery
    Cp_diffuser = 1.0 - (C4 / C2) ** 2
    Cp_diffuser = max(0.0, min(Cp_diffuser * eta_diffuser, 0.8))

    # Overall PR — use impeller PR derated by diffuser losses
    # The diffuser recovers kinetic energy but also has friction losses
    PR_actual = PR_impeller * (1.0 - (1.0 - Cp_diffuser) * (C2**2) / (2 * cp * T2_total) * 0.5)
    PR_actual = max(PR_actual, 1.0)

    P_outlet = P_inlet * PR_actual

    # Recalculate efficiency
    T_out_ideal = T_inlet * PR_actual ** ((gamma - 1) / gamma)
    eta_isen = (T_out_ideal - T_inlet) / (T2_total - T_inlet) if T2_total > T_inlet else 0.0
    eta_isen = max(0.0, min(eta_isen, 1.0))

    # Polytropic efficiency
    if PR_actual > 1.0 and T2_total > T_inlet:
        eta_poly = (
            (gamma - 1) / gamma
            * math.log(PR_actual)
            / math.log(T2_total / T_inlet)
        )
        eta_poly = max(0.0, min(eta_poly, 1.0))
    else:
        eta_poly = eta_isen

    # Specific speed: Ns = omega * sqrt(Q) / (delta_h_s)^0.75
    Q_inlet = mass_flow / rho_inlet  # volume flow
    delta_hs = cp * delta_Ts
    if delta_hs > 0:
        Ns = omega * math.sqrt(Q_inlet) / delta_hs**0.75
    else:
        Ns = 0.0

    return CentrifugalResult(
        pressure_ratio=PR_actual,
        isentropic_efficiency=eta_isen,
        polytropic_efficiency=eta_poly,
        mass_flow=mass_flow,
        rpm=rpm,
        power_kW=power / 1000.0,
        tip_speed=U2,
        specific_speed=Ns,
        T_inlet=T_inlet,
        T_outlet=T2_total,
        P_inlet=P_inlet,
        P_outlet=P_outlet,
        impeller={
            "r1_hub": r1_hub,
            "r1_tip": r1_tip,
            "r1_mean": r1_mean,
            "r2": r2,
            "b2": b2,
            "U1_tip": U1_tip,
            "U2": U2,
            "C_axial1": C_axial1,
            "C_r2": C_r2,
            "C_theta2": C_theta2,
            "C2": C2,
            "alpha2_deg": alpha2,
            "beta2_blade_deg": beta2_blade_deg,
            "slip_factor": slip,
            "n_blades": n_blades,
            "M_w1": M_w1,
            "work_J_kg": work_actual,
            "PR_impeller": PR_impeller,
        },
        diffuser={
            "r3": r3,
            "r4": r4,
            "type": "vaneless",
            "C2": C2,
            "C4": C4,
            "Cp": Cp_diffuser,
            "alpha2_deg": alpha2,
        },
    )
