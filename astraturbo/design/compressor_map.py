"""Compressor map generation from off-design meanline analysis.

Sweeps mass flow and RPM to produce speed lines, then connects
surge points to form the surge line. This is the standard way
to characterise compressor operability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .meanline import MeanlineResult, GasProperties
from .off_design import off_design_compressor, OffDesignResult


@dataclass
class SpeedLine:
    """A single constant-speed operating line on the compressor map."""

    rpm_fraction: float = 1.0
    mass_flows: list[float] = field(default_factory=list)
    pressure_ratios: list[float] = field(default_factory=list)
    efficiencies: list[float] = field(default_factory=list)
    is_stalled: list[bool] = field(default_factory=list)
    is_choked: list[bool] = field(default_factory=list)
    surge_point_index: int | None = None


@dataclass
class CompressorMap:
    """Complete compressor performance map."""

    speed_lines: list[SpeedLine] = field(default_factory=list)
    surge_line: list[tuple[float, float]] = field(default_factory=list)
    design_point: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Text table of all speed lines."""
        lines = [
            "Compressor Map Summary",
            "=" * 72,
        ]
        if self.design_point:
            dp = self.design_point
            lines.append(
                f"Design Point: m_dot={dp.get('mass_flow', 0):.2f} kg/s, "
                f"PR={dp.get('pr', 0):.4f}, "
                f"eta={dp.get('efficiency', 0):.4f}, "
                f"RPM={dp.get('rpm', 0):.0f}"
            )
            lines.append("")

        for sl in self.speed_lines:
            lines.append(f"Speed Line: N/N_des = {sl.rpm_fraction:.2f}")
            lines.append(
                f"  {'m_dot (kg/s)':>14s}  {'PR':>8s}  {'eta':>8s}  "
                f"{'DF_stall':>8s}  {'choked':>6s}"
            )
            lines.append("  " + "-" * 52)
            for j in range(len(sl.mass_flows)):
                stall_marker = " *STALL*" if sl.is_stalled[j] else ""
                choke_marker = " *CHOKE*" if sl.is_choked[j] else ""
                lines.append(
                    f"  {sl.mass_flows[j]:14.3f}  {sl.pressure_ratios[j]:8.4f}  "
                    f"{sl.efficiencies[j]:8.4f}  "
                    f"{'YES' if sl.is_stalled[j] else 'no':>8s}  "
                    f"{'YES' if sl.is_choked[j] else 'no':>6s}"
                    f"{stall_marker}{choke_marker}"
                )
            if sl.surge_point_index is not None:
                idx = sl.surge_point_index
                lines.append(
                    f"  Surge point: m_dot={sl.mass_flows[idx]:.3f}, "
                    f"PR={sl.pressure_ratios[idx]:.4f}"
                )
            lines.append("")

        if self.surge_line:
            lines.append("Surge Line:")
            for mf, pr in self.surge_line:
                lines.append(f"  m_dot={mf:.3f} kg/s, PR={pr:.4f}")

        return "\n".join(lines)


def generate_compressor_map(
    design_result: MeanlineResult,
    rpm_fractions: list[float] | None = None,
    n_points: int = 15,
    gas: GasProperties | None = None,
    T_inlet: float = 288.15,
    P_inlet: float = 101325.0,
) -> CompressorMap:
    """Generate a complete compressor performance map.

    Sweeps mass flow at several RPM fractions to produce speed lines,
    then identifies surge points and builds the surge line.

    Args:
        design_result: MeanlineResult from design-point solver.
        rpm_fractions: List of N/N_design fractions to evaluate.
        n_points: Number of mass flow points per speed line.
        gas: Gas properties.
        T_inlet: Inlet total temperature (K).
        P_inlet: Inlet total pressure (Pa).

    Returns:
        CompressorMap with speed lines and surge line.
    """
    if gas is None:
        gas = GasProperties()

    if rpm_fractions is None:
        rpm_fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05]

    # Extract design-point conditions from stations
    if design_result.stations and len(design_result.stations) >= 2:
        station0 = design_result.stations[0]
        rho_design = station0.P_total / (gas.R * station0.T_total)
        design_mass_flow = rho_design * station0.area * station0.C_axial
        r_mean = station0.r_mean
    else:
        design_mass_flow = 20.0
        r_mean = 0.2

    # Design RPM from first station
    if design_result.stages:
        U_design = design_result.stages[0].rotor_triangles.inlet.U
        omega_design = U_design / r_mean if r_mean > 0 else 1000.0
        design_rpm = omega_design * 60.0 / (2 * math.pi)
    else:
        design_rpm = 15000.0

    # Run design point for reference
    design_od = off_design_compressor(
        design_result, design_mass_flow, design_rpm, gas, T_inlet, P_inlet
    )

    design_point_info = {
        "mass_flow": design_mass_flow,
        "pr": design_od.overall_pr,
        "efficiency": design_od.overall_efficiency,
        "rpm": design_rpm,
    }

    speed_lines = []
    surge_points = []

    for frac in sorted(rpm_fractions):
        actual_rpm = frac * design_rpm

        # Mass flow range: scale by RPM fraction for approximate similarity
        mf_min = 0.50 * design_mass_flow * frac
        mf_max = 1.10 * design_mass_flow * frac
        mass_flows_sweep = [
            mf_min + (mf_max - mf_min) * i / max(n_points - 1, 1)
            for i in range(n_points)
        ]

        sl = SpeedLine(rpm_fraction=frac)
        surge_idx = None

        for mf in mass_flows_sweep:
            if mf <= 0:
                continue
            try:
                result = off_design_compressor(
                    design_result, mf, actual_rpm, gas, T_inlet, P_inlet
                )
                sl.mass_flows.append(mf)
                sl.pressure_ratios.append(result.overall_pr)
                sl.efficiencies.append(result.overall_efficiency)
                sl.is_stalled.append(result.is_stalled)
                sl.is_choked.append(result.is_choked)

                # First stalled point is surge
                if result.is_stalled and surge_idx is None:
                    surge_idx = len(sl.mass_flows) - 1
            except Exception:
                # Skip points that fail numerically
                continue

        sl.surge_point_index = surge_idx
        speed_lines.append(sl)

        if surge_idx is not None and surge_idx < len(sl.mass_flows):
            surge_points.append(
                (sl.mass_flows[surge_idx], sl.pressure_ratios[surge_idx])
            )

    # Compute surge margin at design speed
    cmap = CompressorMap(
        speed_lines=speed_lines,
        surge_line=surge_points,
        design_point=design_point_info,
    )

    return cmap
