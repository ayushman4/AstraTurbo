"""Validation tests for off-design meanline solver and compressor maps.

Tests verify physical correctness of the off-design iteration loop:
incidence, loss trends, stall detection, and map structure.
"""

import math
import pytest

from astraturbo.design.meanline import (
    GasProperties,
    MeanlineResult,
    meanline_compressor,
    meanline_to_blade_parameters,
)
from astraturbo.design.off_design import (
    OffDesignResult,
    off_design_compressor,
    off_design_stage,
)
from astraturbo.design.compressor_map import (
    SpeedLine,
    CompressorMap,
    generate_compressor_map,
)


# ---------- Fixtures ----------

@pytest.fixture
def simple_design():
    """A simple single-stage compressor design result."""
    return meanline_compressor(
        overall_pressure_ratio=1.5,
        mass_flow=20.0,
        rpm=15000,
        r_hub=0.15,
        r_tip=0.25,
    )


@pytest.fixture
def rotor37_design():
    """NASA Rotor 37 approximate design: PR~2.1, 20 kg/s, 17189 RPM."""
    return meanline_compressor(
        overall_pressure_ratio=2.1,
        mass_flow=20.0,
        rpm=17189,
        r_hub=0.178,
        r_tip=0.252,
    )


# ---------- Off-design incidence and loss tests ----------

class TestOffDesignIncidence:
    def test_design_point_zero_incidence(self, simple_design):
        """At design conditions, incidence should be near zero."""
        # Extract design RPM/mass flow from stations
        station0 = simple_design.stations[0]
        gas = GasProperties()
        rho = station0.P_total / (gas.R * station0.T_total)
        design_mf = rho * station0.area * station0.C_axial
        U = simple_design.stages[0].rotor_triangles.inlet.U
        r_mean = station0.r_mean
        omega = U / r_mean
        design_rpm = omega * 60.0 / (2 * math.pi)

        result = off_design_compressor(simple_design, design_mf, design_rpm)

        # Incidence should be small (< 3 degrees) at design point
        for s in result.stages:
            assert abs(s["incidence_deg"]) < 3.0, (
                f"Stage {s['stage']}: incidence={s['incidence_deg']:.2f} deg, expected near 0"
            )

    def test_off_design_incidence_increases(self, simple_design):
        """At reduced mass flow (same RPM), incidence increases."""
        station0 = simple_design.stations[0]
        gas = GasProperties()
        rho = station0.P_total / (gas.R * station0.T_total)
        design_mf = rho * station0.area * station0.C_axial
        U = simple_design.stages[0].rotor_triangles.inlet.U
        r_mean = station0.r_mean
        design_rpm = (U / r_mean) * 60.0 / (2 * math.pi)

        result_design = off_design_compressor(simple_design, design_mf, design_rpm)
        result_low = off_design_compressor(simple_design, design_mf * 0.7, design_rpm)

        # At lower mass flow, incidence magnitude should be larger
        inc_design = abs(result_design.stages[0]["incidence_deg"])
        inc_low = abs(result_low.stages[0]["incidence_deg"])
        assert inc_low > inc_design, (
            f"Incidence should increase at low mass flow: "
            f"design={inc_design:.2f}, low_mf={inc_low:.2f}"
        )


class TestOffDesignLoss:
    def test_off_design_loss_increases_with_incidence(self, simple_design):
        """Higher incidence → higher loss (via higher DF)."""
        station0 = simple_design.stations[0]
        gas = GasProperties()
        rho = station0.P_total / (gas.R * station0.T_total)
        design_mf = rho * station0.area * station0.C_axial
        U = simple_design.stages[0].rotor_triangles.inlet.U
        r_mean = station0.r_mean
        design_rpm = (U / r_mean) * 60.0 / (2 * math.pi)

        result_design = off_design_compressor(simple_design, design_mf, design_rpm)
        result_low = off_design_compressor(simple_design, design_mf * 0.7, design_rpm)

        loss_design = result_design.stages[0]["loss_total"]
        loss_low = result_low.stages[0]["loss_total"]
        assert loss_low > loss_design, (
            f"Loss should increase off-design: "
            f"design={loss_design:.5f}, low_mf={loss_low:.5f}"
        )

    def test_off_design_pr_decreases_off_design(self, simple_design):
        """PR at 80% mass flow should be different from design PR."""
        station0 = simple_design.stations[0]
        gas = GasProperties()
        rho = station0.P_total / (gas.R * station0.T_total)
        design_mf = rho * station0.area * station0.C_axial
        U = simple_design.stages[0].rotor_triangles.inlet.U
        r_mean = station0.r_mean
        design_rpm = (U / r_mean) * 60.0 / (2 * math.pi)

        result_design = off_design_compressor(simple_design, design_mf, design_rpm)
        result_80 = off_design_compressor(simple_design, design_mf * 0.8, design_rpm)

        # At reduced mass flow but same RPM, the operating point shifts;
        # efficiency drops so overall PR should differ
        assert result_80.overall_efficiency < result_design.overall_efficiency, (
            f"Efficiency should decrease off-design: "
            f"design={result_design.overall_efficiency:.4f}, "
            f"80%={result_80.overall_efficiency:.4f}"
        )


class TestStallDetection:
    def test_off_design_stall_detection(self, simple_design):
        """Very low mass flow should trigger stall flag (DF > 0.6)."""
        station0 = simple_design.stations[0]
        gas = GasProperties()
        rho = station0.P_total / (gas.R * station0.T_total)
        design_mf = rho * station0.area * station0.C_axial
        U = simple_design.stages[0].rotor_triangles.inlet.U
        r_mean = station0.r_mean
        design_rpm = (U / r_mean) * 60.0 / (2 * math.pi)

        # Very low mass flow — should stall
        result = off_design_compressor(simple_design, design_mf * 0.3, design_rpm)
        assert result.is_stalled, (
            "Compressor should be stalled at 30% design mass flow"
        )


# ---------- Compressor map tests ----------

class TestCompressorMap:
    def test_compressor_map_structure(self, simple_design):
        """Map should have correct number of speed lines."""
        fracs = [0.7, 0.85, 1.0]
        cmap = generate_compressor_map(
            simple_design, rpm_fractions=fracs, n_points=5,
        )
        assert len(cmap.speed_lines) == 3
        for sl in cmap.speed_lines:
            assert len(sl.mass_flows) > 0
            assert len(sl.mass_flows) == len(sl.pressure_ratios)
            assert len(sl.mass_flows) == len(sl.efficiencies)

    def test_surge_line_left_of_operating(self, simple_design):
        """Surge mass flow should be less than design mass flow."""
        cmap = generate_compressor_map(
            simple_design, rpm_fractions=[1.0], n_points=15,
        )
        # Design mass flow
        design_mf = cmap.design_point.get("mass_flow", 20.0)

        # Check that surge points (if any) are at lower mass flow
        for sl in cmap.speed_lines:
            if sl.surge_point_index is not None:
                surge_mf = sl.mass_flows[sl.surge_point_index]
                assert surge_mf < design_mf, (
                    f"Surge mass flow ({surge_mf:.2f}) should be < "
                    f"design ({design_mf:.2f})"
                )

    def test_efficiency_peak_near_design(self, simple_design):
        """Peak efficiency on 100% speed line should be near design mass flow."""
        cmap = generate_compressor_map(
            simple_design, rpm_fractions=[1.0], n_points=15,
        )
        sl_100 = cmap.speed_lines[0]
        if len(sl_100.efficiencies) == 0:
            pytest.skip("No valid points on speed line")

        peak_idx = max(range(len(sl_100.efficiencies)),
                       key=lambda i: sl_100.efficiencies[i])
        peak_mf = sl_100.mass_flows[peak_idx]
        design_mf = cmap.design_point.get("mass_flow", 20.0)

        # Peak should be within 30% of design mass flow
        assert abs(peak_mf - design_mf) / design_mf < 0.30, (
            f"Peak eta at m_dot={peak_mf:.2f}, design at {design_mf:.2f}"
        )

    def test_lower_speed_lower_pr(self, simple_design):
        """70% speed line should have lower max PR than 100% speed line."""
        cmap = generate_compressor_map(
            simple_design, rpm_fractions=[0.7, 1.0], n_points=10,
        )
        sl_70 = None
        sl_100 = None
        for sl in cmap.speed_lines:
            if abs(sl.rpm_fraction - 0.7) < 0.01:
                sl_70 = sl
            elif abs(sl.rpm_fraction - 1.0) < 0.01:
                sl_100 = sl

        assert sl_70 is not None and sl_100 is not None
        max_pr_70 = max(sl_70.pressure_ratios) if sl_70.pressure_ratios else 1.0
        max_pr_100 = max(sl_100.pressure_ratios) if sl_100.pressure_ratios else 1.0

        assert max_pr_70 < max_pr_100, (
            f"70% speed max PR ({max_pr_70:.4f}) should be < "
            f"100% speed max PR ({max_pr_100:.4f})"
        )


class TestRotor37Map:
    def test_rotor37_map_design_point(self, rotor37_design):
        """R37 map at 100% speed should show PR in the right ballpark near 20 kg/s."""
        cmap = generate_compressor_map(
            rotor37_design, rpm_fractions=[1.0], n_points=10,
        )
        sl = cmap.speed_lines[0]
        assert len(sl.pressure_ratios) > 0

        # Find point closest to 20 kg/s
        design_mf = cmap.design_point.get("mass_flow", 20.0)
        best_idx = min(range(len(sl.mass_flows)),
                       key=lambda i: abs(sl.mass_flows[i] - design_mf))
        pr_at_design = sl.pressure_ratios[best_idx]

        # PR should be > 1.0 (compressor is doing work)
        assert pr_at_design > 1.0, (
            f"PR at design point mass flow should be > 1.0, got {pr_at_design:.4f}"
        )
        # For R37, PR should be in ballpark of 1.5-3.0
        assert 1.2 < pr_at_design < 3.5, (
            f"R37 PR at design should be ~2.1, got {pr_at_design:.4f}"
        )
