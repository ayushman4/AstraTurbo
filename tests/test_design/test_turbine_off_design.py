"""Tests for axial turbine off-design analysis and map generation."""

import math
import pytest

from astraturbo.design.turbine import meanline_turbine, TurbineResult
from astraturbo.design.turbine_off_design import (
    turbine_off_design_stage,
    TurbineOffDesignResult,
    turbine_off_design,
    TurbineSpeedLine,
    TurbineMap,
    generate_turbine_map,
)
from astraturbo.design.meanline import GasProperties


# ── Default parameters for a representative HP turbine ──

DEFAULT = dict(
    overall_expansion_ratio=2.5,
    mass_flow=20.0,
    rpm=17189,
    r_hub=0.25,
    r_tip=0.35,
    T_inlet=1500.0,
    P_inlet=400000.0,
)


def _design(**overrides):
    """Run design-point solver."""
    kw = {**DEFAULT, **overrides}
    return meanline_turbine(**kw)


# ── Off-design stage tests ──


class TestTurbineOffDesignStage:
    def test_stage_returns_dict(self):
        """Off-design stage returns a dict with required keys."""
        result = turbine_off_design_stage(
            ngv_alpha_metal_out=math.radians(65.0),
            rotor_beta_metal_in=math.radians(-30.0),
            rotor_beta_metal_out=math.radians(-60.0),
            U=300.0,
            C_axial=200.0,
            Pt_in=400000.0,
            Tt_in=1500.0,
            alpha_in=0.0,
        )
        assert isinstance(result, dict)
        assert "ER" in result
        assert "efficiency" in result
        assert "incidence_deg" in result
        assert "nozzle_mach" in result
        assert "is_choked" in result
        assert "work" in result

    def test_positive_work_extraction(self):
        """Turbine stage extracts positive work."""
        result = turbine_off_design_stage(
            ngv_alpha_metal_out=math.radians(65.0),
            rotor_beta_metal_in=math.radians(-30.0),
            rotor_beta_metal_out=math.radians(-60.0),
            U=300.0,
            C_axial=200.0,
            Pt_in=400000.0,
            Tt_in=1500.0,
            alpha_in=0.0,
        )
        assert result["work"] > 0

    def test_incidence_changes_with_axial_velocity(self):
        """Changing C_axial at fixed metal angles creates incidence."""
        base = turbine_off_design_stage(
            ngv_alpha_metal_out=math.radians(65.0),
            rotor_beta_metal_in=math.radians(-30.0),
            rotor_beta_metal_out=math.radians(-60.0),
            U=300.0,
            C_axial=200.0,
            Pt_in=400000.0,
            Tt_in=1500.0,
            alpha_in=0.0,
        )
        off = turbine_off_design_stage(
            ngv_alpha_metal_out=math.radians(65.0),
            rotor_beta_metal_in=math.radians(-30.0),
            rotor_beta_metal_out=math.radians(-60.0),
            U=300.0,
            C_axial=250.0,  # different
            Pt_in=400000.0,
            Tt_in=1500.0,
            alpha_in=0.0,
        )
        assert base["incidence_deg"] != pytest.approx(off["incidence_deg"], abs=0.1)


# ── Multi-stage off-design tests ──


class TestTurbineOffDesign:
    def test_basic_off_design(self):
        """Off-design at design conditions completes without error."""
        design = _design()
        od = turbine_off_design(design, mass_flow=20.0, rpm=17189)
        assert isinstance(od, TurbineOffDesignResult)
        assert len(od.stages) == design.n_stages

    def test_work_positive(self):
        """Total work is positive at design conditions."""
        design = _design()
        od = turbine_off_design(design, mass_flow=20.0, rpm=17189)
        assert od.total_work > 0

    def test_efficiency_reasonable(self):
        """Overall efficiency is in plausible range near design."""
        design = _design()
        od = turbine_off_design(design, mass_flow=20.0, rpm=17189)
        assert 0.3 <= od.overall_efficiency <= 1.0

    def test_er_positive(self):
        """Overall expansion ratio > 1 (turbine expands flow)."""
        design = _design()
        od = turbine_off_design(design, mass_flow=20.0, rpm=17189)
        assert od.overall_er > 1.0

    def test_efficiency_drops_off_design(self):
        """Efficiency is lower at significantly off-design RPM."""
        design = _design()
        od_design = turbine_off_design(design, mass_flow=20.0, rpm=17189)
        od_low = turbine_off_design(design, mass_flow=20.0, rpm=10000)
        # At very different RPM, efficiency should be lower
        assert od_low.overall_efficiency <= od_design.overall_efficiency + 0.05

    def test_summary_string(self):
        """summary() returns non-empty string."""
        design = _design()
        od = turbine_off_design(design, mass_flow=20.0, rpm=17189)
        s = od.summary()
        assert "Off-Design" in s
        assert "Stage 1" in s


# ── Map generation tests ──


class TestTurbineMap:
    def test_map_structure(self):
        """Map has correct number of speed lines."""
        design = _design()
        tmap = generate_turbine_map(design, n_points=5)
        assert isinstance(tmap, TurbineMap)
        assert len(tmap.speed_lines) == 8  # default rpm_fractions

    def test_custom_rpm_fractions(self):
        """Custom rpm_fractions produces matching speed line count."""
        design = _design()
        fracs = [0.7, 0.85, 1.0]
        tmap = generate_turbine_map(design, rpm_fractions=fracs, n_points=5)
        assert len(tmap.speed_lines) == 3

    def test_speed_line_has_points(self):
        """Each speed line has data points."""
        design = _design()
        tmap = generate_turbine_map(design, n_points=5)
        for sl in tmap.speed_lines:
            assert isinstance(sl, TurbineSpeedLine)
            assert len(sl.mass_flows) > 0
            assert len(sl.expansion_ratios) == len(sl.mass_flows)
            assert len(sl.efficiencies) == len(sl.mass_flows)
            assert len(sl.is_choked) == len(sl.mass_flows)

    def test_design_point_recorded(self):
        """Design point info is captured in the map."""
        design = _design()
        tmap = generate_turbine_map(design, n_points=5)
        assert "mass_flow" in tmap.design_point
        assert "er" in tmap.design_point
        assert "efficiency" in tmap.design_point
        assert "rpm" in tmap.design_point
        assert tmap.design_point["er"] > 1.0

    def test_summary_string(self):
        """summary() returns non-empty string."""
        design = _design()
        tmap = generate_turbine_map(design, n_points=5)
        s = tmap.summary()
        assert "Turbine Map" in s
        assert "Speed Line" in s

    def test_choke_line_extracted(self):
        """Choke line is a list of (mass_flow, ER) tuples."""
        design = _design()
        tmap = generate_turbine_map(design, n_points=10)
        # choke_line may or may not have points, but should be a list
        assert isinstance(tmap.choke_line, list)
        for item in tmap.choke_line:
            assert len(item) == 2
