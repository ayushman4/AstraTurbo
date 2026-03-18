"""Tests for axial turbine meanline solver."""

import math
import pytest

from astraturbo.design.turbine import (
    TurbineStageResult,
    TurbineResult,
    soderberg_loss,
    zweifel_loading,
    meanline_turbine_stage,
    meanline_turbine,
    meanline_to_turbine_blade_parameters,
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


def _run(**overrides):
    kw = {**DEFAULT, **overrides}
    return meanline_turbine(**kw)


# ── Tests ──


def test_basic_design():
    """Solver completes without error."""
    result = _run()
    assert isinstance(result, TurbineResult)
    assert result.n_stages >= 1
    assert len(result.stages) == result.n_stages
    assert len(result.stations) == result.n_stages + 1


def test_expansion_ratio_close_to_target():
    """Achieved ER is within 10 % of target."""
    result = _run()
    assert result.overall_expansion_ratio == pytest.approx(
        DEFAULT["overall_expansion_ratio"], rel=0.10
    )


def test_work_output_positive():
    """Turbine extracts work (positive)."""
    result = _run()
    assert result.total_work > 0
    for stage in result.stages:
        assert stage.work_output > 0


def test_temperature_drops():
    """Temperature decreases through the turbine."""
    result = _run()
    assert result.overall_temperature_ratio < 1.0
    for i in range(len(result.stations) - 1):
        assert result.stations[i + 1].T_total < result.stations[i].T_total


def test_pressure_drops():
    """Pressure decreases through the turbine."""
    result = _run()
    for i in range(len(result.stations) - 1):
        assert result.stations[i + 1].P_total < result.stations[i].P_total


def test_efficiency_reasonable():
    """Stage isentropic efficiencies are in 0.80-0.96."""
    result = _run()
    for stage in result.stages:
        assert 0.80 <= stage.isentropic_efficiency <= 0.96


def test_loading_typical():
    """Loading coefficient psi in 0.8-3.0 (turbine range)."""
    result = _run()
    for stage in result.stages:
        assert 0.8 <= stage.loading_coefficient <= 3.0


def test_zweifel_in_range():
    """Zweifel coefficient in 0.2-1.5 (depends on flow coefficient)."""
    result = _run()
    for stage in result.stages:
        assert 0.2 <= stage.zweifel_coefficient <= 1.5


def test_nozzle_mach_subsonic():
    """Nozzle exit Mach < 1.0 at moderate ER."""
    result = _run(overall_expansion_ratio=2.0)
    for stage in result.stages:
        assert stage.nozzle_exit_mach < 1.0


def test_multi_stage():
    """2-stage turbine works correctly."""
    result = _run(n_stages=2)
    assert result.n_stages == 2
    assert len(result.stages) == 2
    # Total work should be split
    total = sum(s.work_output for s in result.stages)
    assert total == pytest.approx(result.total_work, rel=0.05)


def test_kaveri_hp_turbine():
    """Kaveri-class HP turbine: ER=2.5, T_in=1500 K, 20 kg/s."""
    result = _run()
    # Should produce significant work
    assert result.total_work > 100_000  # > 100 kJ/kg
    assert result.n_stages >= 1
    assert result.overall_efficiency > 0.85


def test_lp_turbine():
    """LP turbine: ER=3, T_in=1000 K."""
    result = _run(overall_expansion_ratio=3.0, T_inlet=1000.0, P_inlet=200000.0)
    assert result.total_work > 0
    assert result.overall_temperature_ratio < 1.0
    assert result.n_stages >= 1


def test_power_matches_work():
    """Power = work * mass_flow."""
    result = _run()
    mass_flow = DEFAULT["mass_flow"]
    expected_power = result.total_work * mass_flow  # Watts
    assert expected_power > 0
    # Verify consistency: sum of stage work ≈ total work
    stage_sum = sum(s.work_output for s in result.stages)
    assert stage_sum == pytest.approx(result.total_work, rel=0.05)


def test_radial_angles_exist():
    """Free-vortex radial distribution is computed."""
    result = _run(radial_stations=5)
    for stage in result.stages:
        assert len(stage.radial_blade_angles) == 5
        for entry in stage.radial_blade_angles:
            assert "r" in entry
            assert "alpha_ngv_out" in entry
            assert "beta_rotor_in" in entry
            assert "beta_rotor_out" in entry


# ── Supplementary unit tests ──


def test_soderberg_loss_positive():
    """Soderberg loss is always positive."""
    assert soderberg_loss(60.0) > 0
    assert soderberg_loss(90.0) > soderberg_loss(30.0)


def test_zweifel_loading_typical():
    """Zweifel gives reasonable values for typical turbine angles."""
    Z = zweifel_loading(40.0, 60.0, 0.8, 1.0)
    assert 0.3 < Z < 2.0


def test_blade_parameters():
    """meanline_to_turbine_blade_parameters returns correct structure."""
    result = _run()
    params = meanline_to_turbine_blade_parameters(result)
    assert len(params) == result.n_stages
    for p in params:
        assert "ngv_stagger_deg" in p
        assert "rotor_stagger_deg" in p
        assert "zweifel" in p
        assert "stage" in p


def test_summary_string():
    """summary() returns a non-empty string."""
    result = _run()
    s = result.summary()
    assert "Turbine" in s
    assert "ER" in s
    assert "Stage 1" in s
