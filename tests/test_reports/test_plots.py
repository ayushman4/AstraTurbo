"""Tests for report plot generation and image embedding."""

from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs mirroring the result dataclasses used by the plot funcs
# ---------------------------------------------------------------------------

@dataclass
class _Station:
    P_total: float
    T_total: float


@dataclass
class _CombustorResult:
    fuel_air_ratio: float = 0.02


@dataclass
class _AfterburnerResult:
    T_out: float = 1800.0
    fuel_flow: float = 0.5
    eta_afterburner: float = 0.92
    dp_fraction: float = 0.04


@dataclass
class _EngineCycleResult:
    engine_type: str = "turbojet"
    stations: dict = field(default_factory=lambda: {
        "inlet": _Station(101325, 288),
        "compressor_exit": _Station(1013250, 600),
        "combustor_exit": _Station(960000, 1500),
        "turbine_exit": _Station(200000, 900),
        "nozzle_exit": _Station(101325, 700),
        "afterburner_exit": _Station(190000, 1800),
    })
    net_thrust: float = 50000.0
    specific_fuel_consumption: float = 2.5e-5
    thermal_efficiency: float = 0.35
    propulsive_efficiency: float = 0.7
    overall_efficiency: float = 0.245
    shaft_power: float = 0.0
    mass_flow: float = 50.0
    fuel_flow: float = 1.0
    compressor_work: float = 300000.0
    turbine_work: float = 320000.0
    mechanical_efficiency: float = 0.99
    combustor: _CombustorResult = field(default_factory=_CombustorResult)
    afterburner: _AfterburnerResult = field(default_factory=_AfterburnerResult)
    n_spools: int = 1
    spools: list = field(default_factory=list)


@dataclass
class _CoolingRowResult:
    row_index: int
    cooling_effectiveness: float
    coolant_fraction: float
    coolant_mass_flow: float
    T_blade: float


@dataclass
class _CoolingResult:
    cooling_type: str = "film"
    phi: float = 0.4
    T_gas: float = 1600.0
    T_coolant: float = 700.0
    total_coolant_fraction: float = 0.08
    total_coolant_flow: float = 0.4
    overall_effectiveness: float = 0.6
    n_cooled_rows: int = 2
    rows: list = field(default_factory=lambda: [
        _CoolingRowResult(0, 0.55, 0.04, 0.2, 1100),
        _CoolingRowResult(1, 0.50, 0.04, 0.2, 1120),
    ])


@dataclass
class _TurbopumpResult:
    pump_power: float = 50000.0
    turbine_power: float = 55000.0
    shaft_power: float = 52000.0
    power_margin: float = 0.06
    shaft_rpm: float = 30000.0
    cycle_type: str = "gas_generator"
    mechanical_efficiency: float = 0.95
    overall_efficiency: float = 0.65
    pump: None = None
    turbine: None = None


@dataclass
class _PropellerResult:
    thrust: float = 500.0
    power: float = 25000.0
    efficiency: float = 0.82
    figure_of_merit: float = 0.75
    diameter: float = 1.2
    rpm: float = 3000.0
    n_blades: int = 3
    advance_ratio: float = 0.8
    CT: float = 0.012
    CP: float = 0.015
    blade_angle_75: float = 25.0
    solidity: float = 0.08
    disk_loading: float = 442.0
    tip_speed: float = 188.5
    tip_mach: float = 0.55


@dataclass
class _ElectricMotorResult:
    shaft_power: float = 10000.0
    torque: float = 3.2
    rpm: float = 30000.0
    voltage: float = 48.0
    current: float = 220.0
    efficiency: float = 0.95
    weight_kg: float = 2.5
    power_density: float = 4.0
    motor_constant_kv: float = 625.0
    thermal_margin: float = 0.3
    motor_type: str = "BLDC"


@dataclass
class _StructuredBlock:
    name: str = "block0"
    points: np.ndarray = field(default_factory=lambda: np.zeros((5, 5, 2)))
    n_cells_i: int = 4
    n_cells_j: int = 4
    grading_i: float = 1.0
    grading_j: float = 1.0
    patches: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.points.sum() == 0:
            x = np.linspace(0, 1, 5)
            y = np.linspace(0, 0.5, 5)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            self.points = np.stack([xx, yy], axis=-1)


@dataclass
class _MultiBlockMesh:
    blocks: list = field(default_factory=lambda: [_StructuredBlock()])
    name: str = "TestMesh"

    @property
    def n_blocks(self):
        return len(self.blocks)

    @property
    def total_cells(self):
        return sum(b.n_cells_i * b.n_cells_j for b in self.blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_base64_png(s: str) -> bool:
    """Check that s is a valid base64 string that decodes to a PNG."""
    raw = base64.b64decode(s)
    return raw[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Plot function tests
# ---------------------------------------------------------------------------

class TestPlotFunctions:
    def test_plot_engine_stations(self):
        from astraturbo.reports.plots import plot_engine_stations
        result = plot_engine_stations(_EngineCycleResult())
        assert result != ""
        assert _is_valid_base64_png(result)

    def test_plot_blade_profile(self):
        from astraturbo.reports.plots import plot_blade_profile
        theta = np.linspace(0, 2 * np.pi, 60)
        coords = np.column_stack([np.cos(theta), 0.3 * np.sin(theta)])
        result = plot_blade_profile(coords)
        assert result != ""
        assert _is_valid_base64_png(result)

    def test_plot_blade_profile_bad_input(self):
        from astraturbo.reports.plots import plot_blade_profile
        assert plot_blade_profile(np.array([1, 2, 3])) == ""

    def test_plot_cooling_rows(self):
        from astraturbo.reports.plots import plot_cooling_rows
        result = plot_cooling_rows(_CoolingResult())
        assert result != ""
        assert _is_valid_base64_png(result)

    def test_plot_turbopump_power(self):
        from astraturbo.reports.plots import plot_turbopump_power
        result = plot_turbopump_power(_TurbopumpResult())
        assert result != ""
        assert _is_valid_base64_png(result)

    def test_plot_propeller_summary(self):
        from astraturbo.reports.plots import plot_propeller_summary
        result = plot_propeller_summary(_PropellerResult())
        assert result != ""
        assert _is_valid_base64_png(result)

    def test_plot_motor_summary(self):
        from astraturbo.reports.plots import plot_motor_summary
        result = plot_motor_summary(_ElectricMotorResult())
        assert result != ""
        assert _is_valid_base64_png(result)

    def test_plot_mesh_2d(self):
        from astraturbo.reports.plots import plot_mesh_2d
        result = plot_mesh_2d(_MultiBlockMesh())
        assert result != ""
        assert _is_valid_base64_png(result)


# ---------------------------------------------------------------------------
# HTML report integration
# ---------------------------------------------------------------------------

class TestReportImages:
    def test_report_contains_img_tags(self, tmp_path):
        from astraturbo.reports import generate_report, ReportConfig

        out = tmp_path / "report.html"
        config = ReportConfig(output_path=str(out))

        generate_report(
            config=config,
            engine_cycle_result=_EngineCycleResult(),
            electric_motor_result=_ElectricMotorResult(),
            propeller_result=_PropellerResult(),
            turbopump_result=_TurbopumpResult(),
            cooling_result=_CoolingResult(),
            profile_coords=np.column_stack([
                np.cos(np.linspace(0, 2 * np.pi, 40)),
                0.3 * np.sin(np.linspace(0, 2 * np.pi, 40)),
            ]),
            mesh=_MultiBlockMesh(),
        )

        html = out.read_text()
        assert "<img" in html
        assert "data:image/png;base64," in html
        # At least one image per section that has a plot
        assert html.count("<img") >= 5
