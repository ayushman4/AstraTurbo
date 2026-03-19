"""Integration tests for plot image embedding in AI, GUI, and CLI paths."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest


# ── Lightweight stubs for engine cycle result ──────────────────────

@dataclass
class _Station:
    P_total: float
    T_total: float


@dataclass
class _CombustorResult:
    fuel_air_ratio: float = 0.02


@dataclass
class _EngineCycleResult:
    engine_type: str = "turbojet"
    stations: dict = field(default_factory=lambda: {
        "inlet": _Station(101325, 288),
        "compressor_exit": _Station(800000, 580),
        "combustor_exit": _Station(760000, 1500),
        "turbine_exit": _Station(200000, 900),
        "nozzle_exit": _Station(101325, 700),
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
    afterburner: None = None
    n_spools: int = 1
    spools: list = field(default_factory=list)


# ── Tests ──────────────────────────────────────────────────────────


class TestCLIReportImages:
    """CLI engine-cycle --report produces HTML with embedded images."""

    def test_engine_cycle_report_has_images(self, tmp_path):
        from astraturbo.design.engine_cycle import engine_cycle
        from astraturbo.reports import generate_report, ReportConfig

        result = engine_cycle(
            overall_pressure_ratio=8.0,
            turbine_inlet_temp=1400.0,
            mass_flow=20.0,
            rpm=15000,
            r_hub=0.15,
            r_tip=0.30,
        )

        out = tmp_path / "report.html"
        cfg = ReportConfig(
            title="CLI Engine Cycle Test",
            output_path=str(out),
        )
        generate_report(config=cfg, engine_cycle_result=result)

        html = out.read_text()
        assert "<img" in html
        assert "data:image/png;base64," in html

    def test_compressor_report_with_profile_and_mesh(self, tmp_path):
        """Compressor report with profile_coords and mesh includes all images."""
        from astraturbo.design import meanline_compressor
        from astraturbo.camberline import NACA65
        from astraturbo.thickness import NACA65Series
        from astraturbo.profile import Superposition
        from astraturbo.mesh.multiblock import generate_blade_passage_mesh
        from astraturbo.reports import generate_report, ReportConfig

        comp = meanline_compressor(
            overall_pressure_ratio=1.5,
            mass_flow=20.0,
            rpm=15000,
            r_hub=0.15,
            r_tip=0.25,
        )
        prof = Superposition(NACA65(cl0=1.0), NACA65Series())
        coords = prof.as_array()
        mesh = generate_blade_passage_mesh(
            profile=coords, pitch=0.05,
            n_blade=20, n_ogrid=5, n_inlet=8, n_outlet=8, n_passage=10,
        )

        out = tmp_path / "report.html"
        generate_report(
            config=ReportConfig(output_path=str(out)),
            meanline_result=comp,
            profile_coords=coords,
            mesh=mesh,
        )

        html = out.read_text()
        # Should have blade profile + mesh images
        assert html.count("<img") >= 2
        assert "Blade Profile" in html
        assert "Computational Mesh" in html


class TestAIToolReportImages:
    """AI tool generate_report produces HTML with images when engine cycle params given."""

    def test_ai_generate_report_with_engine_cycle(self, tmp_path):
        from astraturbo.ai.tools import execute_tool

        out = tmp_path / "ai_report.html"
        result = execute_tool("generate_report", {
            "output_path": str(out),
            "title": "AI Report Test",
            "overall_pressure_ratio": 8.0,
            "turbine_inlet_temp": 1400.0,
            "mass_flow": 20.0,
            "rpm": 15000,
            "r_hub": 0.15,
            "r_tip": 0.30,
        })

        assert "Report generated" in result
        html = out.read_text()
        # Should have engine station plot and blade profile
        assert "<img" in html
        assert "data:image/png;base64," in html

    def test_ai_engine_cycle_with_report(self, tmp_path):
        from astraturbo.ai.tools import execute_tool

        out = tmp_path / "ec_report.html"
        result = execute_tool("engine_cycle", {
            "overall_pressure_ratio": 8.0,
            "turbine_inlet_temp": 1400.0,
            "mass_flow": 20.0,
            "rpm": 15000,
            "r_hub": 0.15,
            "r_tip": 0.30,
            "report_path": str(out),
        })

        assert "Report generated" in result
        html = out.read_text()
        assert "<img" in html


class TestPipelineEndToEnd:
    """End-to-end pipeline test: engine → compressor → profile → mesh → CFD → report."""

    def test_single_engine_pipeline(self, tmp_path):
        from astraturbo.design.engine_cycle import engine_cycle
        from astraturbo.design import meanline_compressor, meanline_to_blade_parameters
        from astraturbo.camberline import NACA65
        from astraturbo.thickness import NACA65Series
        from astraturbo.profile import Superposition
        from astraturbo.mesh.multiblock import generate_blade_passage_mesh
        from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig
        from astraturbo.reports import generate_report, ReportConfig

        # 1. Engine cycle
        ec = engine_cycle(
            overall_pressure_ratio=8.0,
            turbine_inlet_temp=1400.0,
            mass_flow=20.0,
            rpm=15000,
            r_hub=0.15, r_tip=0.30,
        )
        assert ec.net_thrust > 0

        # 2. Compressor
        comp = meanline_compressor(
            overall_pressure_ratio=8.0,
            mass_flow=20.0, rpm=15000,
            r_hub=0.15, r_tip=0.30,
        )
        assert comp.n_stages > 0
        bp = meanline_to_blade_parameters(comp)

        # 3. Profile
        prof = Superposition(NACA65(cl0=1.0), NACA65Series())
        coords = prof.as_array()
        assert coords.shape[1] == 2

        # 4. Mesh
        mesh = generate_blade_passage_mesh(
            profile=coords, pitch=0.05,
            n_blade=20, n_ogrid=5, n_inlet=8, n_outlet=8, n_passage=10,
        )
        assert mesh.total_cells > 0

        # 5. CFD
        cfd_dir = tmp_path / "cfd_case"
        wf = CFDWorkflow(CFDWorkflowConfig(solver="openfoam", inlet_velocity=100.0))
        case = wf.setup_case(str(cfd_dir))
        assert (cfd_dir / "Allrun").exists() or Path(case).exists()

        # 6. Report with images
        out = tmp_path / "pipeline_report.html"
        generate_report(
            config=ReportConfig(
                title="Pipeline Test",
                output_path=str(out),
            ),
            engine_cycle_result=ec,
            meanline_result=comp,
            blade_params=bp,
            profile_coords=coords,
            mesh=mesh,
        )

        html = out.read_text()
        assert html.count("<img") >= 3  # station chart, blade profile, mesh
        assert "Engine Cycle Analysis" in html
        assert "Blade Profile" in html
        assert "Computational Mesh" in html


class TestCLIMeanlineProfile:
    """CLI meanline report now includes blade profile from blade_params."""

    def test_meanline_report_has_profile_image(self, tmp_path):
        from astraturbo.design import meanline_compressor, meanline_to_blade_parameters
        from astraturbo.reports import generate_report, ReportConfig

        comp = meanline_compressor(
            overall_pressure_ratio=2.0,
            mass_flow=20.0,
            rpm=15000,
            r_hub=0.15,
            r_tip=0.30,
        )
        params = meanline_to_blade_parameters(comp)

        # Simulate what _cmd_meanline now does
        from astraturbo.camberline import NACA65
        from astraturbo.thickness import NACA65Series
        from astraturbo.profile import Superposition
        cl0 = params[0]["rotor_camber_deg"] / 25.0
        prof = Superposition(NACA65(cl0=max(0.2, min(cl0, 2.0))), NACA65Series())
        profile_coords = prof.as_array()

        out = tmp_path / "meanline_profile.html"
        generate_report(
            config=ReportConfig(output_path=str(out)),
            meanline_result=comp,
            blade_params=params,
            profile_coords=profile_coords,
        )

        html = out.read_text()
        assert "Blade Profile" in html
        assert "Blade Surface Loading" in html
        assert html.count("<img") >= 3


class TestAIToolCFDReport:
    """AI generate_report tool with compressor params generates mesh image."""

    def test_ai_report_with_mesh(self, tmp_path):
        from astraturbo.ai.tools import execute_tool

        out = tmp_path / "ai_mesh_report.html"
        result = execute_tool("generate_report", {
            "output_path": str(out),
            "overall_pressure_ratio": 2.0,
            "mass_flow": 20.0,
            "rpm": 15000,
            "r_hub": 0.15,
            "r_tip": 0.30,
        })

        assert "Report generated" in result
        html = out.read_text()
        assert "Computational Mesh" in html
        assert "Blade Profile" in html
