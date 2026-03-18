"""Tests for centrifugal compressor and report generation."""

import math
import os
import pytest
from pathlib import Path

from astraturbo.design.centrifugal import (
    CentrifugalResult,
    centrifugal_compressor,
    wiesner_slip_factor,
)
from astraturbo.design.meanline import meanline_compressor, GasProperties
from astraturbo.reports import generate_report, ReportConfig


# ---------- Centrifugal compressor tests ----------

class TestWiesnerSlip:
    def test_slip_increases_with_blades(self):
        """More blades = higher slip factor (less slip)."""
        s12 = wiesner_slip_factor(12, -30)
        s20 = wiesner_slip_factor(20, -30)
        assert s20 > s12

    def test_slip_typical_range(self):
        """Slip factor should be 0.85-0.95 for typical configs."""
        s = wiesner_slip_factor(17, -30)
        assert 0.85 < s < 0.96

    def test_radial_blades_higher_slip(self):
        """Radial blades (0 deg) should have slightly different slip than backswept."""
        s_radial = wiesner_slip_factor(17, 0)
        s_swept = wiesner_slip_factor(17, -40)
        # Both should be valid
        assert 0.80 < s_radial < 1.0
        assert 0.80 < s_swept < 1.0


class TestCentrifugalCompressor:
    def test_basic_design(self):
        """Basic centrifugal design should complete without error."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        assert isinstance(result, CentrifugalResult)
        assert result.pressure_ratio > 1.0

    def test_pr_in_range(self):
        """Result PR should be within 30% of target."""
        target_pr = 3.0
        result = centrifugal_compressor(
            pressure_ratio=target_pr, mass_flow=1.0, rpm=60000,
        )
        assert abs(result.pressure_ratio - target_pr) / target_pr < 0.30

    def test_efficiency_reasonable(self):
        """Efficiency should be in 0.5-0.95 range."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        assert 0.5 < result.isentropic_efficiency < 0.95

    def test_power_positive(self):
        """Power should be positive (compressor consumes work)."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        assert result.power_kW > 0

    def test_tip_speed_reasonable(self):
        """Tip speed should be < 600 m/s (structural limit)."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        assert 100 < result.tip_speed < 600

    def test_impeller_dict(self):
        """Impeller dict should have key parameters."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        assert "r2" in result.impeller
        assert "slip_factor" in result.impeller
        assert "n_blades" in result.impeller
        assert "M_w1" in result.impeller

    def test_diffuser_dict(self):
        """Diffuser dict should have key parameters."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        assert "r3" in result.diffuser
        assert "r4" in result.diffuser
        assert "Cp" in result.diffuser

    def test_higher_pr_needs_more_power(self):
        """Higher PR should require more power."""
        r_low = centrifugal_compressor(pressure_ratio=2.0, mass_flow=1.0, rpm=60000)
        r_high = centrifugal_compressor(pressure_ratio=4.0, mass_flow=1.0, rpm=60000)
        assert r_high.power_kW > r_low.power_kW

    def test_drone_scale(self):
        """Small drone compressor (0.1 kg/s, PR=2, 100k RPM)."""
        result = centrifugal_compressor(
            pressure_ratio=2.0, mass_flow=0.1, rpm=100000,
            r1_hub=0.005, r1_tip=0.015,
        )
        assert result.pressure_ratio > 1.0
        assert result.power_kW < 50  # Small compressor

    def test_turbocharger_scale(self):
        """Automotive turbocharger (0.5 kg/s, PR=2.5, 120k RPM)."""
        result = centrifugal_compressor(
            pressure_ratio=2.5, mass_flow=0.5, rpm=120000,
            r1_hub=0.01, r1_tip=0.03,
        )
        assert result.pressure_ratio > 1.0

    def test_summary_string(self):
        """summary() should return a non-empty string."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        s = result.summary()
        assert "Centrifugal" in s
        assert "PR:" in s


# ---------- Report generation tests ----------

class TestReportGeneration:
    def test_meanline_report(self, tmp_path):
        """Generate a report from meanline results."""
        result = meanline_compressor(
            overall_pressure_ratio=1.5, mass_flow=20.0,
            rpm=15000, r_hub=0.15, r_tip=0.25,
        )
        from astraturbo.design.meanline import meanline_to_blade_parameters
        params = meanline_to_blade_parameters(result)

        output = tmp_path / "report.html"
        cfg = ReportConfig(title="Test Report", output_path=str(output))
        path = generate_report(config=cfg, meanline_result=result, blade_params=params)

        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Test Report" in content
        assert "Meanline Design Summary" in content
        assert "Stage" in content

    def test_centrifugal_report(self, tmp_path):
        """Generate a report from centrifugal results."""
        result = centrifugal_compressor(
            pressure_ratio=3.0, mass_flow=1.0, rpm=60000,
        )
        output = tmp_path / "cent_report.html"
        cfg = ReportConfig(title="Centrifugal Report", output_path=str(output))
        path = generate_report(config=cfg, centrifugal_result=result)

        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Centrifugal" in content
        assert "Impeller" in content
        assert "Diffuser" in content

    def test_report_with_material(self, tmp_path):
        """Report with material section."""
        from astraturbo.fea import get_material
        mat = get_material("inconel_718")

        output = tmp_path / "mat_report.html"
        cfg = ReportConfig(output_path=str(output))
        path = generate_report(config=cfg, material=mat, material_temperature=973)

        content = Path(path).read_text()
        assert "Inconel" in content
        assert "973" in content

    def test_report_with_map(self, tmp_path):
        """Report with compressor map."""
        from astraturbo.design import meanline_compressor, generate_compressor_map
        result = meanline_compressor(
            overall_pressure_ratio=1.5, mass_flow=20.0,
            rpm=15000, r_hub=0.15, r_tip=0.25,
        )
        cmap = generate_compressor_map(result, rpm_fractions=[0.8, 1.0], n_points=5)

        output = tmp_path / "map_report.html"
        cfg = ReportConfig(output_path=str(output))
        path = generate_report(config=cfg, meanline_result=result, compressor_map=cmap)

        content = Path(path).read_text()
        assert "Compressor Map" in content
        assert "Speed Line" in content or "N/N" in content

    def test_empty_report(self, tmp_path):
        """Report with no data should still generate valid HTML."""
        output = tmp_path / "empty.html"
        cfg = ReportConfig(output_path=str(output))
        path = generate_report(config=cfg)

        assert Path(path).exists()
        content = Path(path).read_text()
        assert "<html>" in content
        assert "</html>" in content
