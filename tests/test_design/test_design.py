"""Tests for velocity triangle and meanline design modules."""

import math
import pytest
import numpy as np

from astraturbo.design import (
    VelocityTriangle,
    BladeRowTriangles,
    compute_triangle_from_angles,
    compute_triangle_from_beta,
    GasProperties,
    meanline_compressor_stage,
    meanline_compressor,
    meanline_to_blade_parameters,
)


class TestVelocityTriangle:
    def test_basic_triangle(self):
        vt = VelocityTriangle(U=300, C_axial=150, C_theta=100)
        assert vt.C == pytest.approx(math.sqrt(150**2 + 100**2), abs=0.1)
        assert vt.W_theta == pytest.approx(100 - 300, abs=0.1)

    def test_zero_swirl(self):
        vt = VelocityTriangle(U=200, C_axial=150, C_theta=0)
        assert vt.alpha_deg == pytest.approx(0, abs=0.01)
        assert vt.W_theta == pytest.approx(-200, abs=0.1)

    def test_from_angles(self):
        vt = compute_triangle_from_angles(U=300, C_axial=150, alpha=math.radians(30))
        assert vt.alpha_deg == pytest.approx(30, abs=0.1)
        assert vt.C_theta == pytest.approx(150 * math.tan(math.radians(30)), abs=0.1)

    def test_from_beta(self):
        vt = compute_triangle_from_beta(U=300, C_axial=150, beta=math.radians(-45))
        assert vt.beta_deg == pytest.approx(-45, abs=0.1)

    def test_blade_row_triangles(self):
        inlet = VelocityTriangle(U=300, C_axial=150, C_theta=50)
        outlet = VelocityTriangle(U=300, C_axial=150, C_theta=200)
        row = BladeRowTriangles(inlet=inlet, outlet=outlet)

        assert row.delta_C_theta == pytest.approx(150, abs=0.1)
        assert row.work_per_unit_mass == pytest.approx(300 * 150, abs=10)
        assert row.de_haller_ratio > 0

    def test_summary(self):
        inlet = VelocityTriangle(U=300, C_axial=150, C_theta=50)
        outlet = VelocityTriangle(U=300, C_axial=150, C_theta=200)
        row = BladeRowTriangles(inlet=inlet, outlet=outlet)
        text = row.summary()
        assert "Inlet" in text
        assert "Outlet" in text
        assert "De Haller" in text


class TestGasProperties:
    def test_air_defaults(self):
        gas = GasProperties()
        assert gas.gamma == pytest.approx(1.4)
        assert gas.cp == pytest.approx(1005)

    def test_speed_of_sound(self):
        gas = GasProperties()
        a = gas.speed_of_sound(288.15)
        assert 330 < a < 350  # ~340 m/s at 288 K

    def test_mach_number(self):
        gas = GasProperties()
        M = gas.mach_number(170.0, 288.15)
        assert 0.4 < M < 0.6


class TestMeanlineStage:
    def test_single_stage(self):
        stage = meanline_compressor_stage(
            U=300, C_axial=150, alpha_in=0,
            stage_pressure_ratio=1.3, eta_stage=0.88,
        )
        assert stage.pressure_ratio == pytest.approx(1.3)
        assert stage.isentropic_efficiency == pytest.approx(0.88)
        assert stage.flow_coefficient > 0
        assert stage.loading_coefficient > 0
        assert 0 < stage.degree_of_reaction < 1

    def test_blade_angles_returned(self):
        stage = meanline_compressor_stage(
            U=300, C_axial=150, alpha_in=0,
            stage_pressure_ratio=1.3,
        )
        angles = stage.blade_angles_deg()
        assert "rotor_inlet_beta" in angles
        assert "rotor_outlet_beta" in angles
        assert "stator_inlet_alpha" in angles
        assert "stator_outlet_alpha" in angles

    def test_de_haller_reasonable(self):
        stage = meanline_compressor_stage(
            U=300, C_axial=150, alpha_in=0,
            stage_pressure_ratio=1.2,
        )
        dh = stage.rotor_triangles.de_haller_ratio
        assert 0.5 < dh < 1.0  # Reasonable range


class TestMeanlineCompressor:
    def test_multi_stage(self):
        result = meanline_compressor(
            overall_pressure_ratio=4.0,
            mass_flow=20.0,
            rpm=12000,
            r_hub=0.15,
            r_tip=0.30,
        )
        assert result.n_stages >= 2
        assert result.overall_pressure_ratio == pytest.approx(4.0, rel=0.05)
        assert len(result.stages) == result.n_stages
        assert len(result.stations) == result.n_stages + 1

    def test_specified_stages(self):
        result = meanline_compressor(
            overall_pressure_ratio=3.0,
            mass_flow=15.0,
            rpm=10000,
            r_hub=0.12,
            r_tip=0.25,
            n_stages=5,
        )
        assert result.n_stages == 5
        assert len(result.stages) == 5

    def test_summary(self):
        result = meanline_compressor(
            overall_pressure_ratio=2.0,
            mass_flow=10.0,
            rpm=15000,
            r_hub=0.10,
            r_tip=0.20,
        )
        text = result.summary()
        assert "Stage" in text
        assert "PR" in text

    def test_blade_parameters(self):
        result = meanline_compressor(
            overall_pressure_ratio=3.0,
            mass_flow=20.0,
            rpm=12000,
            r_hub=0.15,
            r_tip=0.30,
        )
        params = meanline_to_blade_parameters(result)
        assert len(params) == result.n_stages
        for p in params:
            assert "rotor_stagger_deg" in p
            assert "stator_stagger_deg" in p
            assert "de_haller" in p
            assert p["de_haller"] > 0
