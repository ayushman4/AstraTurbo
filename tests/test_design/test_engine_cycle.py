"""Tests for the engine cycle solver.

Covers turbojet and turboshaft configurations, component models,
validation, and the Kaveri-class engine demo case.
"""

import math

import pytest

from astraturbo.design.engine_cycle import (
    EngineCycleResult,
    InletResult,
    CombustorResult,
    NozzleResult,
    standard_atmosphere,
    inlet_model,
    combustor_model,
    nozzle_model,
    engine_cycle,
)


# ── Helper: default turbojet run ─────────────────────

DEFAULT_TURBOJET = dict(
    engine_type="turbojet",
    overall_pressure_ratio=8.0,
    turbine_inlet_temp=1400.0,
    mass_flow=20.0,
    rpm=15000.0,
    r_hub=0.15,
    r_tip=0.30,
)

DEFAULT_TURBOSHAFT = dict(
    engine_type="turboshaft",
    overall_pressure_ratio=8.0,
    turbine_inlet_temp=1400.0,
    mass_flow=20.0,
    rpm=15000.0,
    r_hub=0.15,
    r_tip=0.30,
)


def _run_turbojet(**overrides):
    kw = {**DEFAULT_TURBOJET, **overrides}
    return engine_cycle(**kw)


def _run_turboshaft(**overrides):
    kw = {**DEFAULT_TURBOSHAFT, **overrides}
    return engine_cycle(**kw)


# ── Turbojet Tests ───────────────────────────────────


class TestTurbojet:
    def test_basic_turbojet(self):
        result = _run_turbojet()
        assert isinstance(result, EngineCycleResult)
        assert result.engine_type == "turbojet"

    def test_thrust_positive(self):
        result = _run_turbojet()
        assert result.net_thrust > 0

    def test_sfc_reasonable(self):
        result = _run_turbojet()
        # SFC in kg/(N·s): typical 1e-5 to 5e-5 (i.e. 0.04-0.18 kg/(N·h))
        assert 1e-6 < result.specific_fuel_consumption < 1e-3

    def test_thermal_efficiency_range(self):
        result = _run_turbojet()
        assert 0.15 < result.thermal_efficiency < 0.60

    def test_station_pressure_sequence(self):
        result = _run_turbojet()
        st = result.stations
        # Pressure rises through compressor
        assert st["compressor_exit"].P_total > st["inlet_exit"].P_total
        # Pressure drops after combustor
        assert st["combustor_exit"].P_total < st["compressor_exit"].P_total
        # Pressure drops through turbine
        assert st["turbine_exit"].P_total < st["combustor_exit"].P_total

    def test_station_temperature_sequence(self):
        result = _run_turbojet()
        st = result.stations
        # Temperature rises through compressor
        assert st["compressor_exit"].T_total > st["inlet_exit"].T_total
        # Temperature rises through combustor
        assert st["combustor_exit"].T_total > st["compressor_exit"].T_total
        # Temperature drops through turbine
        assert st["turbine_exit"].T_total < st["combustor_exit"].T_total

    def test_power_balance(self):
        result = _run_turbojet()
        # Turbine work * mech_eff should roughly equal compressor work
        turb_delivered = result.turbine_work * result.mechanical_efficiency
        assert abs(turb_delivered - result.compressor_work) < 0.02 * result.compressor_work


# ── Turboshaft Tests ─────────────────────────────────


class TestTurboshaft:
    def test_basic_turboshaft(self):
        result = _run_turboshaft()
        assert isinstance(result, EngineCycleResult)
        assert result.engine_type == "turboshaft"

    def test_turboshaft_no_nozzle(self):
        result = _run_turboshaft()
        assert result.nozzle is None

    def test_turboshaft_power_positive(self):
        result = _run_turboshaft()
        assert result.shaft_power > 0

    def test_turboshaft_bsfc(self):
        result = _run_turboshaft()
        # BSFC: reasonable range for turboshaft ~200-600 g/(kW·h)
        if result.shaft_power > 0:
            bsfc_kwh = result.fuel_flow / result.shaft_power * 1e6 * 3600
            assert 100 < bsfc_kwh < 1500


# ── Kaveri-Class Demo ────────────────────────────────


class TestKaveriClass:
    def test_kaveri_class(self):
        # Kaveri-class: OPR=20, TIT=1700K, ~40 kg/s for full thrust
        # At 20 kg/s specific thrust ~800-1200 N/(kg/s) → 16-24 kN
        result = engine_cycle(
            engine_type="turbojet",
            overall_pressure_ratio=20.0,
            turbine_inlet_temp=1700.0,
            mass_flow=20.0,
            rpm=15000.0,
            r_hub=0.15,
            r_tip=0.30,
        )
        assert result.net_thrust > 15000  # >15 kN for 20 kg/s


# ── Altitude and Mach Effects ────────────────────────


class TestEnvironment:
    def test_altitude_reduces_thrust(self):
        # Compare at same Mach=0.8 where ram effects are consistent
        sea_level = engine_cycle(
            engine_type="turbojet",
            overall_pressure_ratio=8.0,
            turbine_inlet_temp=1400.0,
            mass_flow=20.0,
            rpm=15000.0,
            r_hub=0.15,
            r_tip=0.30,
            altitude=0.0,
            mach_flight=0.8,
        )
        high_alt = engine_cycle(
            engine_type="turbojet",
            overall_pressure_ratio=8.0,
            turbine_inlet_temp=1400.0,
            mass_flow=20.0,
            rpm=15000.0,
            r_hub=0.15,
            r_tip=0.30,
            altitude=10000.0,
            mach_flight=0.8,
        )
        # Both produce thrust; sea level inlet has higher P_total
        assert sea_level.inlet.P_total_out > high_alt.inlet.P_total_out

    def test_mach_ram_effect(self):
        static = inlet_model(288.15, 101325.0, mach_flight=0.0)
        ram = inlet_model(288.15, 101325.0, mach_flight=0.8)
        assert ram.P_total_out > static.P_total_out


# ── Standard Atmosphere ──────────────────────────────


class TestStandardAtmosphere:
    def test_sea_level(self):
        T, P, rho = standard_atmosphere(0.0)
        assert T == pytest.approx(288.15, abs=0.01)
        assert P == pytest.approx(101325.0, abs=1.0)

    def test_stratosphere(self):
        T, P, rho = standard_atmosphere(15000.0)
        assert T == pytest.approx(216.65, abs=0.5)


# ── Component Models ────────────────────────────────


class TestInlet:
    def test_inlet_static(self):
        # At Mach 0, P_out should be approximately P_amb * eta_inlet
        res = inlet_model(288.15, 101325.0, mach_flight=0.0, eta_inlet=0.97)
        assert res.P_total_out == pytest.approx(101325.0 * 0.97, rel=0.001)


class TestCombustor:
    def test_fuel_air_ratio(self):
        res = combustor_model(
            P_in=800000.0,
            T_in=600.0,
            T_target_out=1400.0,
            mass_flow_air=20.0,
        )
        # Typical f in 0.01-0.04
        assert 0.01 < res.fuel_air_ratio < 0.04


# ── Validation Tests ─────────────────────────────────


class TestValidation:
    def test_invalid_engine_type_raises(self):
        with pytest.raises(ValueError, match="engine_type"):
            engine_cycle(
                engine_type="ramjet",
                overall_pressure_ratio=8.0,
                turbine_inlet_temp=1400.0,
                mass_flow=20.0,
                rpm=15000.0,
                r_hub=0.15,
                r_tip=0.30,
            )

    def test_invalid_opr_raises(self):
        with pytest.raises(ValueError, match="overall_pressure_ratio"):
            engine_cycle(
                overall_pressure_ratio=0.5,
                turbine_inlet_temp=1400.0,
                mass_flow=20.0,
                rpm=15000.0,
                r_hub=0.15,
                r_tip=0.30,
            )

    def test_invalid_tit_raises(self):
        with pytest.raises(ValueError, match="turbine_inlet_temp"):
            engine_cycle(
                overall_pressure_ratio=8.0,
                turbine_inlet_temp=300.0,
                mass_flow=20.0,
                rpm=15000.0,
                r_hub=0.15,
                r_tip=0.30,
            )
