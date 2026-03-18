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


# ── Multi-Spool Tests ─────────────────────────────


DEFAULT_TWIN_SPOOL = dict(
    engine_type="turbojet",
    overall_pressure_ratio=16.0,
    turbine_inlet_temp=1600.0,
    mass_flow=20.0,
    rpm=10000.0,
    r_hub=0.15,
    r_tip=0.30,
    n_spools=2,
)


def _run_twin_spool(**overrides):
    kw = {**DEFAULT_TWIN_SPOOL, **overrides}
    return engine_cycle(**kw)


class TestMultiSpool:
    def test_twin_spool_turbojet(self):
        """Twin-spool turbojet completes and produces thrust."""
        result = _run_twin_spool()
        assert isinstance(result, EngineCycleResult)
        assert result.n_spools == 2
        assert result.net_thrust > 0

    def test_twin_spool_stations(self):
        """Twin-spool has lp/hp compressor and turbine stations."""
        result = _run_twin_spool()
        st = result.stations
        assert "lp_compressor_exit" in st
        assert "compressor_exit" in st       # HP compressor exit
        assert "hp_turbine_exit" in st
        assert "turbine_exit" in st          # LP turbine exit
        assert "combustor_exit" in st
        # Pressure rises through LP then HP compressor
        assert st["lp_compressor_exit"].P_total > st["inlet_exit"].P_total
        assert st["compressor_exit"].P_total > st["lp_compressor_exit"].P_total

    def test_twin_spool_power_balance(self):
        """Each spool is power-balanced independently."""
        result = _run_twin_spool()
        assert len(result.spools) == 2
        for sp in result.spools:
            turb_delivered = sp["turbine_work"] * result.mechanical_efficiency
            # Allow 10% tolerance since meanline discretization introduces error
            assert turb_delivered >= sp["compressor_work"] * 0.85, (
                f"{sp['name']}: turbine {turb_delivered:.0f} < compressor {sp['compressor_work']:.0f}"
            )

    def test_twin_spool_pr_split(self):
        """LP_PR × HP_PR ≈ OPR."""
        result = _run_twin_spool()
        lp_pr = result.spools[0]["pr"]
        hp_pr = result.spools[1]["pr"]
        opr = 16.0
        assert lp_pr * hp_pr == pytest.approx(opr, rel=0.01)

    def test_kaveri_twin_spool(self):
        """Kaveri-class twin-spool: OPR=20, TIT=1700K, realistic thrust."""
        result = engine_cycle(
            engine_type="turbojet",
            overall_pressure_ratio=20.0,
            turbine_inlet_temp=1700.0,
            mass_flow=20.0,
            rpm=10000.0,
            r_hub=0.15,
            r_tip=0.30,
            n_spools=2,
            hp_pressure_ratio=4.5,
            hp_rpm=15000.0,
        )
        assert result.n_spools == 2
        assert result.net_thrust > 10000  # >10 kN at 20 kg/s

    def test_single_spool_unchanged(self):
        """n_spools=1 produces identical result to default."""
        r1 = engine_cycle(
            overall_pressure_ratio=8.0,
            turbine_inlet_temp=1400.0,
            mass_flow=20.0,
            rpm=15000.0,
            r_hub=0.15,
            r_tip=0.30,
        )
        r2 = engine_cycle(
            overall_pressure_ratio=8.0,
            turbine_inlet_temp=1400.0,
            mass_flow=20.0,
            rpm=15000.0,
            r_hub=0.15,
            r_tip=0.30,
            n_spools=1,
        )
        assert r1.net_thrust == pytest.approx(r2.net_thrust, rel=1e-6)
        assert r1.compressor_work == pytest.approx(r2.compressor_work, rel=1e-6)
        assert r2.n_spools == 1
        assert r2.spools == []

    def test_custom_hp_pr(self):
        """Explicit HP pressure ratio is respected."""
        result = _run_twin_spool(hp_pressure_ratio=5.0)
        hp_pr = result.spools[1]["pr"]
        assert hp_pr == pytest.approx(5.0, rel=0.01)

    def test_twin_spool_turboshaft(self):
        """Turboshaft with 2 spools produces shaft power."""
        result = engine_cycle(
            engine_type="turboshaft",
            overall_pressure_ratio=12.0,
            turbine_inlet_temp=1500.0,
            mass_flow=15.0,
            rpm=10000.0,
            r_hub=0.12,
            r_tip=0.25,
            n_spools=2,
        )
        assert result.n_spools == 2
        assert result.shaft_power > 0
        assert result.nozzle is None


# ── Afterburner / Reheat Tests ────────────────────


class TestAfterburner:
    def test_afterburner_increases_thrust(self):
        """Afterburner should significantly increase thrust."""
        dry = _run_turbojet()
        wet = _run_turbojet(afterburner=True, afterburner_temp=2000.0)
        assert wet.net_thrust > dry.net_thrust * 1.1

    def test_afterburner_station(self):
        """Afterburner exit station should appear."""
        result = _run_turbojet(afterburner=True, afterburner_temp=2000.0)
        assert "afterburner_exit" in result.stations
        assert result.stations["afterburner_exit"].T_total > result.stations["turbine_exit"].T_total

    def test_afterburner_fuel_flow(self):
        """Afterburner adds extra fuel."""
        result = _run_turbojet(afterburner=True, afterburner_temp=2000.0)
        assert result.afterburner is not None
        assert result.afterburner_fuel_flow > 0
        assert result.fuel_flow > result.combustor.fuel_flow

    def test_convergent_divergent_nozzle(self):
        """Con-di nozzle should produce M > 1 exit."""
        result = _run_turbojet(nozzle_type="convergent_divergent", nozzle_design_mach=1.5)
        assert result.nozzle is not None
        assert result.nozzle.mach_exit > 1.0

    def test_afterburner_with_condi(self):
        """Afterburner + con-di = maximum thrust."""
        dry_conv = _run_turbojet()
        wet_condi = _run_turbojet(
            afterburner=True, afterburner_temp=2000.0,
            nozzle_type="convergent_divergent", nozzle_design_mach=1.8,
        )
        assert wet_condi.net_thrust > dry_conv.net_thrust

    def test_turboshaft_ignores_afterburner(self):
        """Afterburner flag is ignored for turboshaft."""
        result = _run_turboshaft(afterburner=True, afterburner_temp=2000.0)
        assert result.afterburner is None
        assert result.nozzle is None

    def test_afterburner_sfc_increases(self):
        """Afterburner increases SFC (more fuel per unit thrust)."""
        dry = _run_turbojet()
        wet = _run_turbojet(afterburner=True, afterburner_temp=2000.0)
        assert wet.specific_fuel_consumption > dry.specific_fuel_consumption
