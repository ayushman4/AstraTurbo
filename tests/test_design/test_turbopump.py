"""Tests for turbopump assembly."""
import pytest
from astraturbo.design.turbopump import TurbopumpResult, turbopump


class TestTurbopump:
    def test_basic_lox(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
            fluid_name="LOX",
        )
        assert isinstance(result, TurbopumpResult)

    def test_shaft_balance(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
        )
        # Turbine power * eta_mech should be >= pump power
        assert result.shaft_power >= result.pump_power * 0.8

    def test_pump_result_present(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
        )
        assert result.pump is not None
        assert result.pump.head == pytest.approx(500)

    def test_turbine_result_present(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
        )
        assert result.turbine is not None
        assert result.turbine.n_stages >= 1

    def test_power_margin(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
        )
        # Power margin should be near zero (designed to match)
        assert result.power_margin > -0.5

    def test_cycle_type(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
            cycle_type="staged_combustion",
        )
        assert result.cycle_type == "staged_combustion"

    def test_overall_efficiency(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
        )
        assert result.overall_efficiency > 0

    def test_summary_string(self):
        result = turbopump(
            pump_head=500, pump_flow_rate=0.1, fluid_density=1141.0,
            turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
        )
        s = result.summary()
        assert "Turbopump" in s

    def test_invalid_head_raises(self):
        with pytest.raises(ValueError):
            turbopump(
                pump_head=-100, pump_flow_rate=0.1, fluid_density=1141.0,
                turbine_inlet_temp=900, turbine_inlet_pressure=5000000, rpm=30000,
            )
