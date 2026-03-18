"""Tests for centrifugal pump design."""
import math
import pytest
from astraturbo.design.pump import PumpResult, centrifugal_pump, FLUIDS


class TestPump:
    def test_basic_water_pump(self):
        result = centrifugal_pump(head=50, flow_rate=0.01, rpm=3000)
        assert isinstance(result, PumpResult)
        assert result.fluid_name == "water"

    def test_lox_pump(self):
        result = centrifugal_pump(head=500, flow_rate=0.1, rpm=30000, fluid_name="LOX")
        assert result.fluid_density == pytest.approx(1141.0)

    def test_rp1_pump(self):
        result = centrifugal_pump(head=500, flow_rate=0.1, rpm=30000, fluid_name="RP-1")
        assert result.fluid_density == pytest.approx(810.0)

    def test_lh2_pump(self):
        result = centrifugal_pump(head=500, flow_rate=0.1, rpm=30000, fluid_name="LH2")
        assert result.fluid_density == pytest.approx(71.0)

    def test_specific_speed_positive(self):
        result = centrifugal_pump(head=100, flow_rate=0.05, rpm=5000)
        assert result.specific_speed > 0

    def test_npsh_positive(self):
        result = centrifugal_pump(head=100, flow_rate=0.05, rpm=5000)
        assert result.npsh_required > 0

    def test_power_scales_with_head(self):
        low = centrifugal_pump(head=50, flow_rate=0.05, rpm=5000)
        high = centrifugal_pump(head=200, flow_rate=0.05, rpm=5000)
        assert high.power_kW > low.power_kW

    def test_efficiency_range(self):
        result = centrifugal_pump(head=100, flow_rate=0.05, rpm=5000)
        assert 0.3 < result.efficiency < 0.95

    def test_impeller_diameter_positive(self):
        result = centrifugal_pump(head=100, flow_rate=0.05, rpm=5000)
        assert result.impeller_diameter > 0

    def test_invalid_head_raises(self):
        with pytest.raises(ValueError):
            centrifugal_pump(head=-10, flow_rate=0.05, rpm=5000)
