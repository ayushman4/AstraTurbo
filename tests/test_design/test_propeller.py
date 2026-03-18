"""Tests for propeller/rotor design."""
import math
import pytest
from astraturbo.design.propeller import PropellerResult, propeller_design


class TestPropeller:
    def test_basic_hover(self):
        result = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000)
        assert isinstance(result, PropellerResult)
        assert result.thrust == pytest.approx(50, rel=0.01)

    def test_hover_fm_range(self):
        result = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000)
        # FM should be 0.5-0.9 for a reasonable rotor
        assert 0.4 < result.figure_of_merit < 1.0

    def test_forward_flight_efficiency(self):
        result = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000, V_flight=20.0)
        assert result.efficiency > 0
        assert result.efficiency < 1.0

    def test_tip_mach_subsonic(self):
        result = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000)
        assert result.tip_mach < 1.0

    def test_disk_loading(self):
        result = propeller_design(thrust_required=100, n_blades=2, diameter=1.0, rpm=5000)
        A = math.pi * 0.5**2
        assert result.disk_loading == pytest.approx(100 / A, rel=0.01)

    def test_ct_positive(self):
        result = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000)
        assert result.CT > 0

    def test_cp_positive(self):
        result = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000)
        assert result.CP > 0

    def test_altitude_effect(self):
        sea = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000, altitude=0)
        high = propeller_design(thrust_required=50, n_blades=3, diameter=0.5, rpm=8000, altitude=3000)
        # Higher altitude = less dense air = more power needed
        assert high.power > sea.power

    def test_invalid_thrust_raises(self):
        with pytest.raises(ValueError):
            propeller_design(thrust_required=-10, n_blades=3, diameter=0.5, rpm=8000)
