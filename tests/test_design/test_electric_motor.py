"""Tests for electric motor sizing."""
import math
import pytest
from astraturbo.design.electric_motor import ElectricMotorResult, electric_motor


class TestElectricMotor:
    def test_basic_bldc(self):
        result = electric_motor(shaft_power=50000, rpm=8000, voltage=400)
        assert isinstance(result, ElectricMotorResult)
        assert result.motor_type == "BLDC"

    def test_torque_equals_power_over_omega(self):
        result = electric_motor(shaft_power=50000, rpm=8000, voltage=400)
        omega = 8000 * 2 * math.pi / 60
        assert result.torque == pytest.approx(50000 / omega, rel=1e-6)

    def test_kv_equals_rpm_over_voltage(self):
        result = electric_motor(shaft_power=50000, rpm=8000, voltage=400)
        assert result.motor_constant_kv == pytest.approx(8000 / 400, rel=1e-6)

    def test_weight_bldc(self):
        result = electric_motor(shaft_power=7000, rpm=5000, voltage=48)
        # 7 kW/kg for BLDC -> 7kW motor should weigh 1 kg
        assert result.weight_kg == pytest.approx(1.0, rel=0.01)

    def test_weight_pmsm(self):
        result = electric_motor(shaft_power=5000, rpm=5000, voltage=48, motor_type="PMSM")
        # 5 kW/kg for PMSM -> 5kW motor should weigh 1 kg
        assert result.weight_kg == pytest.approx(1.0, rel=0.01)

    def test_efficiency_at_full_load(self):
        result = electric_motor(shaft_power=50000, rpm=8000, voltage=400, eta_peak=0.92, load_fraction=1.0)
        assert result.efficiency == pytest.approx(0.92, rel=0.01)

    def test_efficiency_drops_at_partial_load(self):
        full = electric_motor(shaft_power=50000, rpm=8000, voltage=400, load_fraction=1.0)
        partial = electric_motor(shaft_power=50000, rpm=8000, voltage=400, load_fraction=0.5)
        assert partial.efficiency < full.efficiency

    def test_thermal_margin_positive(self):
        result = electric_motor(shaft_power=50000, rpm=8000, voltage=400)
        assert result.thermal_margin >= 0

    def test_invalid_power_raises(self):
        with pytest.raises(ValueError):
            electric_motor(shaft_power=-100, rpm=8000, voltage=400)
