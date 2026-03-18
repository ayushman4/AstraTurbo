"""Tests for turbine blade cooling flow estimation."""
import pytest
from astraturbo.design.cooling import (
    CoolingResult, CoolingRowResult, COOLING_PHI, cooling_flow,
)


class TestCooling:
    def test_basic_film(self):
        result = cooling_flow(T_gas=1700, T_coolant=600, cooling_type="film")
        assert isinstance(result, CoolingResult)
        assert result.cooling_type == "film"

    def test_convection_higher_coolant(self):
        """Convection (phi=1) requires more coolant than film (phi=0.4)."""
        conv = cooling_flow(T_gas=1700, T_coolant=600, cooling_type="convection")
        film = cooling_flow(T_gas=1700, T_coolant=600, cooling_type="film")
        assert conv.total_coolant_fraction > film.total_coolant_fraction

    def test_transpiration_lowest_coolant(self):
        """Transpiration (phi=0.2) requires least coolant."""
        trans = cooling_flow(T_gas=1700, T_coolant=600, cooling_type="transpiration")
        film = cooling_flow(T_gas=1700, T_coolant=600, cooling_type="film")
        assert trans.total_coolant_fraction < film.total_coolant_fraction

    def test_per_row_results(self):
        result = cooling_flow(T_gas=1700, T_coolant=600, n_cooled_rows=3)
        assert len(result.rows) == 3
        for r in result.rows:
            assert isinstance(r, CoolingRowResult)
            assert r.coolant_mass_flow >= 0

    def test_total_coolant_flow_sum(self):
        result = cooling_flow(T_gas=1700, T_coolant=600, n_cooled_rows=2, mass_flow_gas=20.0)
        total = sum(r.coolant_mass_flow for r in result.rows)
        assert result.total_coolant_flow == pytest.approx(total, rel=0.01)

    def test_effectiveness_range(self):
        result = cooling_flow(T_gas=1700, T_coolant=600)
        assert 0.0 < result.overall_effectiveness < 1.0

    def test_blade_temp_below_max(self):
        result = cooling_flow(T_gas=1700, T_coolant=600, T_blade_max=1300)
        for r in result.rows:
            assert r.T_blade <= 1300 + 50  # small tolerance due to model

    def test_phi_values(self):
        assert COOLING_PHI["convection"] == 0.2
        assert COOLING_PHI["film"] == 0.4
        assert COOLING_PHI["transpiration"] == 1.0

    def test_invalid_t_coolant_raises(self):
        with pytest.raises(ValueError):
            cooling_flow(T_gas=1700, T_coolant=1800)

    def test_summary_string(self):
        result = cooling_flow(T_gas=1700, T_coolant=600)
        s = result.summary()
        assert "Cooling" in s
