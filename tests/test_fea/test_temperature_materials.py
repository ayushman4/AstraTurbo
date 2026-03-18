"""Tests for temperature-dependent material properties."""

import pytest
from astraturbo.fea.material import Material, get_material, list_materials


class TestTemperatureDependentProperties:
    """Verify temperature interpolation for key Kaveri-relevant alloys."""

    def test_inconel718_room_temp_matches_base(self):
        """At room temp, interpolated value should match base property."""
        mat = get_material("inconel_718")
        assert abs(mat.youngs_modulus_at(293) - 200e9) < 1e6
        assert abs(mat.yield_strength_at(293) - 1035e6) < 1e6

    def test_inconel718_modulus_decreases_with_temp(self):
        """E should decrease monotonically from 293K to 973K."""
        mat = get_material("inconel_718")
        E_293 = mat.youngs_modulus_at(293)
        E_673 = mat.youngs_modulus_at(673)
        E_973 = mat.youngs_modulus_at(973)
        assert E_293 > E_673 > E_973

    def test_inconel718_yield_drops_significantly_at_973K(self):
        """At 973K, yield should be ~56% of room temp (580 vs 1035 MPa)."""
        mat = get_material("inconel_718")
        ratio = mat.yield_strength_at(973) / mat.yield_strength_at(293)
        assert 0.45 < ratio < 0.65, f"Yield ratio at 973K: {ratio:.2f}"

    def test_inconel718_conductivity_increases_with_temp(self):
        """Thermal conductivity should increase with temperature."""
        mat = get_material("inconel_718")
        k_293 = mat.thermal_conductivity_at(293)
        k_973 = mat.thermal_conductivity_at(973)
        assert k_973 > k_293

    def test_cmsx4_modulus_drops_55pct_at_1373K(self):
        """CMSX-4 E at 1373K should be ~55% lower than room temp."""
        mat = get_material("cmsx_4")
        ratio = mat.youngs_modulus_at(1373) / mat.youngs_modulus_at(293)
        assert 0.35 < ratio < 0.55, f"E ratio at 1373K: {ratio:.2f}"

    def test_cmsx4_yield_drops_at_max_temp(self):
        """CMSX-4 yield drops significantly at max service temperature."""
        mat = get_material("cmsx_4")
        sy_room = mat.yield_strength_at(293)
        sy_hot = mat.yield_strength_at(1373)
        assert sy_hot < sy_room * 0.4

    def test_ti6al4v_has_temp_data(self):
        """Ti-6Al-4V should have temperature tables."""
        mat = get_material("ti_6al_4v")
        assert len(mat.youngs_modulus_table) > 0
        assert len(mat.yield_strength_table) > 0

    def test_ti6al4v_modulus_at_673K(self):
        """Ti-6Al-4V E at 673K should be ~79% of room temp."""
        mat = get_material("ti_6al_4v")
        ratio = mat.youngs_modulus_at(673) / mat.youngs_modulus_at(293)
        assert 0.70 < ratio < 0.85

    def test_hastelloy_x_survives_to_1473K(self):
        """Hastelloy X has data up to 1473K (combustor alloy)."""
        mat = get_material("hastelloy_x")
        E_hot = mat.youngs_modulus_at(1473)
        assert E_hot > 0
        assert E_hot < mat.youngs_modulus * 0.6

    def test_haynes188_combustor_temps(self):
        """Haynes 188 at typical combustor liner temp (1100K)."""
        mat = get_material("haynes_188")
        props = mat.properties_at(1073)
        assert props["has_temp_data"]
        assert props["youngs_modulus_GPa"] < 200  # Room temp is 232

    def test_rene_n5_single_crystal(self):
        """Rene N5 should have temp data for turbine blade analysis."""
        mat = get_material("rene_n5")
        assert len(mat.youngs_modulus_table) >= 6
        # At 1073K, E should be well below room temp
        ratio = mat.youngs_modulus_at(1073) / mat.youngs_modulus
        assert ratio < 0.80

    def test_interpolation_midpoint(self):
        """Test linear interpolation at a midpoint between table entries."""
        mat = get_material("inconel_718")
        # Between 293K (200 GPa) and 473K (190 GPa), midpoint ~367K should be ~195 GPa
        E_mid = mat.youngs_modulus_at(383)
        assert 193e9 < E_mid < 197e9

    def test_interpolation_below_table(self):
        """Below the lowest table temp, return lowest table value."""
        mat = get_material("inconel_718")
        E_cold = mat.youngs_modulus_at(100)
        assert E_cold == mat.youngs_modulus_table[0][1]

    def test_interpolation_above_table(self):
        """Above the highest table temp, return highest table value."""
        mat = get_material("inconel_718")
        E_above = mat.youngs_modulus_at(2000)
        assert E_above == mat.youngs_modulus_table[-1][1]

    def test_no_table_returns_base(self):
        """Material without temp tables returns room-temp base value."""
        mat = get_material("al_7075")
        assert mat.youngs_modulus_at(400) == mat.youngs_modulus
        assert mat.yield_strength_at(400) == mat.yield_strength

    def test_properties_at_returns_dict(self):
        """properties_at() should return a complete dict."""
        mat = get_material("cmsx_4")
        props = mat.properties_at(1073)
        assert "temperature_K" in props
        assert "youngs_modulus_Pa" in props
        assert "yield_strength_MPa" in props
        assert "has_temp_data" in props
        assert props["has_temp_data"] is True

    def test_calculix_format_with_temperature(self):
        """to_calculix_format(temperature=...) should include temp in output."""
        mat = get_material("inconel_718")
        fmt = mat.to_calculix_format(temperature=873)
        assert "*ELASTIC" in fmt
        assert "873" in fmt

    def test_calculix_format_full_table(self):
        """to_calculix_format() without temp should write full table."""
        mat = get_material("inconel_718")
        fmt = mat.to_calculix_format()
        # Should have multiple elastic entries (one per temp point)
        elastic_lines = [l for l in fmt.split("\n") if "293" in l or "973" in l]
        assert len(elastic_lines) >= 2

    def test_materials_with_temp_data_count(self):
        """At least 6 materials should have temperature-dependent data."""
        count = 0
        for name in list_materials():
            mat = get_material(name)
            if mat.youngs_modulus_table:
                count += 1
        assert count >= 6, f"Only {count} materials have temp data"
