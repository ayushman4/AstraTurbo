"""Turbomachinery material property database.

Common materials used in compressor and turbine blades with their
mechanical and thermal properties at reference temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Material:
    """Material properties for structural analysis."""

    name: str
    density: float                  # kg/m³
    youngs_modulus: float           # Pa (E)
    poisson_ratio: float            # dimensionless (nu)
    yield_strength: float           # Pa
    ultimate_strength: float        # Pa
    thermal_conductivity: float     # W/(m·K)
    specific_heat: float            # J/(kg·K)
    thermal_expansion: float        # 1/K (CTE)
    max_service_temperature: float  # K
    fatigue_limit: float = 0.0      # Pa (endurance limit at 10^7 cycles)

    def to_calculix_format(self) -> str:
        """Format material for CalculiX input file."""
        lines = [
            f"*MATERIAL, NAME={self.name}",
            "*ELASTIC",
            f"{self.youngs_modulus:.3e}, {self.poisson_ratio}",
            "*DENSITY",
            f"{self.density}",
            "*EXPANSION",
            f"{self.thermal_expansion}",
            "*CONDUCTIVITY",
            f"{self.thermal_conductivity}",
            "*SPECIFIC HEAT",
            f"{self.specific_heat}",
        ]
        return "\n".join(lines)

    def to_abaqus_format(self) -> str:
        """Format material for Abaqus input file (same as CalculiX)."""
        return self.to_calculix_format()


# ── Material Database ──

INCONEL_718 = Material(
    name="Inconel_718",
    density=8190.0,
    youngs_modulus=200e9,
    poisson_ratio=0.30,
    yield_strength=1035e6,
    ultimate_strength=1240e6,
    thermal_conductivity=11.4,
    specific_heat=435.0,
    thermal_expansion=13.0e-6,
    max_service_temperature=973.0,
    fatigue_limit=550e6,
)

INCONEL_625 = Material(
    name="Inconel_625",
    density=8440.0,
    youngs_modulus=205e9,
    poisson_ratio=0.31,
    yield_strength=758e6,
    ultimate_strength=965e6,
    thermal_conductivity=9.8,
    specific_heat=410.0,
    thermal_expansion=12.8e-6,
    max_service_temperature=1073.0,
    fatigue_limit=350e6,
)

TI_6AL_4V = Material(
    name="Ti-6Al-4V",
    density=4430.0,
    youngs_modulus=113.8e9,
    poisson_ratio=0.342,
    yield_strength=880e6,
    ultimate_strength=950e6,
    thermal_conductivity=6.7,
    specific_heat=526.3,
    thermal_expansion=8.6e-6,
    max_service_temperature=673.0,
    fatigue_limit=510e6,
)

CMSX_4 = Material(
    name="CMSX-4",
    density=8700.0,
    youngs_modulus=130e9,
    poisson_ratio=0.30,
    yield_strength=950e6,
    ultimate_strength=1100e6,
    thermal_conductivity=12.0,
    specific_heat=380.0,
    thermal_expansion=12.5e-6,
    max_service_temperature=1373.0,
    fatigue_limit=400e6,
)

STEEL_17_4PH = Material(
    name="Steel_17-4PH",
    density=7780.0,
    youngs_modulus=197e9,
    poisson_ratio=0.27,
    yield_strength=1170e6,
    ultimate_strength=1310e6,
    thermal_conductivity=18.3,
    specific_heat=460.0,
    thermal_expansion=10.8e-6,
    max_service_temperature=623.0,
    fatigue_limit=515e6,
)

ALUMINUM_7075 = Material(
    name="Al_7075-T6",
    density=2810.0,
    youngs_modulus=71.7e9,
    poisson_ratio=0.33,
    yield_strength=503e6,
    ultimate_strength=572e6,
    thermal_conductivity=130.0,
    specific_heat=960.0,
    thermal_expansion=23.6e-6,
    max_service_temperature=473.0,
    fatigue_limit=159e6,
)


MATERIAL_DATABASE = {
    "inconel_718": INCONEL_718,
    "inconel_625": INCONEL_625,
    "ti_6al_4v": TI_6AL_4V,
    "cmsx_4": CMSX_4,
    "steel_17_4ph": STEEL_17_4PH,
    "al_7075": ALUMINUM_7075,
}


def get_material(name: str) -> Material:
    """Look up a material by name."""
    mat = MATERIAL_DATABASE.get(name.lower().replace("-", "_").replace(" ", "_"))
    if mat is None:
        available = ", ".join(sorted(MATERIAL_DATABASE.keys()))
        raise ValueError(f"Unknown material '{name}'. Available: {available}")
    return mat


def list_materials() -> list[str]:
    """Return list of available material names."""
    return sorted(MATERIAL_DATABASE.keys())
