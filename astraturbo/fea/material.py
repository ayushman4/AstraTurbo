"""Turbomachinery material property database.

Aerospace-grade material library covering the 7 material classes used in
gas turbine engines:

  1. Nickel superalloys — turbine blades, vanes, combustors, disks
  2. Titanium alloys — fan, compressor blades, disks
  3. Steels — shafts, casings, structural components
  4. Aluminum alloys — low-temperature structures
  5. Ceramic Matrix Composites (CMC) — next-gen hot section
  6. Thermal Barrier Coatings (TBC) — insulation, oxidation protection
  7. Cobalt / exotic alloys — corrosion-resistant hot section

All properties are at room temperature (20-25 C) unless noted.
For temperature-dependent properties, use max_service_temperature
as the upper bound for applicability.

Sources: MMPDS, ASM International, alloy manufacturer datasheets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Material:
    """Material properties for structural analysis."""

    name: str
    category: str                       # nickel, titanium, steel, aluminum, cmc, coating, cobalt
    density: float                      # kg/m³
    youngs_modulus: float               # Pa (E)
    poisson_ratio: float                # dimensionless (nu)
    yield_strength: float               # Pa
    ultimate_strength: float            # Pa
    thermal_conductivity: float         # W/(m·K)
    specific_heat: float                # J/(kg·K)
    thermal_expansion: float            # 1/K (CTE)
    max_service_temperature: float      # K
    fatigue_limit: float = 0.0          # Pa (endurance limit at 10^7 cycles)
    description: str = ""               # Typical application

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


# =====================================================================
# 1. NICKEL-BASED SUPERALLOYS — Hot section (turbine blades, vanes, combustors)
# =====================================================================

INCONEL_718 = Material(
    name="Inconel_718", category="nickel",
    density=8190.0, youngs_modulus=200e9, poisson_ratio=0.30,
    yield_strength=1035e6, ultimate_strength=1240e6,
    thermal_conductivity=11.4, specific_heat=435.0,
    thermal_expansion=13.0e-6, max_service_temperature=973.0,
    fatigue_limit=550e6,
    description="Workhorse superalloy — compressor disks, LP turbine",
)

INCONEL_625 = Material(
    name="Inconel_625", category="nickel",
    density=8440.0, youngs_modulus=205e9, poisson_ratio=0.31,
    yield_strength=758e6, ultimate_strength=965e6,
    thermal_conductivity=9.8, specific_heat=410.0,
    thermal_expansion=12.8e-6, max_service_temperature=1073.0,
    fatigue_limit=350e6,
    description="Corrosion resistant — combustor, exhaust",
)

RENE_41 = Material(
    name="Rene_41", category="nickel",
    density=8250.0, youngs_modulus=219e9, poisson_ratio=0.31,
    yield_strength=760e6, ultimate_strength=1100e6,
    thermal_conductivity=10.9, specific_heat=421.0,
    thermal_expansion=12.2e-6, max_service_temperature=1143.0,
    fatigue_limit=380e6,
    description="High temp strength — turbine components",
)

RENE_80 = Material(
    name="Rene_80", category="nickel",
    density=8160.0, youngs_modulus=210e9, poisson_ratio=0.30,
    yield_strength=690e6, ultimate_strength=1050e6,
    thermal_conductivity=11.7, specific_heat=418.0,
    thermal_expansion=13.5e-6, max_service_temperature=1255.0,
    fatigue_limit=340e6,
    description="Cast superalloy — turbine blades",
)

HASTELLOY_X = Material(
    name="Hastelloy_X", category="nickel",
    density=8220.0, youngs_modulus=205e9, poisson_ratio=0.32,
    yield_strength=360e6, ultimate_strength=785e6,
    thermal_conductivity=9.2, specific_heat=473.0,
    thermal_expansion=13.8e-6, max_service_temperature=1473.0,
    fatigue_limit=260e6,
    description="Oxidation resistant — combustors, afterburners",
)

WASPALOY = Material(
    name="Waspaloy", category="nickel",
    density=8190.0, youngs_modulus=213e9, poisson_ratio=0.29,
    yield_strength=795e6, ultimate_strength=1275e6,
    thermal_conductivity=11.7, specific_heat=502.0,
    thermal_expansion=12.7e-6, max_service_temperature=1003.0,
    fatigue_limit=480e6,
    description="High strength at temperature — disks, shafts",
)

UDIMET_720 = Material(
    name="Udimet_720", category="nickel",
    density=8080.0, youngs_modulus=222e9, poisson_ratio=0.30,
    yield_strength=1000e6, ultimate_strength=1400e6,
    thermal_conductivity=11.0, specific_heat=460.0,
    thermal_expansion=12.0e-6, max_service_temperature=1023.0,
    fatigue_limit=550e6,
    description="Disk alloy — HP compressor",
)

CMSX_4 = Material(
    name="CMSX-4", category="nickel",
    density=8700.0, youngs_modulus=130e9, poisson_ratio=0.30,
    yield_strength=950e6, ultimate_strength=1100e6,
    thermal_conductivity=12.0, specific_heat=380.0,
    thermal_expansion=12.5e-6, max_service_temperature=1373.0,
    fatigue_limit=400e6,
    description="1st gen single crystal — HP turbine blades",
)

PWA_1484 = Material(
    name="PWA_1484", category="nickel",
    density=8950.0, youngs_modulus=128e9, poisson_ratio=0.30,
    yield_strength=1000e6, ultimate_strength=1200e6,
    thermal_conductivity=11.0, specific_heat=390.0,
    thermal_expansion=12.0e-6, max_service_temperature=1393.0,
    fatigue_limit=450e6,
    description="2nd gen single crystal — turbine blades (Pratt & Whitney)",
)

RENE_N5 = Material(
    name="Rene_N5", category="nickel",
    density=8630.0, youngs_modulus=131e9, poisson_ratio=0.30,
    yield_strength=960e6, ultimate_strength=1150e6,
    thermal_conductivity=11.5, specific_heat=385.0,
    thermal_expansion=12.3e-6, max_service_temperature=1383.0,
    fatigue_limit=420e6,
    description="2nd gen single crystal — turbine (GE)",
)

MAR_M_247 = Material(
    name="MAR-M-247", category="nickel",
    density=8540.0, youngs_modulus=200e9, poisson_ratio=0.30,
    yield_strength=830e6, ultimate_strength=1050e6,
    thermal_conductivity=10.5, specific_heat=400.0,
    thermal_expansion=12.9e-6, max_service_temperature=1253.0,
    fatigue_limit=380e6,
    description="Cast polycrystalline — turbine blades, vanes",
)

INCONEL_713C = Material(
    name="Inconel_713C", category="nickel",
    density=7910.0, youngs_modulus=200e9, poisson_ratio=0.30,
    yield_strength=740e6, ultimate_strength=850e6,
    thermal_conductivity=11.7, specific_heat=444.0,
    thermal_expansion=11.0e-6, max_service_temperature=1143.0,
    fatigue_limit=320e6,
    description="Cast alloy — turbine blades, small engines",
)


# =====================================================================
# 2. TITANIUM ALLOYS — Fan, compressor (weight-critical, moderate temp)
# =====================================================================

TI_6AL_4V = Material(
    name="Ti-6Al-4V", category="titanium",
    density=4430.0, youngs_modulus=113.8e9, poisson_ratio=0.342,
    yield_strength=880e6, ultimate_strength=950e6,
    thermal_conductivity=6.7, specific_heat=526.3,
    thermal_expansion=8.6e-6, max_service_temperature=673.0,
    fatigue_limit=510e6,
    description="Most common Ti alloy — fan, LP compressor",
)

TI_6242 = Material(
    name="Ti-6-2-4-2", category="titanium",
    density=4540.0, youngs_modulus=120e9, poisson_ratio=0.32,
    yield_strength=990e6, ultimate_strength=1070e6,
    thermal_conductivity=7.1, specific_heat=502.0,
    thermal_expansion=8.0e-6, max_service_temperature=813.0,
    fatigue_limit=500e6,
    description="High-temp Ti — compressor blades, disks",
)

TI_5553 = Material(
    name="Ti-5553", category="titanium",
    density=4650.0, youngs_modulus=110e9, poisson_ratio=0.33,
    yield_strength=1200e6, ultimate_strength=1300e6,
    thermal_conductivity=6.2, specific_heat=490.0,
    thermal_expansion=8.5e-6, max_service_temperature=623.0,
    fatigue_limit=550e6,
    description="High-strength beta alloy — landing gear, disks",
)

IMI_834 = Material(
    name="IMI_834", category="titanium",
    density=4550.0, youngs_modulus=120e9, poisson_ratio=0.31,
    yield_strength=1000e6, ultimate_strength=1100e6,
    thermal_conductivity=7.0, specific_heat=500.0,
    thermal_expansion=9.0e-6, max_service_temperature=873.0,
    fatigue_limit=520e6,
    description="High-temp near-alpha alloy — compressor blades",
)

TI_6246 = Material(
    name="Ti-6-2-4-6", category="titanium",
    density=4650.0, youngs_modulus=114e9, poisson_ratio=0.32,
    yield_strength=1100e6, ultimate_strength=1170e6,
    thermal_conductivity=7.7, specific_heat=480.0,
    thermal_expansion=9.4e-6, max_service_temperature=723.0,
    fatigue_limit=530e6,
    description="High-strength alpha-beta alloy — disks, blades",
)


# =====================================================================
# 3. STEELS — Shafts, casings, structural components
# =====================================================================

STEEL_17_4PH = Material(
    name="Steel_17-4PH", category="steel",
    density=7780.0, youngs_modulus=197e9, poisson_ratio=0.27,
    yield_strength=1170e6, ultimate_strength=1310e6,
    thermal_conductivity=18.3, specific_heat=460.0,
    thermal_expansion=10.8e-6, max_service_temperature=623.0,
    fatigue_limit=515e6,
    description="Precipitation-hardened stainless — structural",
)

STEEL_15_5PH = Material(
    name="Steel_15-5PH", category="steel",
    density=7800.0, youngs_modulus=196e9, poisson_ratio=0.27,
    yield_strength=1000e6, ultimate_strength=1070e6,
    thermal_conductivity=18.0, specific_heat=460.0,
    thermal_expansion=10.8e-6, max_service_temperature=623.0,
    fatigue_limit=450e6,
    description="Better toughness than 17-4PH — aerospace components",
)

AISI_4340 = Material(
    name="AISI_4340", category="steel",
    density=7850.0, youngs_modulus=205e9, poisson_ratio=0.29,
    yield_strength=1210e6, ultimate_strength=1380e6,
    thermal_conductivity=44.5, specific_heat=475.0,
    thermal_expansion=12.3e-6, max_service_temperature=673.0,
    fatigue_limit=620e6,
    description="High-strength alloy steel — shafts, gears",
)

MARAGING_300 = Material(
    name="Maraging_300", category="steel",
    density=8000.0, youngs_modulus=190e9, poisson_ratio=0.30,
    yield_strength=2000e6, ultimate_strength=2050e6,
    thermal_conductivity=25.0, specific_heat=450.0,
    thermal_expansion=10.1e-6, max_service_temperature=723.0,
    fatigue_limit=850e6,
    description="Ultra-high strength — shafts, critical fasteners",
)

INCOLOY_909 = Material(
    name="Incoloy_909", category="steel",
    density=8310.0, youngs_modulus=160e9, poisson_ratio=0.30,
    yield_strength=1000e6, ultimate_strength=1170e6,
    thermal_conductivity=14.8, specific_heat=430.0,
    thermal_expansion=7.7e-6, max_service_temperature=923.0,
    fatigue_limit=450e6,
    description="Low CTE — casings, rings (matched expansion with Ti disks)",
)


# =====================================================================
# 4. ALUMINUM ALLOYS — Low-temperature, lightweight structures
# =====================================================================

ALUMINUM_7075 = Material(
    name="Al_7075-T6", category="aluminum",
    density=2810.0, youngs_modulus=71.7e9, poisson_ratio=0.33,
    yield_strength=503e6, ultimate_strength=572e6,
    thermal_conductivity=130.0, specific_heat=960.0,
    thermal_expansion=23.6e-6, max_service_temperature=473.0,
    fatigue_limit=159e6,
    description="High-strength Al — structural, nacelle",
)

ALUMINUM_2024 = Material(
    name="Al_2024-T3", category="aluminum",
    density=2780.0, youngs_modulus=73.1e9, poisson_ratio=0.33,
    yield_strength=345e6, ultimate_strength=483e6,
    thermal_conductivity=121.0, specific_heat=875.0,
    thermal_expansion=23.2e-6, max_service_temperature=473.0,
    fatigue_limit=138e6,
    description="Fatigue-resistant Al — airframe, inlet",
)


# =====================================================================
# 5. CERAMIC MATRIX COMPOSITES (CMC) — Next-gen hot section
# =====================================================================

SIC_SIC_CMC = Material(
    name="SiC-SiC_CMC", category="cmc",
    density=2350.0, youngs_modulus=250e9, poisson_ratio=0.17,
    yield_strength=300e6, ultimate_strength=350e6,
    thermal_conductivity=15.0, specific_heat=700.0,
    thermal_expansion=4.5e-6, max_service_temperature=1588.0,
    fatigue_limit=200e6,
    description="Silicon carbide CMC — turbine shrouds, nozzles (GE LEAP)",
)

OXIDE_OXIDE_CMC = Material(
    name="Oxide-Oxide_CMC", category="cmc",
    density=2800.0, youngs_modulus=100e9, poisson_ratio=0.20,
    yield_strength=170e6, ultimate_strength=220e6,
    thermal_conductivity=4.0, specific_heat=850.0,
    thermal_expansion=8.0e-6, max_service_temperature=1473.0,
    fatigue_limit=120e6,
    description="Alumina-based CMC — combustor liners, exhaust",
)

SI3N4_CERAMIC = Material(
    name="Si3N4", category="cmc",
    density=3200.0, youngs_modulus=310e9, poisson_ratio=0.27,
    yield_strength=700e6, ultimate_strength=900e6,
    thermal_conductivity=28.0, specific_heat=680.0,
    thermal_expansion=3.2e-6, max_service_temperature=1623.0,
    fatigue_limit=350e6,
    description="Silicon nitride — bearings, turbocharger wheels",
)


# =====================================================================
# 6. THERMAL BARRIER COATINGS (TBC)
# =====================================================================

YSZ_TBC = Material(
    name="YSZ_TBC", category="coating",
    density=5600.0, youngs_modulus=40e9, poisson_ratio=0.25,
    yield_strength=50e6, ultimate_strength=100e6,
    thermal_conductivity=1.5, specific_heat=450.0,
    thermal_expansion=10.7e-6, max_service_temperature=1473.0,
    description="Yttria-stabilized zirconia — thermal insulation on blades",
)

MCRALY_COATING = Material(
    name="MCrAlY", category="coating",
    density=7300.0, youngs_modulus=170e9, poisson_ratio=0.30,
    yield_strength=350e6, ultimate_strength=550e6,
    thermal_conductivity=16.0, specific_heat=500.0,
    thermal_expansion=14.0e-6, max_service_temperature=1373.0,
    description="Bond coat — oxidation protection under TBC",
)


# =====================================================================
# 7. COBALT-BASED & EXOTIC ALLOYS
# =====================================================================

HAYNES_188 = Material(
    name="Haynes_188", category="cobalt",
    density=8980.0, youngs_modulus=232e9, poisson_ratio=0.31,
    yield_strength=455e6, ultimate_strength=880e6,
    thermal_conductivity=10.4, specific_heat=403.0,
    thermal_expansion=12.8e-6, max_service_temperature=1363.0,
    fatigue_limit=280e6,
    description="Cobalt superalloy — combustor, transition ducts",
)

HAYNES_25_L605 = Material(
    name="Haynes_25_L-605", category="cobalt",
    density=9130.0, youngs_modulus=225e9, poisson_ratio=0.31,
    yield_strength=475e6, ultimate_strength=1000e6,
    thermal_conductivity=10.1, specific_heat=418.0,
    thermal_expansion=12.3e-6, max_service_temperature=1253.0,
    fatigue_limit=290e6,
    description="Cobalt alloy — turbine vanes, afterburner",
)

C_103_NIOBIUM = Material(
    name="C-103_Niobium", category="exotic",
    density=8860.0, youngs_modulus=100e9, poisson_ratio=0.35,
    yield_strength=350e6, ultimate_strength=480e6,
    thermal_conductivity=42.0, specific_heat=270.0,
    thermal_expansion=7.2e-6, max_service_temperature=1643.0,
    fatigue_limit=200e6,
    description="Niobium alloy — rocket nozzles, hypersonic leading edges",
)


# =====================================================================
# Database registry
# =====================================================================

MATERIAL_DATABASE = {
    # Nickel superalloys
    "inconel_718": INCONEL_718,
    "inconel_625": INCONEL_625,
    "inconel_713c": INCONEL_713C,
    "rene_41": RENE_41,
    "rene_80": RENE_80,
    "rene_n5": RENE_N5,
    "hastelloy_x": HASTELLOY_X,
    "waspaloy": WASPALOY,
    "udimet_720": UDIMET_720,
    "cmsx_4": CMSX_4,
    "pwa_1484": PWA_1484,
    "mar_m_247": MAR_M_247,
    # Titanium alloys
    "ti_6al_4v": TI_6AL_4V,
    "ti_6242": TI_6242,
    "ti_5553": TI_5553,
    "imi_834": IMI_834,
    "ti_6246": TI_6246,
    # Steels
    "steel_17_4ph": STEEL_17_4PH,
    "steel_15_5ph": STEEL_15_5PH,
    "aisi_4340": AISI_4340,
    "maraging_300": MARAGING_300,
    "incoloy_909": INCOLOY_909,
    # Aluminum
    "al_7075": ALUMINUM_7075,
    "al_2024": ALUMINUM_2024,
    # CMC / Ceramics
    "sic_sic_cmc": SIC_SIC_CMC,
    "oxide_oxide_cmc": OXIDE_OXIDE_CMC,
    "si3n4": SI3N4_CERAMIC,
    # Coatings
    "ysz_tbc": YSZ_TBC,
    "mcraly": MCRALY_COATING,
    # Cobalt / Exotic
    "haynes_188": HAYNES_188,
    "haynes_25": HAYNES_25_L605,
    "c_103_niobium": C_103_NIOBIUM,
}


def get_material(name: str) -> Material:
    """Look up a material by name.

    Accepts various formats: 'inconel_718', 'Inconel-718', 'INCONEL 718'.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    mat = MATERIAL_DATABASE.get(key)
    if mat is None:
        available = ", ".join(sorted(MATERIAL_DATABASE.keys()))
        raise ValueError(f"Unknown material '{name}'. Available: {available}")
    return mat


def list_materials(category: str | None = None) -> list[str]:
    """Return list of available material names, optionally filtered by category.

    Categories: nickel, titanium, steel, aluminum, cmc, coating, cobalt, exotic.
    """
    if category:
        cat = category.lower()
        return sorted(
            k for k, v in MATERIAL_DATABASE.items() if v.category == cat
        )
    return sorted(MATERIAL_DATABASE.keys())


def list_categories() -> list[str]:
    """Return list of material categories."""
    return sorted(set(v.category for v in MATERIAL_DATABASE.values()))
