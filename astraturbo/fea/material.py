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

Room-temperature values are stored as base properties. Temperature-dependent
tables are provided for key engine alloys (Inconel 718, CMSX-4, Ti-6Al-4V,
and others) — use ``youngs_modulus_at(T)``, ``yield_strength_at(T)``, and
``thermal_conductivity_at(T)`` for hot-section analysis.

Sources: MMPDS, ASM International, alloy manufacturer datasheets.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Material:
    """Material properties for structural analysis.

    Base properties are at room temperature (~293 K). For hot-section
    analysis, use the ``_at(T)`` methods which interpolate from
    temperature-dependent tables when available, falling back to the
    room-temperature value if no table is provided.
    """

    name: str
    category: str                       # nickel, titanium, steel, aluminum, cmc, coating, cobalt
    density: float                      # kg/m³
    youngs_modulus: float               # Pa (E) at ~293 K
    poisson_ratio: float                # dimensionless (nu)
    yield_strength: float               # Pa at ~293 K
    ultimate_strength: float            # Pa at ~293 K
    thermal_conductivity: float         # W/(m·K) at ~293 K
    specific_heat: float                # J/(kg·K)
    thermal_expansion: float            # 1/K (CTE)
    max_service_temperature: float      # K
    fatigue_limit: float = 0.0          # Pa (endurance limit at 10^7 cycles)
    description: str = ""               # Typical application

    # Temperature-dependent tables: list of (T_kelvin, value) tuples,
    # sorted by ascending temperature.  When present, the _at(T) methods
    # linearly interpolate; when absent they return the room-temp value.
    youngs_modulus_table: list[tuple[float, float]] = field(default_factory=list)
    yield_strength_table: list[tuple[float, float]] = field(default_factory=list)
    thermal_conductivity_table: list[tuple[float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Temperature-dependent interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def _interp(table: list[tuple[float, float]], T: float, fallback: float) -> float:
        """Linearly interpolate a (T, value) table at temperature T."""
        if not table:
            return fallback
        if T <= table[0][0]:
            return table[0][1]
        if T >= table[-1][0]:
            return table[-1][1]
        for i in range(len(table) - 1):
            t0, v0 = table[i]
            t1, v1 = table[i + 1]
            if t0 <= T <= t1:
                frac = (T - t0) / (t1 - t0) if t1 != t0 else 0.0
                return v0 + frac * (v1 - v0)
        return fallback

    def youngs_modulus_at(self, T: float) -> float:
        """Young's modulus (Pa) at temperature T (K)."""
        return self._interp(self.youngs_modulus_table, T, self.youngs_modulus)

    def yield_strength_at(self, T: float) -> float:
        """Yield strength (Pa) at temperature T (K)."""
        return self._interp(self.yield_strength_table, T, self.yield_strength)

    def thermal_conductivity_at(self, T: float) -> float:
        """Thermal conductivity (W/m-K) at temperature T (K)."""
        return self._interp(self.thermal_conductivity_table, T, self.thermal_conductivity)

    def properties_at(self, T: float) -> dict[str, float]:
        """Return all temperature-dependent properties at T (K)."""
        return {
            "temperature_K": T,
            "youngs_modulus_Pa": self.youngs_modulus_at(T),
            "yield_strength_Pa": self.yield_strength_at(T),
            "thermal_conductivity_W_mK": self.thermal_conductivity_at(T),
            "youngs_modulus_GPa": self.youngs_modulus_at(T) / 1e9,
            "yield_strength_MPa": self.yield_strength_at(T) / 1e6,
            "has_temp_data": bool(self.youngs_modulus_table),
        }

    def to_calculix_format(self, temperature: float | None = None) -> str:
        """Format material for CalculiX input file.

        Args:
            temperature: If given, output temperature-dependent cards
                using the full table. If None, output room-temp values.
        """
        lines = [
            f"*MATERIAL, NAME={self.name}",
        ]

        # Elastic — temperature-dependent if table available
        lines.append("*ELASTIC")
        if self.youngs_modulus_table and temperature is None:
            # Write full table for CalculiX temperature interpolation
            for T, E in self.youngs_modulus_table:
                lines.append(f"{E:.3e}, {self.poisson_ratio}, {T:.1f}")
        elif temperature is not None and self.youngs_modulus_table:
            E_at_T = self.youngs_modulus_at(temperature)
            lines.append(f"{E_at_T:.3e}, {self.poisson_ratio}, {temperature:.1f}")
        else:
            lines.append(f"{self.youngs_modulus:.3e}, {self.poisson_ratio}")

        lines.append("*DENSITY")
        lines.append(f"{self.density}")
        lines.append("*EXPANSION")
        lines.append(f"{self.thermal_expansion}")

        # Conductivity — temperature-dependent if table available
        lines.append("*CONDUCTIVITY")
        if self.thermal_conductivity_table and temperature is None:
            for T, k in self.thermal_conductivity_table:
                lines.append(f"{k}, {T:.1f}")
        elif temperature is not None and self.thermal_conductivity_table:
            k_at_T = self.thermal_conductivity_at(temperature)
            lines.append(f"{k_at_T}, {temperature:.1f}")
        else:
            lines.append(f"{self.thermal_conductivity}")

        lines.append("*SPECIFIC HEAT")
        lines.append(f"{self.specific_heat}")

        return "\n".join(lines)

    def to_abaqus_format(self, temperature: float | None = None) -> str:
        """Format material for Abaqus input file (same as CalculiX)."""
        return self.to_calculix_format(temperature)


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
    # MMPDS / Special Metals data
    youngs_modulus_table=[
        (293, 200e9), (473, 190e9), (573, 184e9), (673, 176e9),
        (773, 167e9), (873, 156e9), (973, 140e9),
    ],
    yield_strength_table=[
        (293, 1035e6), (473, 980e6), (573, 950e6), (673, 910e6),
        (773, 860e6), (873, 740e6), (973, 580e6),
    ],
    thermal_conductivity_table=[
        (293, 11.4), (473, 14.1), (573, 15.9), (673, 17.8),
        (773, 19.6), (873, 21.5), (973, 23.6),
    ],
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
    # Haynes International data
    youngs_modulus_table=[
        (293, 205e9), (473, 196e9), (673, 184e9), (873, 170e9),
        (1073, 153e9), (1273, 130e9), (1473, 105e9),
    ],
    yield_strength_table=[
        (293, 360e6), (473, 310e6), (673, 280e6), (873, 260e6),
        (1073, 230e6), (1273, 170e6), (1473, 100e6),
    ],
    thermal_conductivity_table=[
        (293, 9.2), (473, 12.0), (673, 16.0), (873, 20.0),
        (1073, 23.5), (1273, 26.5), (1473, 29.0),
    ],
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
    # Cannon-Muskegon / open literature data for <001> orientation
    youngs_modulus_table=[
        (293, 130e9), (473, 126e9), (673, 120e9), (873, 110e9),
        (1073, 95e9), (1173, 85e9), (1273, 72e9), (1373, 58e9),
    ],
    yield_strength_table=[
        (293, 950e6), (473, 920e6), (673, 880e6), (873, 820e6),
        (1073, 680e6), (1173, 520e6), (1273, 380e6), (1373, 250e6),
    ],
    thermal_conductivity_table=[
        (293, 12.0), (473, 14.0), (673, 16.5), (873, 19.5),
        (1073, 22.5), (1273, 25.5), (1373, 27.0),
    ],
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
    youngs_modulus_table=[
        (293, 131e9), (473, 127e9), (673, 121e9), (873, 112e9),
        (1073, 97e9), (1173, 87e9), (1273, 74e9), (1383, 57e9),
    ],
    yield_strength_table=[
        (293, 960e6), (473, 930e6), (673, 890e6), (873, 830e6),
        (1073, 700e6), (1173, 550e6), (1273, 400e6), (1383, 260e6),
    ],
    thermal_conductivity_table=[
        (293, 11.5), (473, 13.5), (673, 16.0), (873, 19.5),
        (1073, 23.0), (1273, 26.0), (1383, 28.0),
    ],
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
    # MMPDS / ASM Ti data
    youngs_modulus_table=[
        (293, 113.8e9), (373, 110e9), (473, 105e9), (573, 98e9), (673, 90e9),
    ],
    yield_strength_table=[
        (293, 880e6), (373, 810e6), (473, 720e6), (573, 600e6), (673, 480e6),
    ],
    thermal_conductivity_table=[
        (293, 6.7), (373, 7.4), (473, 8.7), (573, 10.3), (673, 12.0),
    ],
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
    youngs_modulus_table=[
        (293, 232e9), (473, 222e9), (673, 210e9), (873, 195e9),
        (1073, 176e9), (1273, 150e9), (1363, 138e9),
    ],
    yield_strength_table=[
        (293, 455e6), (473, 380e6), (673, 330e6), (873, 300e6),
        (1073, 250e6), (1273, 170e6), (1363, 120e6),
    ],
    thermal_conductivity_table=[
        (293, 10.4), (473, 13.2), (673, 17.0), (873, 21.0),
        (1073, 24.5), (1273, 27.5), (1363, 29.0),
    ],
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


# =====================================================================
# Material Selection Advisor
# =====================================================================

# Engine component → material mapping based on temperature + stress regime
# Maps: component_type → list of (max_temp_K, recommended_material_keys)
# Ordered by temperature: lowest first. Advisor picks the first material
# whose max_service_temperature exceeds the operating temperature.

COMPONENT_MATERIAL_MAP: dict[str, list[tuple[float, str]]] = {
    "fan_blade": [
        (673, "ti_6al_4v"),          # Standard fan blade
        (813, "ti_6242"),            # High-temp fan
    ],
    "compressor_blade": [
        (673, "ti_6al_4v"),          # LP compressor
        (813, "ti_6242"),            # Mid compressor
        (873, "imi_834"),            # HP compressor (high-temp Ti)
        (973, "inconel_718"),        # Rear compressor stages
    ],
    "compressor_disk": [
        (673, "ti_6al_4v"),          # LP disk
        (723, "ti_6246"),            # Mid disk
        (973, "inconel_718"),        # HP disk
        (1023, "udimet_720"),        # Advanced HP disk
        (1003, "waspaloy"),          # Alternative HP disk
    ],
    "turbine_blade": [
        (1143, "inconel_713c"),      # Small engines, cast
        (1253, "mar_m_247"),         # Cast polycrystalline
        (1255, "rene_80"),           # Conventional cast
        (1373, "cmsx_4"),            # 1st gen single crystal
        (1383, "rene_n5"),           # 2nd gen single crystal (GE)
        (1393, "pwa_1484"),          # 2nd gen single crystal (P&W)
        (1588, "sic_sic_cmc"),       # Next-gen CMC
    ],
    "turbine_vane": [
        (1143, "rene_41"),           # Conventional
        (1253, "haynes_25"),         # Cobalt-based
        (1255, "rene_80"),           # Cast Ni
        (1373, "cmsx_4"),            # Single crystal
        (1588, "sic_sic_cmc"),       # CMC
    ],
    "turbine_disk": [
        (973, "inconel_718"),        # Standard
        (1003, "waspaloy"),          # High-temp
        (1023, "udimet_720"),        # Advanced
    ],
    "combustor_liner": [
        (1073, "inconel_625"),       # Conventional
        (1363, "haynes_188"),        # Cobalt-based
        (1473, "hastelloy_x"),       # High oxidation resistance
        (1473, "oxide_oxide_cmc"),   # Next-gen CMC
    ],
    "turbine_shroud": [
        (1253, "mar_m_247"),         # Conventional
        (1363, "haynes_188"),        # Cobalt
        (1588, "sic_sic_cmc"),       # CMC (GE LEAP)
    ],
    "shaft": [
        (623, "steel_17_4ph"),       # Low-temp shaft
        (673, "aisi_4340"),          # Standard shaft
        (723, "maraging_300"),       # High-strength shaft
        (1003, "waspaloy"),          # Hot-section shaft
    ],
    "casing": [
        (623, "steel_15_5ph"),       # LP casing
        (923, "incoloy_909"),        # Low-CTE matched casing
        (973, "inconel_718"),        # HP casing
    ],
    "nozzle": [
        (973, "inconel_718"),        # Conventional
        (1473, "hastelloy_x"),       # High-temp nozzle
        (1643, "c_103_niobium"),     # Rocket / hypersonic nozzle
    ],
    "structural": [
        (473, "al_7075"),            # Low-temp structure
        (623, "steel_17_4ph"),       # Moderate-temp structure
        (673, "ti_6al_4v"),          # Weight-critical structure
    ],
}

# Coating recommendations by component
COATING_MAP: dict[str, list[str]] = {
    "turbine_blade":   ["mcraly", "ysz_tbc"],
    "turbine_vane":    ["mcraly", "ysz_tbc"],
    "combustor_liner": ["mcraly"],
    "turbine_shroud":  ["ysz_tbc"],
    "nozzle":          ["mcraly"],
}


@dataclass
class MaterialRecommendation:
    """Result of material selection advisor."""

    component: str
    operating_temperature: float       # K
    primary_material: Material
    primary_key: str
    alternatives: list[tuple[str, Material]]  # (key, material)
    coatings: list[tuple[str, Material]]      # (key, material)
    warnings: list[str]


def recommend_material(
    component: str,
    operating_temperature: float,
    stress_mpa: float = 0.0,
) -> MaterialRecommendation:
    """Recommend materials for a turbomachinery component.

    Selects the best material based on component type, operating
    temperature, and (optionally) stress level.

    Args:
        component: Component type. One of: fan_blade, compressor_blade,
            compressor_disk, turbine_blade, turbine_vane, turbine_disk,
            combustor_liner, turbine_shroud, shaft, casing, nozzle, structural.
        operating_temperature: Gas path temperature at this component (K).
        stress_mpa: Expected stress level (MPa). Used for safety warnings.

    Returns:
        MaterialRecommendation with primary pick, alternatives, and coatings.

    Raises:
        ValueError: If component type is unknown.
    """
    comp = component.lower().replace(" ", "_").replace("-", "_")
    if comp not in COMPONENT_MATERIAL_MAP:
        available = ", ".join(sorted(COMPONENT_MATERIAL_MAP.keys()))
        raise ValueError(f"Unknown component '{component}'. Available: {available}")

    candidates = COMPONENT_MATERIAL_MAP[comp]
    warnings: list[str] = []

    # Find the first material whose max temp exceeds operating temp
    primary_key = candidates[-1][1]  # fallback to highest-temp option
    for max_temp, mat_key in candidates:
        if max_temp >= operating_temperature:
            primary_key = mat_key
            break
    else:
        warnings.append(
            f"Operating temperature {operating_temperature:.0f} K exceeds all "
            f"candidate materials for {comp}. Using highest-rated option."
        )

    primary = MATERIAL_DATABASE[primary_key]

    # Temperature margin check
    margin = primary.max_service_temperature - operating_temperature
    if margin < 50:
        warnings.append(
            f"Temperature margin only {margin:.0f} K — consider a higher-rated material."
        )

    # Stress check
    if stress_mpa > 0 and stress_mpa > primary.yield_strength / 1e6 * 0.67:
        safety = primary.yield_strength / 1e6 / stress_mpa
        warnings.append(
            f"Safety factor only {safety:.2f} (below typical 1.5 minimum for "
            f"{stress_mpa:.0f} MPa at yield={primary.yield_strength/1e6:.0f} MPa)."
        )

    # Collect alternatives (other candidates that could work)
    alternatives: list[tuple[str, Material]] = []
    for max_temp, mat_key in candidates:
        if mat_key != primary_key and max_temp >= operating_temperature:
            alternatives.append((mat_key, MATERIAL_DATABASE[mat_key]))

    # Coating recommendations
    coatings: list[tuple[str, Material]] = []
    if comp in COATING_MAP:
        for coat_key in COATING_MAP[comp]:
            coatings.append((coat_key, MATERIAL_DATABASE[coat_key]))

    return MaterialRecommendation(
        component=comp,
        operating_temperature=operating_temperature,
        primary_material=primary,
        primary_key=primary_key,
        alternatives=alternatives,
        coatings=coatings,
        warnings=warnings,
    )


def recommend_engine_materials(
    t_fan: float = 350.0,
    t_compressor: float = 750.0,
    t_combustor: float = 1400.0,
    t_turbine: float = 1350.0,
    t_nozzle: float = 1000.0,
) -> dict[str, MaterialRecommendation]:
    """Recommend materials for all major engine sections.

    Args:
        t_fan: Fan inlet temperature (K).
        t_compressor: HP compressor exit temperature (K).
        t_combustor: Combustor liner temperature (K).
        t_turbine: Turbine inlet temperature (K).
        t_nozzle: Nozzle temperature (K).

    Returns:
        Dict mapping component name to MaterialRecommendation.
    """
    return {
        "fan_blade": recommend_material("fan_blade", t_fan),
        "compressor_blade": recommend_material("compressor_blade", t_compressor),
        "compressor_disk": recommend_material("compressor_disk", t_compressor),
        "combustor_liner": recommend_material("combustor_liner", t_combustor),
        "turbine_blade": recommend_material("turbine_blade", t_turbine),
        "turbine_vane": recommend_material("turbine_vane", t_turbine),
        "turbine_disk": recommend_material("turbine_disk", min(t_turbine, 1000)),
        "turbine_shroud": recommend_material("turbine_shroud", t_turbine),
        "shaft": recommend_material("shaft", min(t_compressor, 700)),
        "nozzle": recommend_material("nozzle", t_nozzle),
    }
