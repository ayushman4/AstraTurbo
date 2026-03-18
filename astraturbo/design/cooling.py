"""Turbine blade cooling flow estimation.

Models the cooling air requirement for turbine blades using the
Holland-Thake effectiveness model.  Supports convection, film,
and transpiration cooling types.

Physics:
    Overall cooling effectiveness  ε = (T_gas - T_blade) / (T_gas - T_coolant)
    Required cooling mass-flow ratio  mc/mg = ε / (φ × (1 - ε))
    where φ = cooling method effectiveness constant:
        convection  = 0.2  (least effective per unit coolant, highest demand)
        film        = 0.4  (moderate)
        transpiration = 1.0 (most effective per unit coolant, lowest demand)

    Higher φ means more effective heat removal per unit of coolant mass flow,
    therefore less total coolant is needed.

References:
    Holland, M.J. & Thake, T.F., "Rotor Blade Cooling in High Pressure Turbines",
    J. Aircraft, Vol. 17, No. 6, 1980, pp. 412-418.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Cooling method effectiveness constants ──────────

COOLING_PHI: dict[str, float] = {
    "convection": 0.2,
    "film": 0.4,
    "transpiration": 1.0,
}


# ── Result Dataclasses ──────────────────────────────


@dataclass
class CoolingRowResult:
    """Cooling result for a single blade row."""
    row_index: int = 0
    cooling_effectiveness: float = 0.0
    coolant_fraction: float = 0.0          # mc / mg for this row
    coolant_mass_flow: float = 0.0         # kg/s
    T_blade: float = 0.0                   # K (metal temperature)


@dataclass
class CoolingResult:
    """Complete cooling analysis result."""

    T_gas: float = 0.0
    T_coolant: float = 0.0
    T_blade_max: float = 1300.0
    cooling_type: str = "film"
    phi: float = 0.4
    n_cooled_rows: int = 2
    mass_flow_gas: float = 20.0

    # Per-row results
    rows: list[CoolingRowResult] = field(default_factory=list)

    # Totals
    total_coolant_fraction: float = 0.0    # sum of mc/mg across rows
    total_coolant_flow: float = 0.0        # kg/s
    overall_effectiveness: float = 0.0
    material_name: Optional[str] = None

    def summary(self) -> str:
        """Human-readable cooling summary."""
        lines = [
            "Turbine Blade Cooling Analysis",
            "=" * 50,
            f"  Gas Temperature:      {self.T_gas:.0f} K",
            f"  Coolant Temperature:  {self.T_coolant:.0f} K",
            f"  Max Blade Temp:       {self.T_blade_max:.0f} K",
            f"  Cooling Type:         {self.cooling_type} (φ={self.phi:.2f})",
            f"  Cooled Rows:          {self.n_cooled_rows}",
            f"  Gas Mass Flow:        {self.mass_flow_gas:.2f} kg/s",
        ]
        if self.material_name:
            lines.append(f"  Material:             {self.material_name}")
        lines.append("")
        lines.append("  Per-Row Breakdown:")
        lines.append(f"    {'Row':<6s} {'ε':>8s} {'mc/mg':>10s} {'mc (kg/s)':>12s} {'T_blade (K)':>12s}")
        lines.append(f"    {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
        for r in self.rows:
            lines.append(
                f"    {r.row_index:<6d} {r.cooling_effectiveness:>8.4f} "
                f"{r.coolant_fraction:>10.4f} {r.coolant_mass_flow:>12.4f} "
                f"{r.T_blade:>12.1f}"
            )
        lines.append("")
        lines.append(f"  Total coolant fraction: {self.total_coolant_fraction:.4f}")
        lines.append(f"  Total coolant flow:     {self.total_coolant_flow:.4f} kg/s")
        lines.append(f"  Overall effectiveness:  {self.overall_effectiveness:.4f}")

        return "\n".join(lines)


# ── Solver ──────────────────────────────────────────


def cooling_flow(
    T_gas: float,
    T_coolant: float,
    T_blade_max: float = 1300.0,
    cooling_type: str = "film",
    n_cooled_rows: int = 2,
    mass_flow_gas: float = 20.0,
    material_name: Optional[str] = None,
) -> CoolingResult:
    """Estimate turbine blade cooling air requirements.

    Uses the Holland-Thake effectiveness model to compute the coolant
    mass flow fraction required to keep blade metal temperature at or
    below T_blade_max.

    Args:
        T_gas: Hot gas total temperature at turbine entry (K).
        T_coolant: Coolant air temperature, typically compressor bleed (K).
        T_blade_max: Maximum allowable blade metal temperature (K).
            If material_name is given and a lookup succeeds, it overrides this.
        cooling_type: "convection", "film", or "transpiration".
        n_cooled_rows: Number of cooled blade rows (NGV + rotor pairs).
        mass_flow_gas: Hot gas mass flow rate (kg/s).
        material_name: Optional material name for T_blade_max lookup.

    Returns:
        CoolingResult with per-row and total cooling requirements.

    Raises:
        ValueError: On invalid inputs.
    """
    # Validation
    if T_gas <= 0:
        raise ValueError(f"T_gas must be positive, got {T_gas}")
    if T_coolant <= 0:
        raise ValueError(f"T_coolant must be positive, got {T_coolant}")
    if T_coolant >= T_gas:
        raise ValueError(f"T_coolant ({T_coolant}) must be less than T_gas ({T_gas})")
    if cooling_type not in COOLING_PHI:
        raise ValueError(
            f"cooling_type must be one of {list(COOLING_PHI.keys())}, got '{cooling_type}'"
        )
    if n_cooled_rows < 1:
        raise ValueError(f"n_cooled_rows must be >= 1, got {n_cooled_rows}")
    if mass_flow_gas <= 0:
        raise ValueError(f"mass_flow_gas must be positive, got {mass_flow_gas}")

    # Material lookup (optional)
    if material_name is not None:
        try:
            from astraturbo.materials import get_material
            mat = get_material(material_name)
            T_blade_max = mat.max_service_temperature
        except (ImportError, KeyError):
            pass  # use provided T_blade_max

    phi = COOLING_PHI[cooling_type]

    # Overall effectiveness required
    epsilon = (T_gas - T_blade_max) / (T_gas - T_coolant)
    epsilon = max(min(epsilon, 0.99), 0.01)  # clamp to physical range

    # Build per-row results
    # Each successive row sees slightly lower gas temperature
    # Simple model: gas temperature drops proportionally as coolant mixes
    rows: list[CoolingRowResult] = []
    total_coolant_fraction = 0.0
    T_gas_local = T_gas

    for i in range(n_cooled_rows):
        # Local effectiveness for this row
        eps_row = (T_gas_local - T_blade_max) / (T_gas_local - T_coolant)
        eps_row = max(min(eps_row, 0.99), 0.01)

        # Coolant fraction for this row (Holland-Thake)
        mc_mg = eps_row / (phi * (1.0 - eps_row))
        mc_mg = max(mc_mg, 0.0)

        coolant_flow = mc_mg * mass_flow_gas
        T_blade = T_gas_local - eps_row * (T_gas_local - T_coolant)

        rows.append(CoolingRowResult(
            row_index=i + 1,
            cooling_effectiveness=eps_row,
            coolant_fraction=mc_mg,
            coolant_mass_flow=coolant_flow,
            T_blade=T_blade,
        ))

        total_coolant_fraction += mc_mg

        # Gas temperature drops as coolant mixes into mainstream
        # Simple mixing model: T_gas_new = (mg*T_gas + mc*T_coolant) / (mg + mc)
        T_gas_local = (
            (mass_flow_gas * T_gas_local + coolant_flow * T_coolant)
            / (mass_flow_gas + coolant_flow)
        )

    total_coolant_flow = total_coolant_fraction * mass_flow_gas

    return CoolingResult(
        T_gas=T_gas,
        T_coolant=T_coolant,
        T_blade_max=T_blade_max,
        cooling_type=cooling_type,
        phi=phi,
        n_cooled_rows=n_cooled_rows,
        mass_flow_gas=mass_flow_gas,
        rows=rows,
        total_coolant_fraction=total_coolant_fraction,
        total_coolant_flow=total_coolant_flow,
        overall_effectiveness=epsilon,
        material_name=material_name,
    )
