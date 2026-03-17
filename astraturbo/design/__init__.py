"""Aerodynamic design module for AstraTurbo.

Provides the calculation layer between cycle requirements and blade geometry:
  - velocity_triangle: Inlet/outlet velocity diagrams (U, C, W, alpha, beta)
  - meanline: Stage-by-stage 1D analysis (pressure ratio, efficiency, blade angles)

Usage:
    from astraturbo.design import meanline_compressor, meanline_to_blade_parameters

    result = meanline_compressor(
        overall_pressure_ratio=4.0, mass_flow=20.0,
        rpm=12000, r_hub=0.15, r_tip=0.30,
    )
    print(result.summary())
    blade_params = meanline_to_blade_parameters(result)
"""

from .velocity_triangle import (
    VelocityTriangle,
    BladeRowTriangles,
    compute_triangle_from_angles,
    compute_triangle_from_beta,
)
from .meanline import (
    GasProperties,
    StationConditions,
    StageResult,
    MeanlineResult,
    meanline_compressor_stage,
    meanline_compressor,
    meanline_to_blade_parameters,
)

__all__ = [
    "VelocityTriangle",
    "BladeRowTriangles",
    "compute_triangle_from_angles",
    "compute_triangle_from_beta",
    "GasProperties",
    "StationConditions",
    "StageResult",
    "MeanlineResult",
    "meanline_compressor_stage",
    "meanline_compressor",
    "meanline_to_blade_parameters",
]
