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
from .off_design import (
    OffDesignResult,
    off_design_compressor,
)
from .compressor_map import (
    SpeedLine,
    CompressorMap,
    generate_compressor_map,
)
from .centrifugal import (
    CentrifugalResult,
    centrifugal_compressor,
    wiesner_slip_factor,
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
    "OffDesignResult",
    "off_design_compressor",
    "SpeedLine",
    "CompressorMap",
    "generate_compressor_map",
    "CentrifugalResult",
    "centrifugal_compressor",
    "wiesner_slip_factor",
]
