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
from .turbine import (
    TurbineStageResult,
    TurbineResult,
    meanline_turbine,
    meanline_to_turbine_blade_parameters,
)
from .turbine_off_design import (
    TurbineOffDesignResult,
    turbine_off_design,
    TurbineSpeedLine,
    TurbineMap,
    generate_turbine_map,
)
from .engine_cycle import (
    InletResult,
    CombustorResult,
    AfterburnerResult,
    NozzleResult,
    EngineCycleResult,
    standard_atmosphere,
    inlet_model,
    afterburner_model,
    combustor_model,
    nozzle_model,
    engine_cycle,
)
from .electric_motor import (
    ElectricMotorResult,
    electric_motor,
)
from .propeller import (
    PropellerResult,
    propeller_design,
)
from .pump import (
    FLUIDS,
    PumpResult,
    centrifugal_pump,
)
from .turbopump import (
    TurbopumpResult,
    turbopump,
)
from .cooling import (
    COOLING_PHI,
    CoolingRowResult,
    CoolingResult,
    cooling_flow,
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
    "TurbineStageResult",
    "TurbineResult",
    "meanline_turbine",
    "meanline_to_turbine_blade_parameters",
    "TurbineOffDesignResult",
    "turbine_off_design",
    "TurbineSpeedLine",
    "TurbineMap",
    "generate_turbine_map",
    "InletResult",
    "CombustorResult",
    "AfterburnerResult",
    "NozzleResult",
    "EngineCycleResult",
    "standard_atmosphere",
    "inlet_model",
    "afterburner_model",
    "combustor_model",
    "nozzle_model",
    "engine_cycle",
    "ElectricMotorResult",
    "electric_motor",
    "PropellerResult",
    "propeller_design",
    "FLUIDS",
    "PumpResult",
    "centrifugal_pump",
    "TurbopumpResult",
    "turbopump",
    "COOLING_PHI",
    "CoolingRowResult",
    "CoolingResult",
    "cooling_flow",
]
