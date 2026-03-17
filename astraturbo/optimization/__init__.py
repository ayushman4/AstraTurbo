"""Optimization module for AstraTurbo.

Provides design optimization for turbomachinery blades using
evolutionary algorithms (pymoo) or scipy fallback.
"""

from .parameterization import DesignSpace, DesignVariable, create_blade_design_space
from .objectives import ObjectiveResult, efficiency_objective, multi_objective
from .optimizer import Optimizer, OptimizationConfig, OptimizationResult, run_doe

__all__ = [
    "DesignSpace",
    "DesignVariable",
    "create_blade_design_space",
    "ObjectiveResult",
    "efficiency_objective",
    "multi_objective",
    "Optimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "run_doe",
]
