"""Objective functions for turbomachinery optimization.

Defines standard objective and constraint functions used in
blade design optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ObjectiveResult:
    """Result of evaluating an objective function."""

    objectives: NDArray[np.float64]     # (n_obj,) array
    constraints: NDArray[np.float64]    # (n_con,) array, <=0 is feasible
    feasible: bool = True


def efficiency_objective(
    pressure_ratio: float,
    mass_flow: float,
    shaft_power: float,
    target_pressure_ratio: float = 1.5,
) -> ObjectiveResult:
    """Compute isentropic efficiency objective.

    Minimize negative efficiency (pymoo minimizes).

    Args:
        pressure_ratio: Total pressure ratio achieved.
        mass_flow: Mass flow rate (kg/s).
        shaft_power: Shaft power input (W).
        target_pressure_ratio: Design target.

    Returns:
        ObjectiveResult with [-efficiency] and pressure ratio constraint.
    """
    gamma = 1.4  # Air
    cp = 1005.0  # J/(kg*K)
    t_inlet = 288.15  # K

    # Isentropic work
    isentropic_work = cp * t_inlet * (pressure_ratio ** ((gamma - 1) / gamma) - 1)
    actual_work = shaft_power / mass_flow if mass_flow > 0 else 1e10

    efficiency = isentropic_work / actual_work if actual_work > 0 else 0
    efficiency = min(max(efficiency, 0), 1)

    # Constraint: pressure ratio must meet target
    pr_constraint = target_pressure_ratio - pressure_ratio  # <=0 is feasible

    return ObjectiveResult(
        objectives=np.array([-efficiency]),  # Minimize negative efficiency
        constraints=np.array([pr_constraint]),
        feasible=(pr_constraint <= 0),
    )


def multi_objective(
    efficiency: float,
    stall_margin: float,
    weight: float = 1.0,
) -> ObjectiveResult:
    """Multi-objective: maximize efficiency and stall margin.

    Args:
        efficiency: Isentropic efficiency (0-1).
        stall_margin: Stall margin (0-1).
        weight: Weight factor for mechanical stress constraint.

    Returns:
        ObjectiveResult with [-efficiency, -stall_margin].
    """
    return ObjectiveResult(
        objectives=np.array([-efficiency, -stall_margin]),
        constraints=np.array([]),
        feasible=True,
    )
