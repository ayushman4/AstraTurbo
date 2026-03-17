"""Design variable parameterization for optimization.

Maps optimizer variables (normalized [0, 1]) to physical blade design
parameters (angles, thicknesses, chord lengths, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class DesignVariable:
    """A single design variable with bounds."""

    name: str
    lower_bound: float
    upper_bound: float
    initial_value: float | None = None

    def normalize(self, value: float) -> float:
        """Map physical value to [0, 1]."""
        span = self.upper_bound - self.lower_bound
        if span < 1e-15:
            return 0.5
        return (value - self.lower_bound) / span

    def denormalize(self, normalized: float) -> float:
        """Map [0, 1] to physical value."""
        return self.lower_bound + normalized * (self.upper_bound - self.lower_bound)


@dataclass
class DesignSpace:
    """Collection of design variables defining the optimization space."""

    variables: list[DesignVariable] = field(default_factory=list)

    @property
    def n_vars(self) -> int:
        return len(self.variables)

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.array([v.lower_bound for v in self.variables])

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        return np.array([v.upper_bound for v in self.variables])

    @property
    def initial_values(self) -> NDArray[np.float64]:
        return np.array([
            v.initial_value if v.initial_value is not None
            else (v.lower_bound + v.upper_bound) / 2
            for v in self.variables
        ])

    def add(self, name: str, lb: float, ub: float, initial: float | None = None) -> None:
        """Add a design variable."""
        self.variables.append(DesignVariable(name, lb, ub, initial))

    def decode(self, x: NDArray[np.float64]) -> dict[str, float]:
        """Convert an optimizer vector to named parameters."""
        return {v.name: float(x[i]) for i, v in enumerate(self.variables)}


def create_blade_design_space(
    n_profiles: int = 3,
    include_stagger: bool = True,
    include_chord: bool = True,
    include_thickness: bool = True,
    include_camber: bool = True,
) -> DesignSpace:
    """Create a standard design space for blade optimization.

    Args:
        n_profiles: Number of span profiles.
        include_stagger: Include stagger angle variables.
        include_chord: Include chord length variables.
        include_thickness: Include max thickness variables.
        include_camber: Include max camber variables.

    Returns:
        DesignSpace with turbomachinery design variables.
    """
    ds = DesignSpace()

    for i in range(n_profiles):
        suffix = f"_span{i}"
        if include_stagger:
            ds.add(f"stagger{suffix}", lb=-45.0, ub=45.0, initial=0.0)
        if include_chord:
            ds.add(f"chord{suffix}", lb=0.01, ub=0.2, initial=0.05)
        if include_thickness:
            ds.add(f"max_thickness{suffix}", lb=0.03, ub=0.25, initial=0.10)
        if include_camber:
            ds.add(f"max_camber{suffix}", lb=0.0, ub=0.15, initial=0.05)

    return ds
