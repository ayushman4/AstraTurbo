"""Abstract base class for thickness distributions.

A thickness distribution defines the half-thickness of an airfoil
at each chordwise position.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..baseclass import Node, Drawable
from ..distribution import Chebyshev
from ..foundation import (
    BoundedNumericProperty,
    SharedBoundedNumericProperty,
    SharedProperty,
    memoize,
)


class ThicknessDistribution(Node, Drawable):
    """Abstract base for all thickness distribution types.

    Subclasses must implement:
      - as_array() -> (N, 2) array of (x, half_thickness) points

    Properties:
        sample_rate: Number of sample points along the chord.
        distribution: Point distribution strategy.
        max_thickness: Maximum thickness as fraction of chord.
    """

    sample_rate = SharedBoundedNumericProperty(lb=10, ub=9999, default=200)
    distribution = SharedProperty(default=Chebyshev())
    max_thickness = BoundedNumericProperty(lb=0.001, ub=1.0, default=0.1)

    def __init__(self) -> None:
        super().__init__()
        self.name = "Thickness Distribution"

    def as_array(self) -> NDArray[np.float64]:
        """Return (N, 2) array of (x, half_thickness) in normalized coords."""
        raise NotImplementedError

    def get_plot_data_2d(self) -> NDArray[np.float64] | None:
        return self.as_array()
