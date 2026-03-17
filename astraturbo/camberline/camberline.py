"""Abstract base class for camber lines.

A camber line is the mean line of an airfoil profile, defined in
normalized coordinates [0, 1] along the chord.
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


class CamberLine(Node, Drawable):
    """Abstract base for all camber line types.

    Subclasses must implement:
      - as_array() -> (N, 2) array of (x, y) camber line points
      - get_derivations() -> (N,) array of dy/dx at each sample point

    Properties:
        sample_rate: Number of points to sample along the chord.
        distribution: Point distribution strategy (Chebyshev, Linear, etc.)
    """

    sample_rate = SharedBoundedNumericProperty(lb=10, ub=9999, default=200)
    distribution = SharedProperty(default=Chebyshev())

    def __init__(self) -> None:
        super().__init__()
        self.name = "Camber Line"

    def get_derivations(self) -> NDArray[np.float64]:
        """Return dy/dx at each sample point as a 1D array of shape (N,)."""
        raise NotImplementedError

    def as_array(self) -> NDArray[np.float64]:
        """Return camber line as (N, 2) array of (x, y) points in [0, 1]."""
        raise NotImplementedError

    def get_plot_data_2d(self) -> NDArray[np.float64] | None:
        return self.as_array()
