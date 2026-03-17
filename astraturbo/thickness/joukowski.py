"""Joukowski thickness distribution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import BoundedNumericProperty, memoize
from .thickness import ThicknessDistribution


class JoukowskiThickness(ThicknessDistribution):
    """Joukowski thickness distribution.

    Parameters:
        max_thickness: Maximum thickness as fraction of chord.
    """

    max_thickness = BoundedNumericProperty(lb=0.001, ub=1.0, default=0.1)

    def __init__(self, max_thickness: float = 0.1) -> None:
        super().__init__()
        self.max_thickness = max_thickness

    @classmethod
    def default(cls) -> JoukowskiThickness:
        return cls(max_thickness=0.1)

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        x_rev = x[::-1]  # (1 - x) reversed
        y = self.max_thickness * 1.5396007178 * x_rev * np.sqrt(x_rev * x)
        return np.column_stack((x, y))
