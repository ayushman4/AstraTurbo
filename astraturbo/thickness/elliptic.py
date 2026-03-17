"""Elliptic thickness distribution.

Simple semi-ellipse: y = Dmax/2 * sin(arccos(2x - 1))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import BoundedNumericProperty, memoize
from .thickness import ThicknessDistribution


class Elliptic(ThicknessDistribution):
    """Elliptic thickness distribution.

    Produces a symmetric elliptical profile with maximum thickness at mid-chord.

    Parameters:
        max_thickness: Maximum thickness as fraction of chord.
    """

    max_thickness = BoundedNumericProperty(lb=0.001, ub=1.0, default=0.1)

    def __init__(self, max_thickness: float = 0.1) -> None:
        super().__init__()
        self.max_thickness = max_thickness

    @classmethod
    def default(cls) -> Elliptic:
        return cls(max_thickness=0.1)

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        # Semi-ellipse: y = (Dmax/2) * sin(arccos(2x-1))
        arg = np.clip(2.0 * x - 1.0, -1.0, 1.0)
        y = (self.max_thickness / 2.0) * np.sin(np.arccos(arg))
        return np.column_stack((x, y))
