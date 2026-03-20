"""NACA 65-series thickness distribution.

Used for compressor blade profiles.

Formula:
    y = Dmax * (1-x) * (A*sqrt(x) + B*x + C*x^2 + D*x^3) / (1 - E*x)

where A=1.0675, B=-0.2758, C=2.4478, D=-2.8385, E=0.176
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import BoundedNumericProperty, memoize
from .thickness import ThicknessDistribution


class NACA65Series(ThicknessDistribution):
    """NACA 65-series thickness distribution.

    Commonly used for compressor blades. The trailing edge is sharper
    than the NACA 4-digit series.

    Parameters:
        max_thickness: Maximum thickness as fraction of chord.
    """

    max_thickness = BoundedNumericProperty(lb=0.001, ub=1.0, default=0.1)

    def __init__(self, max_thickness: float = 0.1) -> None:
        super().__init__()
        self.max_thickness = max_thickness

    @classmethod
    def default(cls) -> NACA65Series:
        return cls(max_thickness=0.1)

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        A, B, C, D, E = 1.0675, -0.2758, 2.4478, -2.8385, 0.176
        t = self.max_thickness
        x = self.distribution(self.sample_rate)
        e = 1.0 - E * x
        y = t * (1.0 - x) * (A * np.sqrt(x) + x * (B + x * (C + x * D))) / e
        return np.column_stack((x, y))
