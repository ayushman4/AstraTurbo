"""NACA 4-digit thickness distribution.

Standard NACA 4-digit thickness formula:
    y = t * (A*sqrt(x) + B*x + C*x^2 + D*x^3 + E*x^4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import BoundedNumericProperty, memoize
from .thickness import ThicknessDistribution


class NACA4Digit(ThicknessDistribution):
    """NACA 4-digit series thickness distribution.

    Parameters:
        max_thickness: Maximum thickness as fraction of chord (e.g. 0.12).
    """

    max_thickness = BoundedNumericProperty(lb=0.001, ub=1.0, default=0.12)

    def __init__(self, max_thickness: float = 0.12) -> None:
        super().__init__()
        self.max_thickness = max_thickness

    @classmethod
    def default(cls) -> NACA4Digit:
        return cls(max_thickness=0.12)

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        # Standard NACA 4-digit coefficients (open trailing edge)
        A, B, C, D, E = 1.4845, -0.63, -1.758, 1.4215, -0.5075
        t = self.max_thickness
        x = self.distribution(self.sample_rate)
        y = t * (A * np.sqrt(x) + x * (B + x * (C + x * (D + x * E))))
        return np.column_stack((x, y))
