"""NACA 2-digit camber line.

Standard NACA 2-digit camber line definition with max camber and
max camber position parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import BoundedNumericProperty, NumericProperty, memoize
from .camberline import CamberLine


class NACA2Digit(CamberLine):
    """NACA 2-digit series camber line.

    Parameters:
        max_camber: Maximum camber as fraction of chord.
        max_camber_position: Chordwise position of maximum camber (0-1).
    """

    max_camber = NumericProperty(default=0.02)
    max_camber_position = BoundedNumericProperty(lb=0.01, ub=0.99, default=0.4)

    def __init__(
        self, max_camber: float = 0.02, max_camber_position: float = 0.4
    ) -> None:
        super().__init__()
        self.max_camber = max_camber
        self.max_camber_position = max_camber_position

    @classmethod
    def default(cls) -> NACA2Digit:
        return cls(max_camber=0.02, max_camber_position=0.4)

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        p = self.max_camber_position
        m = self.max_camber
        x = self.distribution(self.sample_rate)
        dydx = np.empty_like(x)
        front = x <= p
        back = ~front
        if np.any(front):
            dydx[front] = 2 * m / p**2 * (p - x[front])
        if np.any(back):
            dydx[back] = 2 * m / (1 - p) ** 2 * (p - x[back])
        return dydx

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        p = self.max_camber_position
        m = self.max_camber
        x = self.distribution(self.sample_rate)
        y = np.empty_like(x)
        front = x <= p
        back = ~front
        if np.any(front):
            xf = x[front]
            y[front] = m / p**2 * (xf * (2 * p - xf))
        if np.any(back):
            xb = x[back]
            y[back] = m / (1 - p) ** 2 * (1 - 2 * p + xb * (2 * p - xb))
        return np.column_stack((x, y))
