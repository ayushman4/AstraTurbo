"""Circular arc camber line.

The camber line is a circular arc defined by the angle of inflow.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..foundation import BoundedNumericProperty, memoize
from .camberline import CamberLine


class CircularArc(CamberLine):
    """Circular arc camber line defined by inflow angle.

    Parameters:
        angle_of_inflow: Flow angle at the leading edge in degrees (0-180).
            Values < 90 produce upward camber, > 90 produce downward camber.
    """

    angle_of_inflow = BoundedNumericProperty(lb=0, ub=180)

    def __init__(self, angle_of_inflow: float = 100) -> None:
        super().__init__()
        self.angle_of_inflow = angle_of_inflow

    @classmethod
    def default(cls) -> CircularArc:
        return cls(angle_of_inflow=100)

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        chi = self.angle_of_inflow
        sign = 1 if chi < 90 else -1
        r2 = (0.5 / math.cos(np.deg2rad(chi))) ** 2
        x = self.distribution(self.sample_rate) - 0.5
        return sign * x / np.sqrt(r2 - x**2)

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        chi = self.angle_of_inflow
        sign = -1 if chi < 90 else 1
        r2 = (0.5 / math.cos(np.deg2rad(chi))) ** 2
        x = self.distribution(self.sample_rate)
        y = sign * (np.sqrt(r2 - (x - 0.5) ** 2) - np.sqrt(r2 - 0.25))
        return np.column_stack((x, y))
