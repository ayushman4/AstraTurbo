"""Polynomial camber lines (quadratic, cubic, quartic).

Polynomial camber lines use Hermite boundary conditions at the leading
and trailing edges to define smooth curves.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..foundation import BoundedNumericProperty, memoize
from .camberline import CamberLine


class QuadraticPolynomial(CamberLine):
    """Quadratic (degree 2) polynomial camber line.

    Defined by the inflow angle at the leading edge.
    The outflow is automatically zero-slope at x=1.
    """

    angle_of_inflow = BoundedNumericProperty(lb=0, ub=180)

    def __init__(self, angle_of_inflow: float = 100) -> None:
        super().__init__()
        self.angle_of_inflow = angle_of_inflow

    @classmethod
    def default(cls) -> QuadraticPolynomial:
        return cls(angle_of_inflow=100)

    def _get_coefficients(self) -> tuple[float, float]:
        a1 = math.tan(np.deg2rad(self.angle_of_inflow) - math.pi / 2.0)
        return a1, -a1

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        a1, a2 = self._get_coefficients()
        return x * (2 * a2) + a1

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        a1, a2 = self._get_coefficients()
        y = x * (x * a2 + a1)
        return np.column_stack((x, y))


class CubicPolynomial(CamberLine):
    """Cubic (degree 3) polynomial camber line.

    Defined by inflow and outflow angles at leading and trailing edges.
    """

    angle_of_inflow = BoundedNumericProperty(lb=0, ub=180)
    angle_of_outflow = BoundedNumericProperty(lb=0, ub=180)

    def __init__(
        self, angle_of_inflow: float = 100, angle_of_outflow: float = 90
    ) -> None:
        super().__init__()
        self.angle_of_inflow = angle_of_inflow
        self.angle_of_outflow = angle_of_outflow

    @classmethod
    def default(cls) -> CubicPolynomial:
        return cls(angle_of_inflow=100, angle_of_outflow=90)

    def _get_coefficients(self) -> tuple[float, float, float]:
        slope_inlet = math.tan(np.deg2rad(self.angle_of_inflow) - np.pi / 2.0)
        slope_outlet = math.tan(np.pi / 2.0 - np.deg2rad(self.angle_of_outflow))
        # Constraints: y(0)=0, y(1)=0, y'(0)=slope_inlet, y'(1)=slope_outlet
        # For y = a3*x^3 + a2*x^2 + a1*x:
        #   a1 = slope_inlet
        #   a3 = slope_inlet + slope_outlet  (from y(1)=0 and y'(1)=slope_outlet)
        #   a2 = -a1 - a3
        a1 = slope_inlet
        a3 = slope_inlet + slope_outlet
        a2 = -a1 - a3
        return a1, a2, a3

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        a1, a2, a3 = self._get_coefficients()
        return x * (x * 3 * a3 + 2 * a2) + a1

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        a1, a2, a3 = self._get_coefficients()
        y = x * (x * (x * a3 + a2) + a1)
        return np.column_stack((x, y))


class QuarticPolynomial(CamberLine):
    """Quartic (degree 4) polynomial camber line.

    Defined by inflow angle, outflow angle, and maximum camber position.
    Provides more control over the camber line shape than cubic.
    """

    angle_of_inflow = BoundedNumericProperty(lb=0, ub=180)
    angle_of_outflow = BoundedNumericProperty(lb=0, ub=180)
    max_camber_position = BoundedNumericProperty(lb=0.01, ub=0.99)

    def __init__(
        self,
        angle_of_inflow: float = 100,
        angle_of_outflow: float = 90,
        max_camber_position: float = 0.4,
    ) -> None:
        super().__init__()
        self.angle_of_inflow = angle_of_inflow
        self.angle_of_outflow = angle_of_outflow
        self.max_camber_position = max_camber_position

    @classmethod
    def default(cls) -> QuarticPolynomial:
        return cls(angle_of_inflow=100, angle_of_outflow=90, max_camber_position=0.4)

    def _get_coefficients(self) -> tuple[float, float, float, float]:
        slope_inlet = math.tan(np.deg2rad(self.angle_of_inflow) - np.pi / 2.0)
        slope_outlet = math.tan(np.pi / 2.0 - np.deg2rad(self.angle_of_outflow))
        p = self.max_camber_position
        a4 = (
            ((p * (4 - 3 * p) - 1) * slope_inlet + p * (3 * p - 2) * slope_outlet)
            / (2 * p * (2 * p) * (p - 1))
        )
        a3 = slope_inlet - slope_outlet - 2 * a4
        a2 = -slope_inlet - a3 - a4
        a1 = slope_inlet
        return a1, a2, a3, a4

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        a1, a2, a3, a4 = self._get_coefficients()
        return x * (x * (4 * a4 * x + 3 * a3) + 2 * a2) + a1

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        a1, a2, a3, a4 = self._get_coefficients()
        y = x * (x * (x * (a4 * x + a3) + a2) + a1)
        return np.column_stack((x, y))
