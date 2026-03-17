"""Abstract base for 2D airfoil profiles."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..baseclass import Node, Drawable
from ..distribution import Chebyshev
from ..foundation import (
    ChildProperty,
    SharedBoundedNumericProperty,
    SharedProperty,
    memoize,
)


class Profile(Node, Drawable):
    """Abstract base for 2D airfoil profiles.

    A profile combines a camber line with a thickness distribution to
    produce a closed 2D airfoil shape.

    Properties:
        camber_line: The camber line child object.
        sample_rate: Number of sample points (shared with children).
        distribution: Point distribution strategy (shared with children).
    """

    camber_line = ChildProperty(child_index=0)
    sample_rate = SharedBoundedNumericProperty(lb=10, ub=9999, default=200)
    distribution = SharedProperty(default=Chebyshev())

    def __init__(self, camber_line=None) -> None:
        super().__init__()
        if camber_line is not None:
            self.camber_line = camber_line
        self.name = "Profile"

    @property
    def angle_of_inflow(self) -> float:
        """Return the flow angle at the leading edge in radians."""
        derivations = self.camber_line.get_derivations()
        return math.atan(derivations[0]) + math.pi * 0.5

    @property
    def angle_of_outflow(self) -> float:
        """Return the flow angle at the trailing edge in radians."""
        derivations = self.camber_line.get_derivations()
        return math.atan(derivations[-1]) + math.pi * 0.5

    @property
    def centroid(self) -> NDArray[np.float64]:
        """Return the centroid [x, y] of the closed profile."""
        area = 0.0
        cx, cy = 0.0, 0.0
        pts = self.as_array()
        x, y = pts[:, 0], pts[:, 1]
        n = len(x)
        for i in range(n - 1):
            seg = x[i] * y[i + 1] - x[i + 1] * y[i]
            cx += (x[i] + x[i + 1]) * seg
            cy += (y[i] + y[i + 1]) * seg
            area += seg
        denom = 3.0 * abs(area)
        if denom < 1e-15:
            return np.array([0.5, 0.0])
        return np.array([cx / denom, cy / denom])

    def as_array(self) -> NDArray[np.float64]:
        """Return closed profile as (2N-1, 2) array of (x, y) points."""
        raise NotImplementedError

    def upper_surface(self) -> NDArray[np.float64]:
        """Return upper (suction) surface points as (N, 2) array."""
        raise NotImplementedError

    def lower_surface(self) -> NDArray[np.float64]:
        """Return lower (pressure) surface points as (N, 2) array."""
        raise NotImplementedError

    def get_plot_data_2d(self) -> NDArray[np.float64] | None:
        return self.as_array()
