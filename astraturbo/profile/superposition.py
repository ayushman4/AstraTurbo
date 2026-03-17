"""Superposition profile: camber line + thickness distribution.

Creates a closed 2D airfoil by superimposing thickness perpendicular to the
camber line:
    Upper surface: (x - yt*sin(theta), yc + yt*cos(theta))
    Lower surface: (x + yt*sin(theta), yc - yt*cos(theta))
where theta = arctan(dy_camber/dx) at each point.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..camberline import CamberLine, NACA2Digit
from ..foundation import ChildProperty, memoize
from ..thickness import ThicknessDistribution, NACA4Digit
from .profile import Profile


class Superposition(Profile):
    """Superposition profile: camber line + perpendicular thickness.

    Parameters:
        camber_line: CamberLine object defining the mean line.
        thickness_distribution: ThicknessDistribution object defining thickness.
    """

    thickness_distribution = ChildProperty(child_index=1)

    def __init__(
        self,
        camber_line: CamberLine | None = None,
        thickness_distribution: ThicknessDistribution | None = None,
    ) -> None:
        super().__init__(camber_line)
        if thickness_distribution is not None:
            self.thickness_distribution = thickness_distribution
        self.name = "Superposition Profile"

    @classmethod
    def default(cls) -> Superposition:
        return cls(
            camber_line=NACA2Digit.default(),
            thickness_distribution=NACA4Digit.default(),
        )

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        """Return closed profile as (2N-1, 2) array.

        The upper surface is traversed from trailing edge to leading edge,
        then the lower surface from leading edge to trailing edge,
        forming a closed contour.
        """
        x = self.distribution(self.sample_rate)
        y_c = self.camber_line.as_array()[:, 1]
        y_t = self.thickness_distribution.as_array()[:, 1].copy()

        # Close trailing edge if thickness is nonzero
        if y_t[-1] != 0:
            y_t[-1] = 0.0

        # Compute camber slope angle
        theta = np.arctan(self.camber_line.get_derivations())
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Upper surface (suction side)
        x_u = x - y_t * sin_theta
        y_u = y_c + y_t * cos_theta

        # Lower surface (pressure side)
        x_l = x + y_t * sin_theta
        y_l = y_c - y_t * cos_theta

        # Reverse upper surface so profile goes TE->LE->TE
        x_u = x_u[::-1]
        y_u = y_u[::-1]

        # Combine: upper (reversed, skip last to avoid LE duplication) + lower
        x_closed = np.concatenate([x_u[:-1], x_l])
        y_closed = np.concatenate([y_u[:-1], y_l])

        return np.column_stack((x_closed, y_closed))

    def upper_surface(self) -> NDArray[np.float64]:
        """Return upper (suction) surface as (N, 2) array from LE to TE."""
        x = self.distribution(self.sample_rate)
        y_c = self.camber_line.as_array()[:, 1]
        y_t = self.thickness_distribution.as_array()[:, 1].copy()
        if y_t[-1] != 0:
            y_t[-1] = 0.0
        theta = np.arctan(self.camber_line.get_derivations())
        x_u = x - y_t * np.sin(theta)
        y_u = y_c + y_t * np.cos(theta)
        return np.column_stack((x_u, y_u))

    def lower_surface(self) -> NDArray[np.float64]:
        """Return lower (pressure) surface as (N, 2) array from LE to TE."""
        x = self.distribution(self.sample_rate)
        y_c = self.camber_line.as_array()[:, 1]
        y_t = self.thickness_distribution.as_array()[:, 1].copy()
        if y_t[-1] != 0:
            y_t[-1] = 0.0
        theta = np.arctan(self.camber_line.get_derivations())
        x_l = x + y_t * np.sin(theta)
        y_l = y_c - y_t * np.cos(theta)
        return np.column_stack((x_l, y_l))
