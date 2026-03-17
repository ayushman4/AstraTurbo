"""Hub and shroud contour handling.

Ported from V1 bladeRow.py hub/shroud utilities.
Hub and shroud contours are 2D curves in the meridional (z, r) plane
that define the flow passage boundaries.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..nurbs.curves import interpolate_2d, evaluate_curve_array


class MeridionalContour:
    """A meridional contour (hub or shroud) defined in the (z, r) plane.

    Attributes:
        points: (N, 2) array of [z, r] points defining the contour.
    """

    def __init__(self, points: NDArray[np.float64]) -> None:
        self._points = np.asarray(points, dtype=np.float64)
        self._curve = None

    @property
    def points(self) -> NDArray[np.float64]:
        return self._points

    @points.setter
    def points(self, pts: NDArray[np.float64]) -> None:
        self._points = np.asarray(pts, dtype=np.float64)
        self._curve = None

    @property
    def curve(self):
        """Lazy-build the NURBS curve from points."""
        if self._curve is None:
            # Pad to 3D for geomdl (z, r, 0) -> (x=z, y=r, z=0)
            pts_3d = np.column_stack(
                (self._points, np.zeros(len(self._points)))
            )
            self._curve = interpolate_2d(self._points)
        return self._curve

    def evaluate(self, n_points: int = 200) -> NDArray[np.float64]:
        """Evaluate the contour at n_points uniformly spaced parameters.

        Returns:
            (N, 2) array of [z, r] points.
        """
        pts = evaluate_curve_array(self.curve, n_points)
        return pts[:, :2]

    def radius_at_z(self, z: float) -> float:
        """Get the radius at a given axial position z.

        Uses linear interpolation on the discrete points.
        """
        z_vals = self._points[:, 0]
        r_vals = self._points[:, 1]
        return float(np.interp(z, z_vals, r_vals))


def compute_stacking_line(
    hub: MeridionalContour,
    shroud: MeridionalContour,
    n_spans: int,
    n_points: int = 200,
) -> NDArray[np.float64]:
    """Compute stacking positions along the span between hub and shroud.

    Args:
        hub: Hub contour.
        shroud: Shroud contour.
        n_spans: Number of span positions (including hub and shroud).
        n_points: Points per contour evaluation.

    Returns:
        (n_spans,) array of radial positions from hub to shroud.
    """
    hub_pts = hub.evaluate(n_points)
    shroud_pts = shroud.evaluate(n_points)

    # Use midpoint z to get radii
    z_mid = (hub_pts[:, 0].mean() + shroud_pts[:, 0].mean()) / 2.0
    r_hub = hub.radius_at_z(z_mid)
    r_shroud = shroud.radius_at_z(z_mid)

    return np.linspace(r_hub, r_shroud, n_spans)
