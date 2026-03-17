"""NURBS-based camber line.

Uses the geomdl library for NURBS curve interpolation through control points,
providing maximum flexibility for custom camber line shapes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import memoize
from .camberline import CamberLine


class NURBSCamberLine(CamberLine):
    """NURBS camber line defined by control points.

    Parameters:
        control_points: (N, 2) array of [x, y] control points in [0, 1].
            First point should be (0, 0), last should be (1, 0) for a
            standard normalized camber line.
        degree: NURBS curve degree (default 3).
    """

    def __init__(
        self,
        control_points: NDArray[np.float64] | None = None,
        degree: int = 3,
    ) -> None:
        super().__init__()
        if control_points is None:
            # Default: straight line (no camber)
            control_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        self._control_points = np.asarray(control_points, dtype=np.float64)
        self._degree = degree

    @property
    def control_points(self) -> NDArray[np.float64]:
        return self._control_points

    @control_points.setter
    def control_points(self, pts: NDArray[np.float64]) -> None:
        self._control_points = np.asarray(pts, dtype=np.float64)
        self.invalidate_cache()
        self.update()

    @property
    def degree(self) -> int:
        return self._degree

    def _build_curve(self):
        """Build a geomdl BSpline curve from control points."""
        from geomdl import BSpline, utilities

        crv = BSpline.Curve()
        crv.degree = min(self._degree, len(self._control_points) - 1)
        crv.ctrlpts = self._control_points.tolist()
        crv.knotvector = utilities.generate_knot_vector(
            crv.degree, len(crv.ctrlpts)
        )
        return crv

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        crv = self._build_curve()
        crv.sample_size = self.sample_rate
        crv.evaluate()
        return np.array(crv.evalpts, dtype=np.float64)

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        from geomdl import operations

        crv = self._build_curve()
        n = self.sample_rate
        params = np.linspace(0, 1, n)
        dydx = np.empty(n, dtype=np.float64)
        for i, u in enumerate(params):
            ders = operations.derivatives_curve(crv, u, order=1)
            dx = ders[1][0]
            dy = ders[1][1]
            dydx[i] = dy / dx if abs(dx) > 1e-15 else 0.0
        return dydx
