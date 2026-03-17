"""Converters between geomdl and other geometry representations.

Provides conversion between geomdl NURBS objects and numpy arrays,
with optional conversion to cadquery/OCP for CAD export.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geomdl import BSpline


def curve_to_points(
    curve: BSpline.Curve, n_points: int = 200
) -> NDArray[np.float64]:
    """Convert a geomdl curve to an array of evaluated points."""
    curve.sample_size = n_points
    curve.evaluate()
    return np.array(curve.evalpts, dtype=np.float64)


def surface_to_points(
    surface: BSpline.Surface, n_u: int = 50, n_v: int = 50
) -> NDArray[np.float64]:
    """Convert a geomdl surface to an (n_u * n_v, 3) array of points."""
    surface.sample_size_u = n_u
    surface.sample_size_v = n_v
    surface.evaluate()
    return np.array(surface.evalpts, dtype=np.float64)


def points_to_curve(
    points: NDArray[np.float64], degree: int = 3
) -> BSpline.Curve:
    """Create a geomdl BSpline.Curve from control points.

    Args:
        points: (N, 3) or (N, 2) array of control points.
        degree: Curve degree.

    Returns:
        geomdl BSpline.Curve.
    """
    from geomdl import utilities

    n = len(points)
    dim = points.shape[1]
    degree = min(degree, n - 1)

    crv = BSpline.Curve()
    crv.degree = degree
    if dim == 2:
        crv.ctrlpts = [[float(p[0]), float(p[1]), 0.0] for p in points]
    else:
        crv.ctrlpts = [[float(p[0]), float(p[1]), float(p[2])] for p in points]
    crv.knotvector = utilities.generate_knot_vector(degree, n)
    return crv


def convert_2d_to_3d_curve(
    curve_2d: BSpline.Curve, plane: str = "rz"
) -> BSpline.Curve:
    """Convert a 2D NURBS curve to 3D by embedding in a coordinate plane.

    Replaces V1 convertNurbs2Dto3D().

    Args:
        curve_2d: A 2D curve (points have [x, y, 0]).
        plane: 'rz' maps (x, y) -> (y, 0, x) i.e. r->y, z->x
               'xy' keeps (x, y, 0)

    Returns:
        3D BSpline.Curve.
    """
    from geomdl import utilities

    crv3d = BSpline.Curve()
    crv3d.degree = curve_2d.degree

    ctrlpts_3d = []
    for pt in curve_2d.ctrlpts:
        if plane == "rz":
            # V1 convention: 2D is (z, r), 3D is (r, phi=0, z)
            ctrlpts_3d.append([float(pt[1]), 0.0, float(pt[0])])
        else:  # xy
            ctrlpts_3d.append([float(pt[0]), float(pt[1]), 0.0])

    crv3d.ctrlpts = ctrlpts_3d
    crv3d.knotvector = list(curve_2d.knotvector)
    return crv3d
