"""NURBS surface utilities using geomdl.

Replaces V1 nurbsToolSet.py surface functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geomdl import BSpline, fitting, utilities


def interpolate_surface(
    point_matrix: NDArray[np.float64],
    degree_u: int = 3,
    degree_v: int = 3,
) -> BSpline.Surface:
    """Interpolate a NURBS surface through a grid of 3D points.

    Replaces V1 interpolateNurbsSurface() and globalInterpolateNurbsSurface().

    Args:
        point_matrix: (Nu, Nv, 3) array of [x, y, z] points.
            Nu = number of rows (e.g. spanwise profiles).
            Nv = number of columns (e.g. points per profile).
        degree_u: Surface degree in u-direction.
        degree_v: Surface degree in v-direction.

    Returns:
        geomdl BSpline.Surface.
    """
    nu, nv = point_matrix.shape[0], point_matrix.shape[1]
    degree_u = min(degree_u, nu - 1)
    degree_v = min(degree_v, nv - 1)

    # geomdl expects a flat list of points + size_u, size_v
    pts = []
    for i in range(nu):
        for j in range(nv):
            p = point_matrix[i, j]
            pts.append([float(p[0]), float(p[1]), float(p[2])])

    srf = fitting.interpolate_surface(pts, nu, nv, degree_u, degree_v)
    return srf


def approximate_surface(
    point_matrix: NDArray[np.float64],
    degree_u: int = 3,
    degree_v: int = 3,
    num_ctrlpts_u: int = 0,
    num_ctrlpts_v: int = 0,
) -> BSpline.Surface:
    """Approximate a NURBS surface through a grid of 3D points (least squares).

    Replaces V1 leastSquares surface fitting.

    Args:
        point_matrix: (Nu, Nv, 3) array.
        degree_u: Degree in u.
        degree_v: Degree in v.
        num_ctrlpts_u: Number of control points in u (0 = auto).
        num_ctrlpts_v: Number of control points in v (0 = auto).

    Returns:
        geomdl BSpline.Surface.
    """
    nu, nv = point_matrix.shape[0], point_matrix.shape[1]
    if num_ctrlpts_u <= 0:
        num_ctrlpts_u = nu
    if num_ctrlpts_v <= 0:
        num_ctrlpts_v = nv
    degree_u = min(degree_u, num_ctrlpts_u - 1)
    degree_v = min(degree_v, num_ctrlpts_v - 1)

    pts = []
    for i in range(nu):
        for j in range(nv):
            p = point_matrix[i, j]
            pts.append([float(p[0]), float(p[1]), float(p[2])])

    srf = fitting.approximate_surface(
        pts, nu, nv, degree_u, degree_v,
        ctrlpts_size_u=num_ctrlpts_u, ctrlpts_size_v=num_ctrlpts_v,
    )
    return srf


def evaluate_surface(
    surface: BSpline.Surface, u: float, v: float
) -> NDArray[np.float64]:
    """Evaluate a surface at parameters (u, v)."""
    pt = surface.evaluate_single((u, v))
    return np.array(pt, dtype=np.float64)


def evaluate_surface_grid(
    surface: BSpline.Surface,
    n_u: int = 50,
    n_v: int = 50,
) -> NDArray[np.float64]:
    """Evaluate a surface on a uniform (u, v) grid.

    Returns:
        (n_u, n_v, 3) array of evaluated points.
    """
    surface.sample_size_u = n_u
    surface.sample_size_v = n_v
    surface.evaluate()
    pts = np.array(surface.evalpts, dtype=np.float64)
    return pts.reshape(n_u, n_v, 3)
