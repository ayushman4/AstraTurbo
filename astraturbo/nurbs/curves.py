"""NURBS curve utilities using geomdl.

Replaces V1 nurbsToolSet.py curve functions. All functions operate on
geomdl BSpline.Curve objects and numpy arrays instead of PythonNURBS.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from geomdl import BSpline, utilities, fitting


def interpolate_2d(points: NDArray[np.float64], degree: int = 3) -> BSpline.Curve:
    """Interpolate a 2D NURBS curve through points.

    Args:
        points: (N, 2) array of [x, y] points.
        degree: Curve degree (clamped to N-1 if needed).

    Returns:
        geomdl BSpline.Curve (2D, z=0).
    """
    n = len(points)
    degree = min(degree, n - 1)
    pts_3d = [[float(p[0]), float(p[1]), 0.0] for p in points]
    crv = fitting.interpolate_curve(pts_3d, degree)
    return crv


def interpolate_3d(
    points: NDArray[np.float64], degree: int = 3
) -> BSpline.Curve:
    """Interpolate a 3D NURBS curve through points.

    Replaces V1 interpolateNurbs3D() and globalInterpolateNurbs3D().

    Args:
        points: (N, 3) array of [x, y, z] points.
        degree: Curve degree.

    Returns:
        geomdl BSpline.Curve.
    """
    n = len(points)
    degree = min(degree, n - 1)
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in points]
    crv = fitting.interpolate_curve(pts, degree)
    return crv


def approximate_3d(
    points: NDArray[np.float64],
    degree: int = 3,
    num_ctrlpts: int = 0,
) -> BSpline.Curve:
    """Approximate a 3D NURBS curve through points (least squares).

    Replaces V1 leastSquares curve fitting.

    Args:
        points: (N, 3) array.
        degree: Curve degree.
        num_ctrlpts: Number of control points (0 = same as data points).

    Returns:
        geomdl BSpline.Curve.
    """
    n = len(points)
    if num_ctrlpts <= 0:
        num_ctrlpts = n
    degree = min(degree, num_ctrlpts - 1)
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in points]
    crv = fitting.approximate_curve(pts, degree, ctrlpts_size=num_ctrlpts)
    return crv


def evaluate_curve(curve: BSpline.Curve, u: float) -> NDArray[np.float64]:
    """Evaluate a curve at parameter u, returning a numpy array."""
    pt = curve.evaluate_single(u)
    return np.array(pt, dtype=np.float64)


def evaluate_curve_array(
    curve: BSpline.Curve, n_points: int = 200
) -> NDArray[np.float64]:
    """Evaluate a curve at n_points uniformly spaced parameters.

    Returns:
        (N, 3) array of [x, y, z] points.
    """
    curve.sample_size = n_points
    curve.evaluate()
    return np.array(curve.evalpts, dtype=np.float64)


def curve_length(
    curve: BSpline.Curve, u_start: float = 0.0, u_end: float = 1.0,
    n_samples: int = 500,
) -> float:
    """Estimate arc length of a curve segment by sampling.

    Args:
        curve: The NURBS curve.
        u_start: Start parameter.
        u_end: End parameter.
        n_samples: Number of sample points for numerical integration.

    Returns:
        Approximate arc length.
    """
    params = np.linspace(u_start, u_end, n_samples)
    pts = np.array([curve.evaluate_single(u) for u in params], dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return float(np.sum(segment_lengths))


def find_u_from_point(
    curve: BSpline.Curve,
    target: NDArray[np.float64],
    n_samples: int = 500,
) -> float:
    """Find the parameter u on curve closest to target point.

    Replaces V1 calculateUFromPoint() which used PythonNURBS minDist2.

    Args:
        curve: The NURBS curve.
        target: (3,) target point.
        n_samples: Number of samples for initial search.

    Returns:
        Parameter u in [0, 1] closest to target.
    """
    from scipy.optimize import minimize_scalar

    # Coarse search
    params = np.linspace(0, 1, n_samples)
    pts = np.array([curve.evaluate_single(u) for u in params], dtype=np.float64)
    dists = np.sqrt(np.sum((pts - target) ** 2, axis=1))
    best_idx = np.argmin(dists)

    # Refine with bounded minimization
    u_lo = params[max(0, best_idx - 1)]
    u_hi = params[min(len(params) - 1, best_idx + 1)]

    def dist_func(u):
        pt = np.array(curve.evaluate_single(u))
        return float(np.sum((pt - target) ** 2))

    result = minimize_scalar(dist_func, bounds=(u_lo, u_hi), method="bounded")
    return float(result.x)


def find_u_from_z(
    curve: BSpline.Curve,
    z_target: float,
    n_samples: int = 500,
) -> float:
    """Find parameter u where curve z-coordinate matches z_target.

    Replaces V1 calculateUFromZ().
    """
    from scipy.optimize import brentq

    params = np.linspace(0, 1, n_samples)
    z_vals = np.array([curve.evaluate_single(u)[2] for u in params])

    # Find bracket
    diffs = z_vals - z_target
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]
    if len(sign_changes) == 0:
        # Return closest point
        return float(params[np.argmin(np.abs(diffs))])

    idx = sign_changes[0]
    u_lo, u_hi = float(params[idx]), float(params[idx + 1])

    def f(u):
        return curve.evaluate_single(u)[2] - z_target

    return float(brentq(f, u_lo, u_hi))
