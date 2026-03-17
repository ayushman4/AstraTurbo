"""3D blade surface generation via NURBS lofting.

Ported from V1 bladeRow.py blade surface creation. Takes stacked 3D
profiles and lofts a smooth NURBS surface through them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..nurbs.surfaces import interpolate_surface


def loft_blade_surface(
    profiles_3d: list[NDArray[np.float64]],
    degree_u: int = 3,
    degree_v: int = 3,
):
    """Loft a NURBS surface through a set of 3D blade profiles.

    Args:
        profiles_3d: List of M arrays, each (N, 3). All profiles must
            have the same number of points N.
        degree_u: Surface degree in spanwise (u) direction.
        degree_v: Surface degree in profile (v) direction.

    Returns:
        geomdl BSpline.Surface lofted through the profiles.
    """
    n_profiles = len(profiles_3d)
    n_points = profiles_3d[0].shape[0]

    # Validate all profiles have same point count
    for i, p in enumerate(profiles_3d):
        if p.shape[0] != n_points:
            raise ValueError(
                f"Profile {i} has {p.shape[0]} points, expected {n_points}"
            )

    # Build point matrix (n_profiles, n_points, 3)
    point_matrix = np.array(profiles_3d, dtype=np.float64)

    return interpolate_surface(point_matrix, degree_u, degree_v)


def compute_leading_trailing_edges(
    profiles_3d: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract leading edge and trailing edge curves from stacked profiles.

    Assumes profiles are ordered from LE (index=0 after reversal in
    superposition) through suction side, LE, pressure side to TE.

    For a closed superposition profile, the midpoint corresponds to the LE.

    Args:
        profiles_3d: List of M arrays, each (N, 3).

    Returns:
        Tuple of (le_points, te_points), each (M, 3).
    """
    le_points = []
    te_points = []
    for profile in profiles_3d:
        n = len(profile)
        # Leading edge is approximately at the midpoint of the closed contour
        le_idx = n // 2
        le_points.append(profile[le_idx])
        # Trailing edge is at the start/end
        te_points.append(profile[0])

    return np.array(le_points, dtype=np.float64), np.array(te_points, dtype=np.float64)
