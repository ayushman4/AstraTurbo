"""Camber surface extraction from 3D blade geometry.

Ported from V1 camberSurface.py. Extracts the mean (camber) surface
from the pressure and suction sides of a blade.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_camber_surface(
    profiles_3d: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Extract camber surface points from closed 3D profiles.

    For each closed profile (suction side + pressure side), computes
    the midpoint between corresponding upper and lower surface points.

    Args:
        profiles_3d: List of M closed profiles, each (2N-1, 3).

    Returns:
        (M, N, 3) array of camber surface points.
    """
    camber_profiles = []
    for profile in profiles_3d:
        n_total = len(profile)
        n_half = (n_total + 1) // 2

        # Upper surface: first n_half points (reversed from TE to LE)
        upper = profile[:n_half][::-1]
        # Lower surface: last n_half points (LE to TE)
        lower = profile[n_half - 1:]

        # Ensure same length
        n = min(len(upper), len(lower))
        upper = upper[:n]
        lower = lower[:n]

        # Camber = midpoint between upper and lower
        camber = (upper + lower) / 2.0
        camber_profiles.append(camber)

    return np.array(camber_profiles, dtype=np.float64)


def compute_blade_angles(
    camber_profiles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute blade metal angles along the camber surface.

    Computes the tangent angle (beta) at each point along each
    camber profile. Beta is the angle of the tangent vector in
    the meridional-circumferential plane.

    Args:
        camber_profiles: (M, N, 3) array of camber points.

    Returns:
        (M, N) array of blade angles in radians.
    """
    m, n, _ = camber_profiles.shape
    angles = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        profile = camber_profiles[i]
        for j in range(1, n - 1):
            tangent = profile[j + 1] - profile[j - 1]
            # Angle in the xy-plane (circumferential vs axial)
            angles[i, j] = np.arctan2(tangent[1], tangent[0])
        # Boundary: copy nearest
        angles[i, 0] = angles[i, 1]
        angles[i, -1] = angles[i, -2]

    return angles
