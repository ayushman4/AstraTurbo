"""Coordinate transform and geometry operations.

Replaces V1 nurbsToolSet.py utility functions for coordinate transforms
and geometric computations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def xyz_to_rpz(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Cartesian (x, y, z) to cylindrical (r, phi, z).

    Convention: r = sqrt(x^2 + y^2), phi = atan2(y, x), z = z
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    result = np.empty_like(xyz)
    result[..., 0] = r
    result[..., 1] = phi
    result[..., 2] = z
    return result


def rpz_to_xyz(rpz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert cylindrical (r, phi, z) to Cartesian (x, y, z)."""
    r, phi, z = rpz[..., 0], rpz[..., 1], rpz[..., 2]
    result = np.empty_like(rpz)
    result[..., 0] = r * np.cos(phi)
    result[..., 1] = r * np.sin(phi)
    result[..., 2] = z
    return result


def norm(v: NDArray[np.float64]) -> float:
    """Compute Euclidean norm of a vector."""
    return float(np.sqrt(np.sum(v**2)))


def distance(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> float:
    """Compute distance between two points."""
    return norm(p2 - p1)


def normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a vector to unit length."""
    n = norm(v)
    if n < 1e-15:
        return v
    return v / n


def angle_between(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    """Compute angle (radians) between two vectors."""
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 < 1e-15 or n2 < 1e-15:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def meridional_coordinate(xyz: NDArray[np.float64]) -> float:
    """Compute meridional coordinate m = sqrt(x^2 + y^2 + z^2)."""
    return float(np.sqrt(np.sum(xyz**2)))
