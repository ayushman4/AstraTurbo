"""Mesh quality metrics for AstraTurbo.

Provides functions to evaluate structured mesh quality including
aspect ratio, skewness, and y+ estimation. These metrics are critical
for ensuring CFD simulation accuracy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_aspect_ratio(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute aspect ratio for each cell in a structured block.

    Aspect ratio = max(edge_length) / min(edge_length) for each cell.
    Ideal value is 1.0 (square cells).

    Args:
        block: (Ni, Nj, D) structured mesh points (D=2 or 3).

    Returns:
        (Ni-1, Nj-1) array of aspect ratios.
    """
    ni, nj = block.shape[0] - 1, block.shape[1] - 1
    aspect = np.zeros((ni, nj), dtype=np.float64)

    for i in range(ni):
        for j in range(nj):
            # Four edges of the quad cell
            e1 = np.linalg.norm(block[i + 1, j] - block[i, j])
            e2 = np.linalg.norm(block[i, j + 1] - block[i, j])
            e3 = np.linalg.norm(block[i + 1, j + 1] - block[i + 1, j])
            e4 = np.linalg.norm(block[i + 1, j + 1] - block[i, j + 1])

            edges = [e1, e2, e3, e4]
            min_e = min(edges)
            max_e = max(edges)
            aspect[i, j] = max_e / min_e if min_e > 1e-15 else 1e10

    return aspect


def compute_skewness(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute equiangle skewness for each cell in a 2D structured block.

    Skewness = max((theta_max - 90) / 90, (90 - theta_min) / 90)
    where theta are the interior angles of the quad cell.

    Ideal = 0.0 (perfectly rectangular).

    Args:
        block: (Ni, Nj, 2) structured mesh points.

    Returns:
        (Ni-1, Nj-1) array of skewness values in [0, 1].
    """
    ni, nj = block.shape[0] - 1, block.shape[1] - 1
    skew = np.zeros((ni, nj), dtype=np.float64)

    for i in range(ni):
        for j in range(nj):
            # Four corners
            p0 = block[i, j]
            p1 = block[i + 1, j]
            p2 = block[i + 1, j + 1]
            p3 = block[i, j + 1]

            corners = [p0, p1, p2, p3]
            angles = []
            for k in range(4):
                v1 = corners[(k - 1) % 4] - corners[k]
                v2 = corners[(k + 1) % 4] - corners[k]
                cos_angle = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15
                )
                cos_angle = np.clip(cos_angle, -1, 1)
                angles.append(np.degrees(np.arccos(cos_angle)))

            theta_max = max(angles)
            theta_min = min(angles)
            skew[i, j] = max(
                (theta_max - 90.0) / 90.0,
                (90.0 - theta_min) / 90.0,
            )

    return skew


def estimate_yplus(
    first_cell_height: float,
    density: float,
    velocity: float,
    dynamic_viscosity: float,
    chord: float,
) -> float:
    """Estimate y+ for the first cell near a wall.

    Uses a flat-plate turbulent boundary layer correlation:
        Re = rho * U * L / mu
        Cf = 0.058 * Re^(-0.2)
        tau_w = 0.5 * Cf * rho * U^2
        u_tau = sqrt(tau_w / rho)
        y+ = rho * u_tau * y / mu

    Args:
        first_cell_height: Height of the first cell near the wall.
        density: Fluid density (kg/m^3).
        velocity: Freestream velocity (m/s).
        dynamic_viscosity: Dynamic viscosity (Pa.s).
        chord: Reference length / chord (m).

    Returns:
        Estimated y+ value.
    """
    re = density * velocity * chord / dynamic_viscosity
    if re < 1.0:
        return 0.0

    # Schlichting flat-plate correlation
    cf = 0.058 * re ** (-0.2)
    tau_w = 0.5 * cf * density * velocity**2
    u_tau = np.sqrt(tau_w / density)
    y_plus = density * u_tau * first_cell_height / dynamic_viscosity

    return float(y_plus)


def first_cell_height_for_yplus(
    target_yplus: float,
    density: float,
    velocity: float,
    dynamic_viscosity: float,
    chord: float,
) -> float:
    """Compute the first cell height needed for a target y+.

    Inverse of estimate_yplus.

    Args:
        target_yplus: Desired y+ value (typically 1.0 for resolved BL).
        density: Fluid density (kg/m^3).
        velocity: Freestream velocity (m/s).
        dynamic_viscosity: Dynamic viscosity (Pa.s).
        chord: Reference length (m).

    Returns:
        Required first cell height (m).
    """
    re = density * velocity * chord / dynamic_viscosity
    if re < 1.0:
        return 0.0

    cf = 0.058 * re ** (-0.2)
    tau_w = 0.5 * cf * density * velocity**2
    u_tau = np.sqrt(tau_w / density)
    y = target_yplus * dynamic_viscosity / (density * u_tau)

    return float(y)


def mesh_quality_report(block: NDArray[np.float64]) -> dict:
    """Generate a summary quality report for a mesh block.

    Args:
        block: (Ni, Nj, 2) structured mesh points.

    Returns:
        Dictionary with quality metrics.
    """
    aspect = compute_aspect_ratio(block)
    skew = compute_skewness(block)

    return {
        "aspect_ratio_max": float(np.max(aspect)),
        "aspect_ratio_mean": float(np.mean(aspect)),
        "skewness_max": float(np.max(skew)),
        "skewness_mean": float(np.mean(skew)),
        "n_cells": int((block.shape[0] - 1) * (block.shape[1] - 1)),
        "n_points": int(block.shape[0] * block.shape[1]),
    }
