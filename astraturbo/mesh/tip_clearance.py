"""Tip clearance mesh generation for turbomachinery blades.

Generates a structured hexahedral mesh in the gap between blade tip
and casing, critical for accurate tip leakage flow predictions.

The mesh uses transfinite interpolation with configurable grading
toward both the blade tip and casing walls.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_tip_clearance_mesh(
    blade_tip_curve: NDArray[np.float64],
    casing_contour: NDArray[np.float64],
    gap_height: float,
    n_radial: int = 10,
    n_streamwise: int | None = None,
    grading_ratio: float = 1.2,
    grading_mode: str = "both",
) -> dict[str, NDArray[np.float64] | dict]:
    """Generate a structured mesh in the tip clearance gap.

    Creates a mesh between the blade tip curve and the casing contour.
    The blade tip and casing may have different shapes in the meridional
    plane, so the mesh smoothly transitions between them.

    Args:
        blade_tip_curve: (N, 3) array of points defining the blade tip.
            Points should be ordered along the chord (leading to trailing edge).
            Coordinates are (x, y, z) where z is the radial/spanwise direction.
        casing_contour: (N, 3) or (M, 3) array of points on the casing surface
            above the blade tip. If M != N, it will be resampled to N points.
        gap_height: Tip clearance gap height (m). Used when casing_contour
            is not provided or for validation.
        n_radial: Number of cells in the radial (gap) direction.
        n_streamwise: Number of cells in the streamwise direction.
            If None, uses len(blade_tip_curve) - 1.
        grading_ratio: Cell size ratio for grading. > 1.0 means cells near
            walls are smaller than interior cells.
        grading_mode: Where to apply grading:
            'tip' - grade toward blade tip only
            'casing' - grade toward casing only
            'both' - grade toward both walls (double-sided)
            'uniform' - no grading

    Returns:
        Dictionary containing:
            'points': (Ns, Nr, 3) structured mesh point array
            'quality': dict with mesh quality metrics
            'n_streamwise': number of streamwise points
            'n_radial': number of radial points
    """
    n_tip = len(blade_tip_curve)

    if n_streamwise is None:
        n_streamwise = n_tip - 1

    n_s_pts = n_streamwise + 1
    n_r_pts = n_radial + 1

    # Resample blade tip to n_s_pts if needed
    blade_tip = _resample_curve(blade_tip_curve, n_s_pts)

    # Build casing curve
    if casing_contour is not None and len(casing_contour) > 0:
        casing = _resample_curve(casing_contour, n_s_pts)
    else:
        # Offset blade tip by gap_height in the local normal direction
        casing = _offset_curve_radially(blade_tip, gap_height)

    # Generate graded radial distribution
    r_params = _graded_distribution(n_radial, grading_ratio, grading_mode)

    # Build the structured mesh via transfinite interpolation
    points = np.zeros((n_s_pts, n_r_pts, 3), dtype=np.float64)

    for i in range(n_s_pts):
        p_tip = blade_tip[i]
        p_cas = casing[i]

        for j in range(n_r_pts):
            t = r_params[j]
            points[i, j] = (1.0 - t) * p_tip + t * p_cas

    # Compute quality metrics
    quality = _compute_tip_mesh_quality(points)

    return {
        "points": points,
        "quality": quality,
        "n_streamwise": n_s_pts,
        "n_radial": n_r_pts,
    }


def _resample_curve(
    curve: NDArray[np.float64], n_points: int
) -> NDArray[np.float64]:
    """Resample a 3D curve to a given number of equally-spaced points.

    Args:
        curve: (M, 3) input curve points.
        n_points: Desired number of output points.

    Returns:
        (n_points, 3) resampled curve.
    """
    if len(curve) == n_points:
        return curve.copy()

    # Compute cumulative arc length
    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_length = np.zeros(len(curve))
    arc_length[1:] = np.cumsum(seg_lengths)
    total_length = arc_length[-1]

    if total_length < 1e-15:
        return np.tile(curve[0], (n_points, 1))

    # Normalize to [0, 1]
    arc_length /= total_length

    # Interpolate at uniform arc-length spacing
    t_new = np.linspace(0, 1, n_points)
    resampled = np.zeros((n_points, 3), dtype=np.float64)
    for d in range(3):
        resampled[:, d] = np.interp(t_new, arc_length, curve[:, d])

    return resampled


def _offset_curve_radially(
    curve: NDArray[np.float64], offset: float
) -> NDArray[np.float64]:
    """Offset a curve in the radial direction (assumed to be the z-axis or
    the direction perpendicular to the streamwise plane).

    For a general 3D curve, computes local normals and offsets along them.
    Falls back to z-direction if normals are degenerate.

    Args:
        curve: (N, 3) input curve.
        offset: Offset distance.

    Returns:
        (N, 3) offset curve.
    """
    n = len(curve)
    result = curve.copy()

    # Compute tangent vectors
    tangents = np.zeros_like(curve)
    tangents[0] = curve[1] - curve[0]
    tangents[-1] = curve[-1] - curve[-2]
    tangents[1:-1] = curve[2:] - curve[:-2]

    # Normalize tangents
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    tangents = tangents / norms

    # Compute normal in the plane containing the tangent and z-axis
    z_axis = np.array([0.0, 0.0, 1.0])

    for i in range(n):
        t = tangents[i]
        # Normal = z_axis - (z_axis . t) * t (Gram-Schmidt against tangent)
        normal = z_axis - np.dot(z_axis, t) * t
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            # Tangent is parallel to z, use y-axis instead
            normal = np.array([0.0, 1.0, 0.0])
            normal = normal - np.dot(normal, t) * t
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                normal = np.array([0.0, 0.0, 1.0])
            else:
                normal = normal / norm_len
        else:
            normal = normal / norm_len

        result[i] = curve[i] + offset * normal

    return result


def _graded_distribution(
    n_cells: int, grading_ratio: float, mode: str
) -> NDArray[np.float64]:
    """Generate a graded distribution in [0, 1].

    Args:
        n_cells: Number of cells.
        grading_ratio: Expansion ratio (>1 means smaller cells at walls).
        mode: 'tip', 'casing', 'both', or 'uniform'.

    Returns:
        (n_cells + 1,) array of parameter values in [0, 1].
    """
    n_pts = n_cells + 1

    if mode == "uniform" or abs(grading_ratio - 1.0) < 1e-10:
        return np.linspace(0, 1, n_pts)

    # Single-sided geometric grading
    r = grading_ratio ** (1.0 / max(n_cells - 1, 1))
    sizes = r ** np.arange(n_cells)

    if mode == "tip":
        # Small cells at t=0 (blade tip), growing toward casing
        cumul = np.zeros(n_pts)
        cumul[1:] = np.cumsum(sizes)
        cumul /= cumul[-1]
        return cumul

    elif mode == "casing":
        # Small cells at t=1 (casing), growing toward tip
        cumul = np.zeros(n_pts)
        cumul[1:] = np.cumsum(sizes[::-1])
        cumul /= cumul[-1]
        return cumul

    elif mode == "both":
        # Double-sided: split into two halves, grade each toward its wall
        n_half1 = n_cells // 2
        n_half2 = n_cells - n_half1

        # First half: small at tip (t=0)
        if n_half1 > 0:
            r1 = grading_ratio ** (1.0 / max(n_half1 - 1, 1))
            sizes1 = r1 ** np.arange(n_half1)
            cumul1 = np.zeros(n_half1 + 1)
            cumul1[1:] = np.cumsum(sizes1)
            cumul1 /= cumul1[-1]
            cumul1 *= 0.5  # Scale to [0, 0.5]
        else:
            cumul1 = np.array([0.0])

        # Second half: small at casing (t=1)
        if n_half2 > 0:
            r2 = grading_ratio ** (1.0 / max(n_half2 - 1, 1))
            sizes2 = r2 ** np.arange(n_half2)
            cumul2 = np.zeros(n_half2 + 1)
            cumul2[1:] = np.cumsum(sizes2[::-1])
            cumul2 /= cumul2[-1]
            cumul2 = 0.5 + 0.5 * cumul2  # Scale to [0.5, 1.0]
        else:
            cumul2 = np.array([1.0])

        # Merge (avoid duplicating the midpoint)
        return np.concatenate([cumul1, cumul2[1:]])

    else:
        return np.linspace(0, 1, n_pts)


def _compute_tip_mesh_quality(
    points: NDArray[np.float64],
) -> dict[str, float]:
    """Compute quality metrics for the tip clearance mesh.

    Args:
        points: (Ns, Nr, 3) structured mesh points.

    Returns:
        Dictionary with quality metrics.
    """
    ns, nr = points.shape[0], points.shape[1]
    n_cells_s = ns - 1
    n_cells_r = nr - 1

    aspect_ratios = []
    min_angles = []
    max_angles = []

    for i in range(n_cells_s):
        for j in range(n_cells_r):
            # Four corners of the cell
            p00 = points[i, j]
            p10 = points[i + 1, j]
            p11 = points[i + 1, j + 1]
            p01 = points[i, j + 1]

            # Edge lengths
            e1 = np.linalg.norm(p10 - p00)
            e2 = np.linalg.norm(p01 - p00)
            e3 = np.linalg.norm(p11 - p10)
            e4 = np.linalg.norm(p11 - p01)

            edges = [e1, e2, e3, e4]
            min_e = min(edges)
            max_e = max(edges)
            ar = max_e / min_e if min_e > 1e-15 else 1e10
            aspect_ratios.append(ar)

            # Interior angles at each corner
            corners = [p00, p10, p11, p01]
            for k in range(4):
                v1 = corners[(k - 1) % 4] - corners[k]
                v2 = corners[(k + 1) % 4] - corners[k]
                cos_a = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15
                )
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_a))
                min_angles.append(angle)
                max_angles.append(angle)

    aspect_arr = np.array(aspect_ratios)
    all_angles = np.array(min_angles)

    return {
        "n_cells": n_cells_s * n_cells_r,
        "n_points": ns * nr,
        "aspect_ratio_max": float(np.max(aspect_arr)) if len(aspect_arr) > 0 else 0.0,
        "aspect_ratio_mean": float(np.mean(aspect_arr)) if len(aspect_arr) > 0 else 0.0,
        "min_angle": float(np.min(all_angles)) if len(all_angles) > 0 else 0.0,
        "max_angle": float(np.max(all_angles)) if len(all_angles) > 0 else 0.0,
        "orthogonality_min": float(np.min(all_angles)) if len(all_angles) > 0 else 0.0,
    }
