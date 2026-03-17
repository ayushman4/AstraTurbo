"""Transfinite interpolation (TFI) for structured mesh generation.

Implements algebraic mesh generation using transfinite interpolation,
which creates interior mesh points from boundary curves. This is the
core algorithm used by both the SCM mesher and the O-grid generator.

References:
    Gordon, W.J. and Hall, C.A., "Construction of curvilinear
    co-ordinate systems and applications to mesh generation",
    Int. J. Num. Methods Eng., 1973.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tfi_2d(
    bottom: NDArray[np.float64],
    top: NDArray[np.float64],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Generate a 2D structured mesh via transfinite interpolation.

    Given four boundary curves, computes interior points using the
    Boolean sum formula:

        P(s,t) = (1-t)*bottom(s) + t*top(s)
               + (1-s)*left(t) + s*right(t)
               - (1-s)*(1-t)*P00 - s*(1-t)*P10
               - (1-s)*t*P01 - s*t*P11

    Args:
        bottom: (Ni, D) points along bottom edge (s=0..1, t=0).
        top: (Ni, D) points along top edge (s=0..1, t=1).
        left: (Nj, D) points along left edge (s=0, t=0..1).
        right: (Nj, D) points along right edge (s=1, t=0..1).

    Returns:
        (Ni, Nj, D) mesh point array.

    Note:
        Corner points must be consistent:
        bottom[0] == left[0], bottom[-1] == right[0]
        top[0] == left[-1], top[-1] == right[-1]
    """
    ni = len(bottom)
    nj = len(left)
    dim = bottom.shape[1]

    # Corner points
    p00 = bottom[0]
    p10 = bottom[-1]
    p01 = top[0]
    p11 = top[-1]

    # Normalized parameters
    s = np.linspace(0, 1, ni)
    t = np.linspace(0, 1, nj)

    mesh = np.zeros((ni, nj, dim), dtype=np.float64)

    for i in range(ni):
        for j in range(nj):
            si, tj = s[i], t[j]

            # Boundary interpolation (projectors)
            p_st = (1 - tj) * bottom[i] + tj * top[i]
            p_ts = (1 - si) * left[j] + si * right[j]

            # Corner correction (tensor product)
            p_corner = (
                (1 - si) * (1 - tj) * p00
                + si * (1 - tj) * p10
                + (1 - si) * tj * p01
                + si * tj * p11
            )

            # Boolean sum
            mesh[i, j] = p_st + p_ts - p_corner

    return mesh


def tfi_2d_vectorized(
    bottom: NDArray[np.float64],
    top: NDArray[np.float64],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized version of tfi_2d for better performance.

    Same interface as tfi_2d but uses numpy broadcasting instead of loops.
    """
    ni = len(bottom)
    nj = len(left)
    dim = bottom.shape[1]

    p00 = bottom[0]
    p10 = bottom[-1]
    p01 = top[0]
    p11 = top[-1]

    s = np.linspace(0, 1, ni)[:, None, None]  # (Ni, 1, 1)
    t = np.linspace(0, 1, nj)[None, :, None]  # (1, Nj, 1)

    # Expand boundary arrays for broadcasting
    b = bottom[:, None, :]    # (Ni, 1, D)
    tp = top[:, None, :]      # (Ni, 1, D)
    l = left[None, :, :]      # (1, Nj, D)
    r = right[None, :, :]     # (1, Nj, D)

    # Projectors
    p_st = (1 - t) * b + t * tp   # (Ni, Nj, D)
    p_ts = (1 - s) * l + s * r    # (Ni, Nj, D)

    # Corner correction
    p_corner = (
        (1 - s) * (1 - t) * p00
        + s * (1 - t) * p10
        + (1 - s) * t * p01
        + s * t * p11
    )

    return p_st + p_ts - p_corner


def apply_grading(
    n_cells: int,
    grading_ratio: float = 1.0,
) -> NDArray[np.float64]:
    """Generate a graded parameter distribution for mesh refinement.

    Args:
        n_cells: Number of cells (n_points = n_cells + 1).
        grading_ratio: Ratio of last cell size to first cell size.
            1.0 = uniform, >1.0 = cells grow, <1.0 = cells shrink.

    Returns:
        (n_cells + 1,) array of parameter values in [0, 1].
    """
    n_points = n_cells + 1
    if abs(grading_ratio - 1.0) < 1e-10:
        return np.linspace(0, 1, n_points)

    # Geometric series
    r = grading_ratio ** (1.0 / (n_cells - 1))
    sizes = r ** np.arange(n_cells)
    cumulative = np.zeros(n_points)
    cumulative[1:] = np.cumsum(sizes)
    cumulative /= cumulative[-1]
    return cumulative


def tfi_2d_graded(
    bottom: NDArray[np.float64],
    top: NDArray[np.float64],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    grading_s: float = 1.0,
    grading_t: float = 1.0,
    n_cells_s: int = 0,
    n_cells_t: int = 0,
) -> NDArray[np.float64]:
    """TFI with graded parameter distributions.

    Args:
        bottom, top, left, right: Boundary curves.
        grading_s: Grading ratio in s-direction.
        grading_t: Grading ratio in t-direction.
        n_cells_s: Cells in s (0 = use boundary point count - 1).
        n_cells_t: Cells in t (0 = use boundary point count - 1).

    Returns:
        (Ns, Nt, D) graded mesh.
    """
    ni_orig = len(bottom)
    nj_orig = len(left)

    if n_cells_s <= 0:
        n_cells_s = ni_orig - 1
    if n_cells_t <= 0:
        n_cells_t = nj_orig - 1

    # Generate graded parameter distributions
    s_params = apply_grading(n_cells_s, grading_s)
    t_params = apply_grading(n_cells_t, grading_t)

    ni = len(s_params)
    nj = len(t_params)
    dim = bottom.shape[1]

    # Resample boundaries at graded parameters
    s_orig = np.linspace(0, 1, ni_orig)
    t_orig = np.linspace(0, 1, nj_orig)

    bottom_g = np.zeros((ni, dim))
    top_g = np.zeros((ni, dim))
    left_g = np.zeros((nj, dim))
    right_g = np.zeros((nj, dim))

    for d in range(dim):
        bottom_g[:, d] = np.interp(s_params, s_orig, bottom[:, d])
        top_g[:, d] = np.interp(s_params, s_orig, top[:, d])
        left_g[:, d] = np.interp(t_params, t_orig, left[:, d])
        right_g[:, d] = np.interp(t_params, t_orig, right[:, d])

    return tfi_2d_vectorized(bottom_g, top_g, left_g, right_g)
