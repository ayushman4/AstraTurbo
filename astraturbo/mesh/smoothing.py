"""Mesh smoothing algorithms for structured grids.

Provides Laplacian smoothing and orthogonality correction for
structured (Ni, Nj, D) mesh blocks. These are iterative methods
that improve cell quality while preserving boundary positions.

References:
    Knupp, P.M., "Winslow smoothing on two-dimensional unstructured
    meshes", Engineering with Computers, 1999.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def laplacian_smooth(
    block: NDArray[np.float64],
    n_iterations: int = 50,
    omega: float = 0.5,
    fix_boundaries: bool = True,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Apply Laplacian smoothing to a structured mesh block.

    Each interior point is moved toward the average of its neighbors.
    Boundary points are optionally held fixed.

    The update rule is:
        P_new = (1 - omega) * P_old + omega * average(neighbors)

    where neighbors are the 4 adjacent grid points in i and j directions.

    Args:
        block: (Ni, Nj, D) structured mesh points (D=2 or 3).
        n_iterations: Number of smoothing iterations.
        omega: Relaxation factor in (0, 1]. 0.5 is moderate smoothing,
            1.0 is full Laplacian replacement. Values > 1 may diverge.
        fix_boundaries: If True, boundary points (i=0, i=Ni-1, j=0, j=Nj-1)
            are held fixed. If False, all points including boundaries are
            smoothed (use with caution).

    Returns:
        Tuple of (smoothed_block, metrics) where metrics contains
        quality measurements before and after smoothing.
    """
    ni, nj = block.shape[0], block.shape[1]
    dim = block.shape[2]

    # Compute quality before smoothing
    metrics_before = _compute_quality_metrics(block)

    smoothed = block.copy()
    omega = np.clip(omega, 0.01, 1.0)

    for _iteration in range(n_iterations):
        old = smoothed.copy()

        i_start = 1 if fix_boundaries else 0
        i_end = ni - 1 if fix_boundaries else ni
        j_start = 1 if fix_boundaries else 0
        j_end = nj - 1 if fix_boundaries else nj

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                # Gather neighbor points (4-connectivity)
                neighbors = []
                if i > 0:
                    neighbors.append(old[i - 1, j])
                if i < ni - 1:
                    neighbors.append(old[i + 1, j])
                if j > 0:
                    neighbors.append(old[i, j - 1])
                if j < nj - 1:
                    neighbors.append(old[i, j + 1])

                if len(neighbors) > 0:
                    avg = np.mean(neighbors, axis=0)
                    smoothed[i, j] = (1.0 - omega) * old[i, j] + omega * avg

    # Compute quality after smoothing
    metrics_after = _compute_quality_metrics(smoothed)

    metrics = {
        "before_aspect_ratio_max": metrics_before["aspect_ratio_max"],
        "before_aspect_ratio_mean": metrics_before["aspect_ratio_mean"],
        "before_skewness_max": metrics_before["skewness_max"],
        "before_skewness_mean": metrics_before["skewness_mean"],
        "after_aspect_ratio_max": metrics_after["aspect_ratio_max"],
        "after_aspect_ratio_mean": metrics_after["aspect_ratio_mean"],
        "after_skewness_max": metrics_after["skewness_max"],
        "after_skewness_mean": metrics_after["skewness_mean"],
        "iterations": n_iterations,
        "omega": float(omega),
    }

    return smoothed, metrics


def laplacian_smooth_vectorized(
    block: NDArray[np.float64],
    n_iterations: int = 50,
    omega: float = 0.5,
    fix_boundaries: bool = True,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Vectorized Laplacian smoothing for better performance on large meshes.

    Same semantics as laplacian_smooth but uses numpy array operations
    instead of explicit loops.

    Args:
        block: (Ni, Nj, D) structured mesh points.
        n_iterations: Number of smoothing iterations.
        omega: Relaxation factor in (0, 1].
        fix_boundaries: If True, hold boundary points fixed.

    Returns:
        Tuple of (smoothed_block, metrics).
    """
    ni, nj = block.shape[0], block.shape[1]

    metrics_before = _compute_quality_metrics(block)

    smoothed = block.copy()
    omega = np.clip(omega, 0.01, 1.0)

    for _iteration in range(n_iterations):
        # Compute average of neighbors for interior points
        avg = np.zeros_like(smoothed)
        count = np.zeros((ni, nj, 1), dtype=np.float64)

        # i-1 neighbor
        avg[1:, :, :] += smoothed[:-1, :, :]
        count[1:, :, :] += 1.0

        # i+1 neighbor
        avg[:-1, :, :] += smoothed[1:, :, :]
        count[:-1, :, :] += 1.0

        # j-1 neighbor
        avg[:, 1:, :] += smoothed[:, :-1, :]
        count[:, 1:, :] += 1.0

        # j+1 neighbor
        avg[:, :-1, :] += smoothed[:, 1:, :]
        count[:, :-1, :] += 1.0

        # Avoid division by zero
        count = np.maximum(count, 1.0)
        avg /= count

        # Update interior points
        if fix_boundaries:
            smoothed[1:-1, 1:-1] = (
                (1.0 - omega) * smoothed[1:-1, 1:-1]
                + omega * avg[1:-1, 1:-1]
            )
        else:
            smoothed = (1.0 - omega) * smoothed + omega * avg

    metrics_after = _compute_quality_metrics(smoothed)

    metrics = {
        "before_aspect_ratio_max": metrics_before["aspect_ratio_max"],
        "before_aspect_ratio_mean": metrics_before["aspect_ratio_mean"],
        "before_skewness_max": metrics_before["skewness_max"],
        "before_skewness_mean": metrics_before["skewness_mean"],
        "after_aspect_ratio_max": metrics_after["aspect_ratio_max"],
        "after_aspect_ratio_mean": metrics_after["aspect_ratio_mean"],
        "after_skewness_max": metrics_after["skewness_max"],
        "after_skewness_mean": metrics_after["skewness_mean"],
        "iterations": n_iterations,
        "omega": float(omega),
    }

    return smoothed, metrics


def orthogonality_correction(
    block: NDArray[np.float64],
    n_iterations: int = 20,
    omega: float = 0.3,
    fix_boundaries: bool = True,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Improve mesh orthogonality by adjusting interior point positions.

    At each interior point, the algorithm computes the angle between
    the i-direction and j-direction mesh lines. It then adjusts the point
    position to make the crossing angle closer to 90 degrees.

    The correction is based on minimizing the dot product of tangent
    vectors in the i and j directions at each point.

    Args:
        block: (Ni, Nj, D) structured mesh points (D=2 or 3).
        n_iterations: Number of correction iterations.
        omega: Relaxation factor (0, 1]. Smaller values give more
            conservative corrections.
        fix_boundaries: If True, boundary points are held fixed.

    Returns:
        Tuple of (corrected_block, metrics) with quality before/after.
    """
    ni, nj = block.shape[0], block.shape[1]
    dim = block.shape[2]

    metrics_before = _compute_quality_metrics(block)

    corrected = block.copy()
    omega = np.clip(omega, 0.01, 1.0)

    for _iteration in range(n_iterations):
        old = corrected.copy()

        for i in range(1, ni - 1):
            for j in range(1, nj - 1):
                # Tangent vectors at (i, j)
                di = old[i + 1, j] - old[i - 1, j]  # i-direction tangent
                dj = old[i, j + 1] - old[i, j - 1]  # j-direction tangent

                di_norm = np.linalg.norm(di)
                dj_norm = np.linalg.norm(dj)

                if di_norm < 1e-15 or dj_norm < 1e-15:
                    continue

                # Unit tangents
                di_hat = di / di_norm
                dj_hat = dj / dj_norm

                # Non-orthogonality: dot product should be zero
                dot = np.dot(di_hat, dj_hat)

                if abs(dot) < 1e-10:
                    continue  # Already orthogonal

                # Correction: move point to reduce non-orthogonality
                # Project the cross-term out
                # Move in the direction perpendicular to the average of tangents
                correction = -0.25 * dot * (
                    dj_norm * di_hat + di_norm * dj_hat
                )

                corrected[i, j] = old[i, j] + omega * correction

    metrics_after = _compute_quality_metrics(corrected)

    metrics = {
        "before_aspect_ratio_max": metrics_before["aspect_ratio_max"],
        "before_aspect_ratio_mean": metrics_before["aspect_ratio_mean"],
        "before_skewness_max": metrics_before["skewness_max"],
        "before_skewness_mean": metrics_before["skewness_mean"],
        "before_orthogonality_min": metrics_before["orthogonality_min_angle"],
        "after_aspect_ratio_max": metrics_after["aspect_ratio_max"],
        "after_aspect_ratio_mean": metrics_after["aspect_ratio_mean"],
        "after_skewness_max": metrics_after["skewness_max"],
        "after_skewness_mean": metrics_after["skewness_mean"],
        "after_orthogonality_min": metrics_after["orthogonality_min_angle"],
        "iterations": n_iterations,
        "omega": float(omega),
    }

    return corrected, metrics


def combined_smooth(
    block: NDArray[np.float64],
    laplacian_iterations: int = 30,
    ortho_iterations: int = 10,
    laplacian_omega: float = 0.5,
    ortho_omega: float = 0.3,
    n_cycles: int = 3,
    fix_boundaries: bool = True,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Apply alternating Laplacian smoothing and orthogonality correction.

    Runs multiple cycles of Laplacian smoothing followed by orthogonality
    correction for best results.

    Args:
        block: (Ni, Nj, D) structured mesh points.
        laplacian_iterations: Laplacian iterations per cycle.
        ortho_iterations: Orthogonality correction iterations per cycle.
        laplacian_omega: Laplacian relaxation factor.
        ortho_omega: Orthogonality correction relaxation factor.
        n_cycles: Number of alternating cycles.
        fix_boundaries: If True, hold boundary points fixed.

    Returns:
        Tuple of (smoothed_block, metrics).
    """
    metrics_before = _compute_quality_metrics(block)

    current = block.copy()

    for _cycle in range(n_cycles):
        current, _ = laplacian_smooth_vectorized(
            current,
            n_iterations=laplacian_iterations,
            omega=laplacian_omega,
            fix_boundaries=fix_boundaries,
        )
        current, _ = orthogonality_correction(
            current,
            n_iterations=ortho_iterations,
            omega=ortho_omega,
            fix_boundaries=fix_boundaries,
        )

    metrics_after = _compute_quality_metrics(current)

    metrics = {
        "before_aspect_ratio_max": metrics_before["aspect_ratio_max"],
        "before_skewness_max": metrics_before["skewness_max"],
        "after_aspect_ratio_max": metrics_after["aspect_ratio_max"],
        "after_skewness_max": metrics_after["skewness_max"],
        "n_cycles": n_cycles,
        "total_laplacian_iters": n_cycles * laplacian_iterations,
        "total_ortho_iters": n_cycles * ortho_iterations,
    }

    return current, metrics


def _compute_quality_metrics(block: NDArray[np.float64]) -> dict[str, float]:
    """Compute comprehensive quality metrics for a structured block.

    Args:
        block: (Ni, Nj, D) structured mesh points.

    Returns:
        Dictionary with aspect_ratio_max, aspect_ratio_mean,
        skewness_max, skewness_mean, orthogonality_min_angle.
    """
    ni, nj = block.shape[0] - 1, block.shape[1] - 1
    dim = block.shape[2]

    if ni <= 0 or nj <= 0:
        return {
            "aspect_ratio_max": 0.0,
            "aspect_ratio_mean": 0.0,
            "skewness_max": 0.0,
            "skewness_mean": 0.0,
            "orthogonality_min_angle": 90.0,
        }

    aspect_ratios = np.zeros(ni * nj)
    skewness_vals = np.zeros(ni * nj)
    min_angles_all = np.zeros(ni * nj)

    idx = 0
    for i in range(ni):
        for j in range(nj):
            p0 = block[i, j]
            p1 = block[i + 1, j]
            p2 = block[i + 1, j + 1]
            p3 = block[i, j + 1]

            # Edge lengths
            edges = [
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p3 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p2 - p3),
            ]
            min_e = min(edges)
            max_e = max(edges)
            aspect_ratios[idx] = max_e / min_e if min_e > 1e-15 else 1e10

            # Corner angles
            corners = [p0, p1, p2, p3]
            angles = []
            for k in range(4):
                v1 = corners[(k - 1) % 4] - corners[k]
                v2 = corners[(k + 1) % 4] - corners[k]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-15 or n2 < 1e-15:
                    angles.append(90.0)
                    continue
                cos_a = np.dot(v1, v2) / (n1 * n2)
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_a)))

            theta_max = max(angles)
            theta_min = min(angles)
            skewness_vals[idx] = max(
                (theta_max - 90.0) / 90.0,
                (90.0 - theta_min) / 90.0,
            )
            min_angles_all[idx] = theta_min

            idx += 1

    return {
        "aspect_ratio_max": float(np.max(aspect_ratios)),
        "aspect_ratio_mean": float(np.mean(aspect_ratios)),
        "skewness_max": float(np.max(skewness_vals)),
        "skewness_mean": float(np.mean(skewness_vals)),
        "orthogonality_min_angle": float(np.min(min_angles_all)),
    }
