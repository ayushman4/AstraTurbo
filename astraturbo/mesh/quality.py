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


def auto_first_cell_height(
    velocity: float,
    chord: float,
    density: float = 1.225,
    dynamic_viscosity: float = 1.8e-5,
    target_yplus: float = 1.0,
    ogrid_layers: int = 20,
    total_bl_thickness_fraction: float = 0.1,
) -> dict[str, float]:
    """Compute recommended first cell height and O-grid grading automatically.

    Given flow conditions and a target y+, this function returns:
      - The first cell height needed
      - A recommended O-grid geometric grading ratio
      - The estimated boundary layer thickness
      - The y+ that would result

    The boundary layer thickness is estimated from the Blasius solution
    for turbulent flow (1/7 power law):
        delta = 0.37 * chord * Re^(-0.2)

    The grading ratio is chosen so that the geometric series of cell
    heights fills from the first cell to the total O-grid thickness.

    Args:
        velocity: Freestream velocity (m/s).
        chord: Reference chord length (m).
        density: Fluid density (kg/m^3). Default: air at STP.
        dynamic_viscosity: Dynamic viscosity (Pa.s). Default: air at STP.
        target_yplus: Desired y+ value. 1.0 for wall-resolved LES/DNS,
            30-100 for wall functions.
        ogrid_layers: Number of O-grid cells in the wall-normal direction.
        total_bl_thickness_fraction: Fraction of chord for total O-grid
            thickness. The O-grid should capture ~1-2x the boundary layer.

    Returns:
        Dictionary with:
            'first_cell_height': Required first cell height (m).
            'grading_ratio': Recommended geometric grading ratio for O-grid.
            'boundary_layer_thickness': Estimated BL thickness (m).
            'reynolds_number': Chord-based Reynolds number.
            'estimated_yplus': The y+ from the computed cell height.
            'ogrid_total_thickness': Total O-grid radial extent (m).
    """
    # Reynolds number
    re = density * velocity * chord / dynamic_viscosity
    if re < 1.0:
        return {
            "first_cell_height": 0.0,
            "grading_ratio": 1.0,
            "boundary_layer_thickness": 0.0,
            "reynolds_number": re,
            "estimated_yplus": 0.0,
            "ogrid_total_thickness": 0.0,
        }

    # Estimate friction coefficient (Schlichting flat-plate correlation)
    cf = 0.058 * re ** (-0.2)
    tau_w = 0.5 * cf * density * velocity**2
    u_tau = np.sqrt(tau_w / density)

    # First cell height for target y+
    y1 = target_yplus * dynamic_viscosity / (density * u_tau)

    # Estimated boundary layer thickness (turbulent, 1/7 power law)
    delta = 0.37 * chord * re ** (-0.2)

    # O-grid total thickness (capture the full BL)
    ogrid_thickness = max(total_bl_thickness_fraction * chord, 1.5 * delta)

    # Compute grading ratio
    # Geometric series: y1 * sum(r^k, k=0..N-1) = ogrid_thickness
    # sum = (r^N - 1) / (r - 1) = ogrid_thickness / y1
    # Solve for r iteratively
    target_sum = ogrid_thickness / y1 if y1 > 1e-15 else 1e10
    n = ogrid_layers

    # Newton's method to find grading ratio r
    r = 1.2  # initial guess
    for _ in range(50):
        if abs(r - 1.0) < 1e-10:
            f = n - target_sum
            fp = 0.5 * n * (n - 1)
            r = max(1.001, r - f / fp if abs(fp) > 1e-15 else r + 0.1)
            continue

        geom_sum = (r**n - 1.0) / (r - 1.0)
        f = geom_sum - target_sum
        # Derivative: d/dr [(r^N - 1)/(r-1)]
        fp = (n * r ** (n - 1) * (r - 1.0) - (r**n - 1.0)) / (r - 1.0) ** 2
        if abs(fp) < 1e-15:
            break
        r_new = r - f / fp
        if abs(r_new - r) < 1e-10:
            break
        r = max(1.001, r_new)

    # Verify y+
    estimated_yplus = density * u_tau * y1 / dynamic_viscosity

    return {
        "first_cell_height": float(y1),
        "grading_ratio": float(r),
        "boundary_layer_thickness": float(delta),
        "reynolds_number": float(re),
        "estimated_yplus": float(estimated_yplus),
        "ogrid_total_thickness": float(ogrid_thickness),
    }


def validate_cfd_mesh(
    mesh,
    velocity: float,
    chord: float,
    density: float = 1.225,
    dynamic_viscosity: float = 1.8e-5,
    min_cells: int = 10000,
) -> dict:
    """Validate mesh suitability for CFD simulation.

    Checks total cell count, y+ estimates, and quality metrics across all
    blocks. Returns a summary with warnings for conditions that would
    produce unphysical results.

    Args:
        mesh: MultiBlockMesh object with .blocks attribute.
        velocity: Freestream velocity (m/s).
        chord: Physical blade chord length (m).
        density: Fluid density (kg/m^3).
        dynamic_viscosity: Dynamic viscosity (Pa.s).
        min_cells: Minimum acceptable total cell count.

    Returns:
        Dictionary with 'total_cells', 'estimated_yplus', 'warnings' list,
        'ogrid_recommendation', and per-block quality summaries.
    """
    warnings = []
    total_cells = mesh.total_cells

    if total_cells < min_cells:
        warnings.append(
            f"Total cells ({total_cells}) below minimum ({min_cells}). "
            "Results may be mesh-dependent."
        )

    # Estimate y+ from the O-grid blocks (those with 'blade' patch)
    yplus_est = None
    for block in mesh.blocks:
        if "blade" in block.patches.values():
            # First cell height ≈ distance from blade wall to first interior point
            # In the O-grid, j=0 is blade, j=1 is first interior
            ni, nj = block.points.shape[0], block.points.shape[1]
            if nj >= 2:
                wall_pts = block.points[:, 0, :]
                first_pts = block.points[:, 1, :]
                heights = np.linalg.norm(first_pts - wall_pts, axis=1)
                first_cell_h = float(np.median(heights))
                yplus_est = estimate_yplus(
                    first_cell_h, density, velocity, dynamic_viscosity, chord
                )
                break

    if yplus_est is not None:
        if yplus_est < 1.0:
            warnings.append(
                f"Estimated y+ = {yplus_est:.2f} (< 1). Ensure wall-resolved "
                "turbulence model or refine O-grid."
            )
        elif yplus_est > 300.0:
            warnings.append(
                f"Estimated y+ = {yplus_est:.1f} (> 300). Too coarse for wall "
                "functions. Increase O-grid resolution."
            )

    # Check Mach number
    speed_of_sound = 343.0  # m/s at STP
    mach = velocity / speed_of_sound
    if mach > 0.3:
        warnings.append(
            f"Mach = {mach:.2f} > 0.3. Use compressible solver (rhoSimpleFoam)."
        )

    # Check profile vs pitch scale (chord should be same order as pitch)
    # This catches the unit-chord-in-small-passage bug
    block_quality = []
    for block in mesh.blocks:
        report = mesh_quality_report(block.points)
        block_quality.append({"name": block.name, **report})
        if report["aspect_ratio_max"] > 1000:
            warnings.append(
                f"Block '{block.name}': aspect ratio {report['aspect_ratio_max']:.0f} "
                "> 1000. Likely geometry scale mismatch."
            )

    # Compute recommended O-grid parameters
    ogrid_rec = auto_first_cell_height(
        velocity=velocity,
        chord=chord,
        density=density,
        dynamic_viscosity=dynamic_viscosity,
        target_yplus=30.0,
    )

    return {
        "total_cells": total_cells,
        "estimated_yplus": yplus_est,
        "mach_number": mach,
        "warnings": warnings,
        "ogrid_recommendation": ogrid_rec,
        "block_quality": block_quality,
    }


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
