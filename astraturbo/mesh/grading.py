"""Edge grading and clustering for structured mesh generation.

Translates edge grading ratios into point distributions on arbitrary
Polyline and Arc curves. This is the key capability that was missing:

  "Edge grading to be translated into clustering information
   and project it on to new Polyline and Arc."
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_graded_parameters(
    n_points: int, grading_ratio: float = 1.0
) -> NDArray[np.float64]:
    """Compute graded parameter distribution in [0, 1].

    Args:
        n_points: Number of points (including endpoints).
        grading_ratio: Ratio of last cell size to first cell size.
            1.0 = uniform
            > 1.0 = cells grow from start to end
            < 1.0 = cells shrink from start to end

    Returns:
        (n_points,) array of parameter values in [0, 1].
    """
    if n_points < 2:
        return np.array([0.0])

    n_cells = n_points - 1

    if abs(grading_ratio - 1.0) < 1e-10:
        return np.linspace(0, 1, n_points)

    # Geometric series: cell sizes are s, s*r, s*r^2, ..., s*r^(n-1)
    # where r = grading_ratio^(1/(n-1))
    r = grading_ratio ** (1.0 / (n_cells - 1)) if n_cells > 1 else 1.0
    sizes = r ** np.arange(n_cells)
    cumulative = np.zeros(n_points)
    cumulative[1:] = np.cumsum(sizes)
    cumulative /= cumulative[-1]
    return cumulative


def compute_double_sided_grading(
    n_points: int,
    start_grading: float = 1.0,
    end_grading: float = 1.0,
) -> NDArray[np.float64]:
    """Compute double-sided grading that clusters toward both ends.

    Useful for passage channels where refinement is needed at
    both the inlet and outlet.

    Args:
        n_points: Total number of points.
        start_grading: Grading ratio at the start (< 1 = finer at start).
        end_grading: Grading ratio at the end (< 1 = finer at end).

    Returns:
        (n_points,) array of parameter values in [0, 1].
    """
    n_half = n_points // 2
    n_rest = n_points - n_half

    # First half: grade from start
    t1 = compute_graded_parameters(n_half + 1, start_grading)[:n_half]
    t1 = t1 * 0.5  # Scale to [0, 0.5]

    # Second half: grade from end (reversed)
    t2 = compute_graded_parameters(n_rest + 1, end_grading)
    t2 = 1.0 - (1.0 - t2) * 0.5  # Scale to [0.5, 1.0]
    t2 = t2[1:]  # Skip the duplicate midpoint

    return np.concatenate([t1, t2])


def compute_boundary_layer_grading(
    n_points: int,
    first_cell_height: float,
    total_thickness: float,
    growth_rate: float = 1.2,
) -> NDArray[np.float64]:
    """Compute grading for boundary layer mesh resolution.

    Generates a parameter distribution where the first cell is
    a specific height (for y+ control) and cells grow geometrically.

    Args:
        n_points: Number of points in the wall-normal direction.
        first_cell_height: Height of the first cell (near wall).
        total_thickness: Total thickness of the layer.
        growth_rate: Geometric growth rate (> 1.0).

    Returns:
        (n_points,) array of parameter values in [0, 1].
    """
    n_cells = n_points - 1
    if n_cells < 1:
        return np.array([0.0])

    # Build cell sizes: h1, h1*r, h1*r^2, ...
    sizes = first_cell_height * growth_rate ** np.arange(n_cells)

    # If total exceeds target, scale down
    total_computed = np.sum(sizes)
    if total_computed > 0:
        sizes *= total_thickness / total_computed

    cumulative = np.zeros(n_points)
    cumulative[1:] = np.cumsum(sizes)
    cumulative /= cumulative[-1]
    return cumulative


def project_grading_onto_polyline(
    polyline,
    n_points: int,
    grading_ratio: float = 1.0,
):
    """Project a grading distribution onto a Polyline.

    Resamples the polyline at graded parameter values.

    Args:
        polyline: A Polyline object.
        n_points: Number of output points.
        grading_ratio: Edge grading ratio.

    Returns:
        New Polyline with graded point distribution.
    """
    t_graded = compute_graded_parameters(n_points, grading_ratio)
    params = polyline.normalized_parameters()
    new_pts = np.empty((n_points, polyline.dim), dtype=np.float64)
    for d in range(polyline.dim):
        new_pts[:, d] = np.interp(t_graded, params, polyline.points[:, d])

    from .polyline import Polyline
    return Polyline(new_pts, polyline.start_vertex, polyline.end_vertex)


def project_grading_onto_arc(
    arc,
    n_points: int,
    grading_ratio: float = 1.0,
):
    """Project a grading distribution onto an Arc.

    Converts the arc to a high-res polyline, then resamples with grading.

    Args:
        arc: An Arc object.
        n_points: Number of output points.
        grading_ratio: Edge grading ratio.

    Returns:
        Polyline with graded point distribution along the arc.
    """
    # Convert arc to dense polyline first
    dense_pl = arc.to_polyline(max(n_points * 10, 500))
    return project_grading_onto_polyline(dense_pl, n_points, grading_ratio)


def project_boundary_layer_onto_polyline(
    polyline,
    n_points: int,
    first_cell_height: float,
    growth_rate: float = 1.2,
):
    """Project boundary layer grading onto a Polyline.

    Used for the wall-normal direction in O-grid meshes where y+
    control is required.

    Args:
        polyline: A Polyline (typically a radial line from blade to O-grid outer).
        n_points: Number of output points.
        first_cell_height: First cell height at the wall.
        growth_rate: Geometric growth rate.

    Returns:
        New Polyline with boundary layer distribution.
    """
    total_length = polyline.total_length()
    t_bl = compute_boundary_layer_grading(
        n_points, first_cell_height, total_length, growth_rate
    )
    params = polyline.normalized_parameters()
    new_pts = np.empty((n_points, polyline.dim), dtype=np.float64)
    for d in range(polyline.dim):
        new_pts[:, d] = np.interp(t_bl, params, polyline.points[:, d])

    from .polyline import Polyline
    return Polyline(new_pts, polyline.start_vertex, polyline.end_vertex)


@staticmethod
def compute_openfoam_grading(
    n_cells: int, grading_ratio: float
) -> dict:
    """Convert grading to OpenFOAM blockMeshDict simpleGrading format.

    Args:
        n_cells: Number of cells.
        grading_ratio: Last/first cell size ratio.

    Returns:
        Dict with 'nCells' and 'grading' for blockMeshDict.
    """
    return {
        "nCells": n_cells,
        "simpleGrading": grading_ratio,
    }
