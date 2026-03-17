"""Blade section extraction at a given span.

Provides utilities to extract blade profile cross-sections at arbitrary
span positions from the 3D blade surface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_blade_section(
    blade_surface, span: float, n_points: int = 200
) -> NDArray[np.float64]:
    """Extract a blade section at a given span position.

    Args:
        blade_surface: geomdl BSpline.Surface.
        span: Normalized span position (0 = hub, 1 = shroud).
        n_points: Number of points in the extracted section.

    Returns:
        (N, 3) array of section points.
    """
    params = np.linspace(0, 1, n_points)
    pts = [blade_surface.evaluate_single((span, v)) for v in params]
    return np.array(pts, dtype=np.float64)
