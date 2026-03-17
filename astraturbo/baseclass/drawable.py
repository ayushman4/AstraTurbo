"""Mixin for objects that can provide 2D/3D visualization data.

This version is visualization-framework agnostic -- it just provides
the data arrays; the GUI layer handles rendering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class Drawable:
    """Mixin that marks an object as having visual representation.

    Subclasses should implement the relevant methods to provide
    plot data for the GUI layer.
    """

    def get_plot_data_2d(self) -> NDArray[np.float64] | None:
        """Return 2D plot data as (N, 2) array of [x, y] points.

        Returns None if no 2D representation is available.
        """
        return None

    def get_plot_data_3d(self) -> NDArray[np.float64] | None:
        """Return 3D plot data as (N, 3) array of [x, y, z] points.

        Returns None if no 3D representation is available.
        """
        return None

    def get_display_properties(self) -> dict[str, Any]:
        """Return display hints for the GUI (color, linewidth, etc.).

        Override to customize appearance.
        """
        return {
            "color": "black",
            "linewidth": 1.0,
            "linestyle": "-",
            "visible": True,
        }
