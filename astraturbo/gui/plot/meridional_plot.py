"""Meridional (S2m) plane plot widget."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class MeridionalPlot(QWidget):
    """Displays the meridional view (z vs r) with hub/shroud contours
    and optional SCM mesh overlay."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self._pw = pg.PlotWidget()
            self._pw.setBackground("w")
            self._pw.setLabel("bottom", "z (axial)")
            self._pw.setLabel("left", "r (radial)")
            self._pw.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self._pw)
        else:
            self._pw = None

    def plot_contour(
        self, points: NDArray[np.float64], name: str = "hub",
        color: str = "k", width: int = 2,
    ) -> None:
        if self._pw is None:
            return
        self._pw.plot(
            points[:, 0], points[:, 1],
            pen=pg.mkPen(color, width=width), name=name,
        )

    def plot_mesh_block(self, block: NDArray[np.float64], color: str = "gray") -> None:
        """Overlay a mesh block as grid lines."""
        if self._pw is None:
            return
        ni, nj = block.shape[0], block.shape[1]
        pen = pg.mkPen(color, width=0.5)
        # Streamwise lines (constant j)
        for j in range(nj):
            self._pw.plot(block[:, j, 0], block[:, j, 1], pen=pen)
        # Quasi-orthogonal lines (constant i)
        for i in range(ni):
            self._pw.plot(block[i, :, 0], block[i, :, 1], pen=pen)

    def clear(self) -> None:
        if self._pw is not None:
            self._pw.clear()
