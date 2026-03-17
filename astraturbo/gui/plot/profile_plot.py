"""2D profile plot widget using pyqtgraph."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class ProfilePlot(QWidget):
    """Standalone 2D profile plot widget."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self._pw = pg.PlotWidget()
            self._pw.setBackground("w")
            self._pw.setAspectLocked(True)
            self._pw.setLabel("bottom", "x/c")
            self._pw.setLabel("left", "y/c")
            self._pw.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self._pw)
            self._curves: dict[str, pg.PlotDataItem] = {}
        else:
            self._pw = None

    def plot_profile(
        self, data: NDArray[np.float64], name: str = "profile",
        color: str = "b", width: int = 2,
    ) -> None:
        if self._pw is None:
            return
        if name in self._curves:
            self._curves[name].setData(data[:, 0], data[:, 1])
        else:
            curve = self._pw.plot(
                data[:, 0], data[:, 1],
                pen=pg.mkPen(color, width=width), name=name,
            )
            self._curves[name] = curve

    def clear(self) -> None:
        if self._pw is None:
            return
        self._pw.clear()
        self._curves.clear()
