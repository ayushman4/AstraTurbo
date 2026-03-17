"""Point cloud viewer widget.

Displays imported mesh point clouds using pyqtgraph's OpenGL scatter plot.
Falls back to a 2D projection (X-Y, X-Z, Y-Z) if 3D is not available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox,
    QPushButton, QSlider, QGroupBox, QFormLayout,
)

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    HAS_GL = True
except ImportError:
    HAS_GL = False

try:
    import pyqtgraph as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False


class PointCloudViewer(QWidget):
    """3D/2D viewer for imported mesh point clouds."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points: NDArray[np.float64] | None = None
        self._stats: dict | None = None

        layout = QVBoxLayout(self)

        # Info bar
        self._info_label = QLabel("No points loaded")
        layout.addWidget(self._info_label)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        if HAS_GL:
            self._view_combo.addItems(["3D", "X-Y (Top)", "X-Z (Side)", "Y-Z (Front)"])
        else:
            self._view_combo.addItems(["X-Y (Top)", "X-Z (Side)", "Y-Z (Front)"])
        self._view_combo.currentIndexChanged.connect(self._update_view)
        controls.addWidget(self._view_combo)

        # Subsample slider for large point clouds
        controls.addWidget(QLabel("Detail:"))
        self._detail_slider = QSlider(Qt.Horizontal)
        self._detail_slider.setRange(1, 100)
        self._detail_slider.setValue(100)
        self._detail_slider.setMaximumWidth(150)
        self._detail_slider.valueChanged.connect(self._update_view)
        controls.addWidget(self._detail_slider)
        self._detail_label = QLabel("100%")
        controls.addWidget(self._detail_label)

        controls.addStretch()
        layout.addLayout(controls)

        # 3D view (pyqtgraph OpenGL)
        if HAS_GL:
            self._gl_widget = gl.GLViewWidget()
            self._gl_widget.setBackgroundColor("w")
            self._gl_widget.opts["distance"] = 0.5
            layout.addWidget(self._gl_widget)
            self._scatter = None
        else:
            self._gl_widget = None

        # 2D fallback view (pyqtgraph PlotWidget)
        if HAS_PG:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setBackground("w")
            self._plot_widget.setAspectLocked(True)
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self._plot_widget)
            self._scatter_2d = None
            if HAS_GL:
                self._plot_widget.hide()
        else:
            self._plot_widget = None
            layout.addWidget(QLabel(
                "Install pyqtgraph for visualization:\n  pip install pyqtgraph"
            ))

    def set_points(
        self,
        points: NDArray[np.float64],
        stats: dict | None = None,
    ) -> None:
        """Load a point cloud for display.

        Args:
            points: (N, 3) array of point coordinates.
            stats: Optional statistics dict from openfoam_points_to_cloud.
        """
        self._points = points
        self._stats = stats

        n = len(points)
        info_parts = [f"{n:,} points"]
        if stats:
            info_parts.append(
                f"X: {stats['x_range']*1000:.1f}mm  "
                f"Y: {stats['y_range']*1000:.1f}mm  "
                f"Z: {stats['z_range']*1000:.1f}mm"
            )
        self._info_label.setText("  |  ".join(info_parts))

        # Auto-reduce detail for large clouds
        if n > 50000:
            pct = max(5, int(50000 / n * 100))
            self._detail_slider.setValue(pct)
        else:
            self._detail_slider.setValue(100)

        self._update_view()

    def _get_subsampled_points(self) -> NDArray[np.float64]:
        """Return points subsampled based on the detail slider."""
        if self._points is None:
            return np.empty((0, 3))

        pct = self._detail_slider.value()
        self._detail_label.setText(f"{pct}%")

        if pct >= 100:
            return self._points

        n_target = max(100, int(len(self._points) * pct / 100))
        indices = np.linspace(0, len(self._points) - 1, n_target, dtype=int)
        return self._points[indices]

    def _update_view(self) -> None:
        """Redraw the current view."""
        if self._points is None:
            return

        pts = self._get_subsampled_points()
        view_text = self._view_combo.currentText()

        if "3D" in view_text and HAS_GL and self._gl_widget is not None:
            self._show_3d(pts)
            self._gl_widget.show()
            if self._plot_widget:
                self._plot_widget.hide()
        else:
            self._show_2d(pts, view_text)
            if self._gl_widget:
                self._gl_widget.hide()
            if self._plot_widget:
                self._plot_widget.show()

    def _show_3d(self, pts: NDArray[np.float64]) -> None:
        """Display points in 3D OpenGL view."""
        if not HAS_GL or self._gl_widget is None:
            return

        # Remove old scatter
        if self._scatter is not None:
            self._gl_widget.removeItem(self._scatter)

        # Center and scale
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        scale = np.max(np.abs(pts_centered)) or 1.0
        pts_normalized = pts_centered / scale

        # Color by Z coordinate
        z_vals = pts_normalized[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_range = z_max - z_min if (z_max - z_min) > 1e-10 else 1.0
        z_norm = (z_vals - z_min) / z_range

        colors = np.zeros((len(pts), 4), dtype=np.float32)
        colors[:, 0] = z_norm          # Red channel
        colors[:, 1] = 0.2             # Green
        colors[:, 2] = 1.0 - z_norm    # Blue channel
        colors[:, 3] = 0.6             # Alpha

        self._scatter = gl.GLScatterPlotItem(
            pos=pts_normalized.astype(np.float32),
            color=colors,
            size=1.5,
            pxMode=True,
        )
        self._gl_widget.addItem(self._scatter)

        # Add axes
        ax = gl.GLAxisItem()
        ax.setSize(0.5, 0.5, 0.5)
        self._gl_widget.addItem(ax)

        self._gl_widget.opts["distance"] = 2.0

    def _show_2d(self, pts: NDArray[np.float64], view: str) -> None:
        """Display a 2D projection of the points."""
        if not HAS_PG or self._plot_widget is None:
            return

        self._plot_widget.clear()

        if "X-Y" in view:
            x, y = pts[:, 0], pts[:, 1]
            self._plot_widget.setLabel("bottom", "X")
            self._plot_widget.setLabel("left", "Y")
        elif "X-Z" in view:
            x, y = pts[:, 0], pts[:, 2]
            self._plot_widget.setLabel("bottom", "X")
            self._plot_widget.setLabel("left", "Z")
        elif "Y-Z" in view:
            x, y = pts[:, 1], pts[:, 2]
            self._plot_widget.setLabel("bottom", "Y")
            self._plot_widget.setLabel("left", "Z")
        else:
            x, y = pts[:, 0], pts[:, 1]

        scatter = pg.ScatterPlotItem(
            x=x, y=y,
            size=1, pen=None,
            brush=pg.mkBrush(50, 50, 200, 120),
        )
        self._plot_widget.addItem(scatter)

    def clear(self) -> None:
        """Clear all displayed data."""
        self._points = None
        self._stats = None
        self._info_label.setText("No points loaded")

        if HAS_GL and self._gl_widget is not None:
            if self._scatter is not None:
                self._gl_widget.removeItem(self._scatter)
                self._scatter = None

        if HAS_PG and self._plot_widget is not None:
            self._plot_widget.clear()
