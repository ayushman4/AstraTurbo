"""2D profile editor panel.

Displays and allows interactive editing of 2D blade profiles
(camber lines, thickness distributions, superposition results).
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QComboBox, QHBoxLayout

import numpy as np

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class ProfileEditorPanel(QWidget):
    """Interactive 2D profile editor with live plot."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Camber:"))
        self._camber_combo = QComboBox()
        self._camber_combo.addItems([
            "Circular Arc", "Quadratic", "Cubic", "Quartic",
            "Joukowski", "NACA 2-Digit", "NACA 65", "NURBS",
        ])
        self._camber_combo.setCurrentIndex(6)  # Default: NACA 65
        controls.addWidget(self._camber_combo)

        controls.addWidget(QLabel("Thickness:"))
        self._thick_combo = QComboBox()
        self._thick_combo.addItems([
            "NACA 4-Digit", "NACA 65-Series", "Joukowski", "Elliptic",
        ])
        self._thick_combo.setCurrentIndex(0)  # Default: NACA 4-Digit
        controls.addWidget(self._thick_combo)
        controls.addStretch()
        layout.addLayout(controls)

        # Plot area
        if HAS_PYQTGRAPH:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setBackground("w")
            self._plot_widget.setAspectLocked(True)
            self._plot_widget.setLabel("bottom", "x/c")
            self._plot_widget.setLabel("left", "y/c")
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self._plot_widget)

            # Plot items
            self._profile_curve = self._plot_widget.plot(
                pen=pg.mkPen("b", width=2)
            )
            from PySide6.QtCore import Qt as QtCore_Qt
            self._camber_curve = self._plot_widget.plot(
                pen=pg.mkPen("r", width=1, style=QtCore_Qt.DashLine)
            )
        else:
            layout.addWidget(QLabel(
                "Install pyqtgraph for interactive plots:\n"
                "  pip install pyqtgraph"
            ))
            self._plot_widget = None

        self._camber_combo.currentIndexChanged.connect(self._update_plot)
        self._thick_combo.currentIndexChanged.connect(self._update_plot)

        # Draw default profile
        self._update_plot()

    def _update_plot(self) -> None:
        """Regenerate and redraw the profile."""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return

        from astraturbo.profile import Superposition
        from astraturbo.camberline import create_camberline
        from astraturbo.thickness import create_thickness

        camber_map = {
            0: "circular_arc", 1: "quadratic", 2: "cubic", 3: "quartic",
            4: "joukowski", 5: "naca2digit", 6: "naca65", 7: "nurbs",
        }
        thick_map = {
            0: "naca4digit", 1: "naca65", 2: "joukowski", 3: "elliptic",
        }

        try:
            cl = create_camberline(camber_map[self._camber_combo.currentIndex()])
            td = create_thickness(thick_map[self._thick_combo.currentIndex()])
            profile = Superposition(cl, td)

            arr = profile.as_array()
            self._profile_curve.setData(arr[:, 0], arr[:, 1])

            camber = cl.as_array()
            self._camber_curve.setData(camber[:, 0], camber[:, 1])
        except Exception:
            pass

    def set_profile(self, profile) -> None:
        """Display a specific profile object."""
        if not HAS_PYQTGRAPH or self._plot_widget is None:
            return
        try:
            arr = profile.as_array()
            self._profile_curve.setData(arr[:, 0], arr[:, 1])
        except Exception:
            pass
