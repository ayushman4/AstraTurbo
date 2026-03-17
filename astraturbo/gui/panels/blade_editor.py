"""Blade editor panel for 3D blade row parameters."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QLabel,
)


class BladeEditorPanel(QWidget):
    """Editor for 3D blade row parameters (stacking, hub/shroud, etc.)."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        # Blade row properties
        row_group = QGroupBox("Blade Row")
        row_layout = QFormLayout(row_group)

        self._n_blades = QSpinBox()
        self._n_blades.setRange(1, 200)
        self._n_blades.setValue(20)
        row_layout.addRow("Number of Blades:", self._n_blades)

        self._omega = QDoubleSpinBox()
        self._omega.setRange(0, 100000)
        self._omega.setSuffix(" rad/s")
        row_layout.addRow("Angular Velocity:", self._omega)

        self._stacking_mode = QComboBox()
        self._stacking_mode.addItems(["Axial (Mode 0)", "Radial (Mode 1)", "Cascade (Mode 2)"])
        row_layout.addRow("Stacking Mode:", self._stacking_mode)

        layout.addWidget(row_group)

        # Span positions
        span_group = QGroupBox("Span Profiles")
        span_layout = QFormLayout(span_group)

        self._n_profiles = QSpinBox()
        self._n_profiles.setRange(2, 50)
        self._n_profiles.setValue(3)
        span_layout.addRow("Number of Profiles:", self._n_profiles)

        layout.addWidget(span_group)

        # Compute button
        self._compute_btn = QPushButton("Compute 3D Blade")
        layout.addWidget(self._compute_btn)

        layout.addStretch()
