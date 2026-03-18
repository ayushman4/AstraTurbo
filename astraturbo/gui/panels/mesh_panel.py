"""Mesh configuration panel."""

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
    QTextEdit,
)


class MeshPanel(QWidget):
    """Panel for mesh generation parameters and quality display."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        # Mesh type selector
        type_group = QGroupBox("Mesh Type")
        type_layout = QFormLayout(type_group)
        self._mesh_type = QComboBox()
        self._mesh_type.addItems(["SCM (S2m Plane)", "O-Grid (Blade Passage)"])
        type_layout.addRow("Type:", self._mesh_type)
        layout.addWidget(type_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)

        self._n_axial = QSpinBox()
        self._n_axial.setRange(5, 500)
        self._n_axial.setValue(30)
        params_layout.addRow("Axial Cells:", self._n_axial)

        self._n_radial = QSpinBox()
        self._n_radial.setRange(5, 200)
        self._n_radial.setValue(20)
        params_layout.addRow("Radial Cells:", self._n_radial)

        self._grading = QDoubleSpinBox()
        self._grading.setRange(0.1, 10.0)
        self._grading.setValue(1.0)
        params_layout.addRow("Grading:", self._grading)

        layout.addWidget(params_group)

        # Generate button
        self._generate_btn = QPushButton("Generate Mesh")
        layout.addWidget(self._generate_btn)

        # Quality report
        self._quality_text = QTextEdit()
        self._quality_text.setReadOnly(True)
        self._quality_text.setMaximumHeight(120)
        self._quality_text.setPlaceholderText("Mesh quality report will appear here...")
        layout.addWidget(self._quality_text)

        layout.addStretch()

    def show_quality_report(self, report: dict) -> None:
        """Display a mesh quality report."""
        def _fmt(key: str, precision: int = 2) -> str:
            val = report.get(key)
            if val is None:
                return "N/A"
            return f"{val:.{precision}f}"

        lines = [
            f"Cells: {report.get('n_cells', 'N/A')}",
            f"Points: {report.get('n_points', 'N/A')}",
            f"Max Aspect Ratio: {_fmt('aspect_ratio_max')}",
            f"Mean Aspect Ratio: {_fmt('aspect_ratio_mean')}",
            f"Max Skewness: {_fmt('skewness_max', 3)}",
            f"Mean Skewness: {_fmt('skewness_mean', 3)}",
        ]
        self._quality_text.setPlainText("\n".join(lines))
