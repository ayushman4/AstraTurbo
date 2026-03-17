"""Export format selection dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox,
    QLineEdit, QPushButton, QFileDialog, QDialogButtonBox,
    QGroupBox,
)


class ExportDialog(QDialog):
    """Dialog for selecting export format and destination."""

    FORMATS = {
        "CGNS (.cgns)": "cgns",
        "OpenFOAM blockMeshDict": "openfoam",
        "STEP (.step)": "step",
        "STL (.stl)": "stl",
        "VTK (.vtk)": "vtk",
        "SU2 (.su2)": "su2",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        group = QGroupBox("Export Settings")
        form = QFormLayout(group)

        self.format_combo = QComboBox()
        self.format_combo.addItems(self.FORMATS.keys())
        form.addRow("Format:", self.format_combo)

        path_layout = QVBoxLayout()
        self.path_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        form.addRow("Output Path:", path_layout)

        layout.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_format(self) -> str:
        return self.FORMATS[self.format_combo.currentText()]

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export to", "")
        if path:
            self.path_edit.setText(path)
