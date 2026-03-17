"""New project wizard dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QComboBox, QSpinBox, QDialogButtonBox, QGroupBox,
)


class NewProjectDialog(QDialog):
    """Dialog for creating a new turbomachine project."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Machine config
        machine_group = QGroupBox("Machine Configuration")
        form = QFormLayout(machine_group)

        self.name_edit = QLineEdit("NewMachine")
        form.addRow("Project Name:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Axial Compressor", "Axial Turbine",
                                   "Radial Compressor", "Radial Turbine", "Cascade"])
        form.addRow("Machine Type:", self.type_combo)

        self.n_rows = QSpinBox()
        self.n_rows.setRange(1, 20)
        self.n_rows.setValue(1)
        form.addRow("Blade Rows:", self.n_rows)

        self.n_profiles = QSpinBox()
        self.n_profiles.setRange(2, 50)
        self.n_profiles.setValue(3)
        form.addRow("Profiles per Row:", self.n_profiles)

        layout.addWidget(machine_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
