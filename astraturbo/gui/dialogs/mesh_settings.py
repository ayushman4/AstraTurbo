"""Mesh settings dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QSpinBox,
    QDoubleSpinBox, QDialogButtonBox, QGroupBox, QComboBox,
)

from ...mesh import SCMMeshConfig, OGridMeshConfig


class MeshSettingsDialog(QDialog):
    """Dialog for configuring mesh generation parameters."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mesh Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Mesh type
        type_group = QGroupBox("Mesh Type")
        type_form = QFormLayout(type_group)
        self.mesh_type = QComboBox()
        self.mesh_type.addItems(["SCM (S2m Plane)", "O-Grid (Blade Passage)"])
        type_form.addRow("Type:", self.mesh_type)
        layout.addWidget(type_group)

        # SCM settings
        scm_group = QGroupBox("SCM Parameters")
        scm_form = QFormLayout(scm_group)
        self.scm_inlet = QSpinBox(); self.scm_inlet.setRange(3, 200); self.scm_inlet.setValue(15)
        self.scm_blade = QSpinBox(); self.scm_blade.setRange(5, 500); self.scm_blade.setValue(30)
        self.scm_outlet = QSpinBox(); self.scm_outlet.setRange(3, 200); self.scm_outlet.setValue(15)
        self.scm_radial = QSpinBox(); self.scm_radial.setRange(3, 100); self.scm_radial.setValue(20)
        scm_form.addRow("Inlet Axial Cells:", self.scm_inlet)
        scm_form.addRow("Blade Axial Cells:", self.scm_blade)
        scm_form.addRow("Outlet Axial Cells:", self.scm_outlet)
        scm_form.addRow("Radial Cells:", self.scm_radial)
        layout.addWidget(scm_group)

        # O-grid settings
        ogrid_group = QGroupBox("O-Grid Parameters")
        ogrid_form = QFormLayout(ogrid_group)
        self.ogrid_normal = QSpinBox(); self.ogrid_normal.setRange(3, 100); self.ogrid_normal.setValue(10)
        self.ogrid_wrap = QSpinBox(); self.ogrid_wrap.setRange(10, 200); self.ogrid_wrap.setValue(40)
        self.ogrid_thickness = QDoubleSpinBox(); self.ogrid_thickness.setRange(0.001, 1.0); self.ogrid_thickness.setValue(0.01)
        ogrid_form.addRow("Wall-Normal Cells:", self.ogrid_normal)
        ogrid_form.addRow("Blade Wrap Cells:", self.ogrid_wrap)
        ogrid_form.addRow("O-Grid Thickness:", self.ogrid_thickness)
        layout.addWidget(ogrid_group)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_scm_config(self) -> SCMMeshConfig:
        return SCMMeshConfig(
            n_inlet_axial=self.scm_inlet.value(),
            n_blade_axial=self.scm_blade.value(),
            n_outlet_axial=self.scm_outlet.value(),
            n_radial=self.scm_radial.value(),
        )

    def get_ogrid_config(self) -> OGridMeshConfig:
        return OGridMeshConfig(
            n_ogrid_normal=self.ogrid_normal.value(),
            n_blade_wrap=self.ogrid_wrap.value(),
            ogrid_thickness=self.ogrid_thickness.value(),
        )
