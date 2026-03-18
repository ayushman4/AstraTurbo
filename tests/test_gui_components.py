"""GUI component tests using pytest-qt.

Tests MainWindow state management, panel logic, dialogs, and data flow
without requiring a visible display (uses offscreen QApplication).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip entire module if PySide6 is not installed
PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication for the session (required by all Qt widgets)."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def main_window(qapp):
    """Create and return a MainWindow instance."""
    from astraturbo.gui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.close()


# ────────────────────────────────────────────────────────────────
# 1. MainWindow initialization
# ────────────────────────────────────────────────────────────────

class TestMainWindowInit:
    """Test MainWindow creates correct initial state."""

    def test_window_title(self, main_window):
        assert "AstraTurbo" in main_window.windowTitle()

    def test_default_machine_exists(self, main_window):
        assert main_window._machine is not None

    def test_default_row_exists(self, main_window):
        assert main_window._current_row is not None

    def test_default_profiles_created(self, main_window):
        row = main_window._current_row
        assert len(row.profiles) >= 2

    def test_tabs_created(self, main_window):
        tabs = main_window._tabs
        assert tabs.count() >= 4  # Profile, Blade, 3D Viewer, AI

    def test_menu_bar_exists(self, main_window):
        menubar = main_window.menuBar()
        assert menubar is not None
        # Check key menus exist
        menus = [a.text() for a in menubar.actions()]
        menu_text = " ".join(menus).lower()
        assert "file" in menu_text
        assert "compute" in menu_text
        assert "tools" in menu_text


# ────────────────────────────────────────────────────────────────
# 2. Project management
# ────────────────────────────────────────────────────────────────

class TestProjectManagement:
    """Test new/save/load project workflows."""

    def test_new_project_resets_state(self, main_window):
        # Modify state first
        main_window._project_path = "/tmp/fake.yaml"
        main_window._last_mesh = "fake_mesh"

        main_window._new_project()

        assert main_window._project_path is None
        assert main_window._last_mesh is None
        assert main_window._machine is not None
        assert main_window._current_row is not None

    def test_save_project(self, main_window, tmp_path):
        path = tmp_path / "test_project.yaml"
        main_window._project_path = str(path)
        main_window._save_project()
        assert path.exists()

    def test_open_project(self, main_window, tmp_path):
        # Save first
        path = tmp_path / "test_project.yaml"
        main_window._project_path = str(path)
        main_window._save_project()

        # Reset and reload
        main_window._new_project()
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=(str(path), "YAML Files")):
            main_window._open_project()

        assert main_window._machine is not None


# ────────────────────────────────────────────────────────────────
# 3. Blade row management
# ────────────────────────────────────────────────────────────────

class TestBladeRowManagement:
    """Test add/remove blade row operations."""

    def test_add_blade_row(self, main_window):
        initial = len(main_window._machine.blade_rows)
        main_window._add_blade_row()
        assert len(main_window._machine.blade_rows) == initial + 1

    def test_add_profile_to_row(self, main_window):
        initial = len(main_window._current_row.profiles)
        main_window._add_profile_to_row()
        assert len(main_window._current_row.profiles) == initial + 1


# ────────────────────────────────────────────────────────────────
# 4. Mesh panel
# ────────────────────────────────────────────────────────────────

class TestMeshPanel:
    """Test MeshPanel widget behavior."""

    def test_quality_report_with_data(self, qapp):
        from astraturbo.gui.panels.mesh_panel import MeshPanel

        panel = MeshPanel()
        report = {
            "n_cells": 1000,
            "n_points": 1200,
            "aspect_ratio_max": 5.2,
            "aspect_ratio_mean": 1.8,
            "skewness_max": 0.45,
            "skewness_mean": 0.12,
        }
        panel.show_quality_report(report)
        text = panel._quality_text.toPlainText()
        assert "1000" in text
        assert "5.20" in text
        assert "0.450" in text

    def test_quality_report_with_missing_keys(self, qapp):
        from astraturbo.gui.panels.mesh_panel import MeshPanel

        panel = MeshPanel()
        # Partial report — should not crash
        report = {"n_cells": 500}
        panel.show_quality_report(report)
        text = panel._quality_text.toPlainText()
        assert "500" in text
        assert "N/A" in text

    def test_default_widget_values(self, qapp):
        from astraturbo.gui.panels.mesh_panel import MeshPanel

        panel = MeshPanel()
        assert panel._n_axial.value() == 30
        assert panel._n_radial.value() == 20
        assert panel._grading.value() == 1.0


# ────────────────────────────────────────────────────────────────
# 5. Properties panel
# ────────────────────────────────────────────────────────────────

class TestPropertiesPanel:
    """Test PropertiesPanel binds to ATObjects correctly."""

    def test_set_object_populates_form(self, qapp):
        from astraturbo.gui.panels.properties_panel import PropertiesPanel
        from astraturbo.camberline import CircularArc

        panel = PropertiesPanel()
        obj = CircularArc()
        panel.set_object(obj)

        # Should have created some form widgets
        layout = panel._form_layout
        assert layout.rowCount() > 0

    def test_set_none_clears_object(self, qapp):
        from astraturbo.gui.panels.properties_panel import PropertiesPanel
        from astraturbo.camberline import CircularArc

        panel = PropertiesPanel()
        panel.set_object(CircularArc())
        panel.set_object(None)
        # Object should be cleared even if layout rows persist
        assert panel._current_obj is None


# ────────────────────────────────────────────────────────────────
# 6. Machine tree panel
# ────────────────────────────────────────────────────────────────

class TestMachineTreePanel:
    """Test tree panel reflects machine structure."""

    def test_tree_populated(self, qapp):
        from astraturbo.gui.panels.machine_tree import MachineTreePanel
        from astraturbo.machine import TurboMachine
        from astraturbo.blade import BladeRow

        machine = TurboMachine()
        row = BladeRow()
        row.name = "Rotor1"
        machine.add_blade_row(row)

        panel = MachineTreePanel(machine)
        root = panel._tree.invisibleRootItem()
        assert root.childCount() >= 1  # At least machine root item

    def test_set_machine_refreshes(self, qapp):
        from astraturbo.gui.panels.machine_tree import MachineTreePanel
        from astraturbo.machine import TurboMachine
        from astraturbo.blade import BladeRow

        machine1 = TurboMachine()
        r1 = BladeRow(); r1.name = "R1"
        machine1.add_blade_row(r1)

        machine2 = TurboMachine()
        s1 = BladeRow(); s1.name = "S1"
        r2 = BladeRow(); r2.name = "R2"
        machine2.add_blade_row(s1)
        machine2.add_blade_row(r2)

        panel = MachineTreePanel(machine1)
        panel.set_machine(machine2)
        root = panel._tree.invisibleRootItem()
        assert root.childCount() >= 1


# ────────────────────────────────────────────────────────────────
# 7. Dialogs
# ────────────────────────────────────────────────────────────────

class TestDialogs:
    """Test dialog widgets create without errors."""

    def test_new_project_dialog(self, qapp):
        from astraturbo.gui.dialogs.new_project import NewProjectDialog

        dialog = NewProjectDialog()
        assert dialog.name_edit.text() == "NewMachine"
        assert dialog.n_rows.value() == 1
        assert dialog.n_profiles.value() == 3
        assert dialog.type_combo.count() >= 4

    def test_mesh_settings_dialog(self, qapp):
        from astraturbo.gui.dialogs.mesh_settings import MeshSettingsDialog

        dialog = MeshSettingsDialog()
        assert dialog.scm_blade.value() == 30
        assert dialog.ogrid_normal.value() == 10

        scm_config = dialog.get_scm_config()
        assert scm_config.n_blade_axial == 30

        ogrid_config = dialog.get_ogrid_config()
        assert ogrid_config.n_ogrid_normal == 10


# ────────────────────────────────────────────────────────────────
# 8. AI chat panel
# ────────────────────────────────────────────────────────────────

class TestAIChatPanel:
    """Test AI chat panel logic (without API calls)."""

    def test_html_escaping(self, qapp):
        from astraturbo.gui.panels.ai_chat import AIChatPanel

        panel = AIChatPanel()
        escaped = panel._escape_html("<script>alert('xss')</script>")
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_no_api_key_shows_error(self, qapp, monkeypatch):
        from astraturbo.gui.panels.ai_chat import AIChatPanel

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        panel = AIChatPanel()
        result = panel._ensure_assistant()
        assert result is False

    def test_initial_state_not_busy(self, qapp):
        from astraturbo.gui.panels.ai_chat import AIChatPanel

        panel = AIChatPanel()
        assert panel._busy is False


# ────────────────────────────────────────────────────────────────
# 9. Profile editor panel
# ────────────────────────────────────────────────────────────────

class TestProfileEditorPanel:
    """Test profile editor creates valid widget state."""

    def test_camber_combo_populated(self, qapp):
        from astraturbo.gui.panels.profile_editor import ProfileEditorPanel

        panel = ProfileEditorPanel()
        assert panel._camber_combo.count() >= 3  # At least circular_arc, naca65, etc.

    def test_thickness_combo_populated(self, qapp):
        from astraturbo.gui.panels.profile_editor import ProfileEditorPanel

        panel = ProfileEditorPanel()
        assert panel._thick_combo.count() >= 2


# ────────────────────────────────────────────────────────────────
# 10. Blade editor panel
# ────────────────────────────────────────────────────────────────

class TestBladeEditorPanel:
    """Test blade editor default values."""

    def test_defaults(self, qapp):
        from astraturbo.gui.panels.blade_editor import BladeEditorPanel

        panel = BladeEditorPanel()
        assert panel._n_blades.value() >= 1
        assert panel._omega.value() >= 0
        assert panel._n_profiles.value() >= 2


# ────────────────────────────────────────────────────────────────
# 11. Computation actions (mocked)
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# 11. Computation actions (headless-safe after lazy VTK init)
# ────────────────────────────────────────────────────────────────

class TestComputeActions:
    """Test compute-related state on MainWindow.

    Note: Methods like _compute_blade() trigger QMessageBox which forces
    a repaint of pyqtgraph's OpenGL-backed PlotWidget, causing segfaults
    in offscreen mode. Those code paths are tested via CLI integration
    tests instead. Here we test the lazy-init pattern.
    """

    def test_viewer_lazy_init(self, main_window):
        """3D viewer should be None until explicitly created."""
        assert main_window._point_cloud_viewer is None

    def test_viewer_tab_exists(self, main_window):
        """3D Viewer tab should be present (as a placeholder)."""
        tab_labels = [main_window._tabs.tabText(i) for i in range(main_window._tabs.count())]
        assert "3D Viewer" in tab_labels

    def test_last_mesh_initially_none(self, main_window):
        """No mesh should exist before any computation."""
        assert main_window._last_mesh is None
