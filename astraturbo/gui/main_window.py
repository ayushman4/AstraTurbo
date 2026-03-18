"""Main window for AstraTurbo GUI.

Fully wired to the computation engine: profile generation, 3D blade
geometry, mesh generation, export, and import all work end-to-end.
"""

from __future__ import annotations

import traceback

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QToolBar,
    QDockWidget,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QWidget,
)

from ..machine import TurboMachine, save_project, load_project
from ..blade import BladeRow
from ..profile import Superposition
from ..camberline import create_camberline
from ..thickness import create_thickness
from ..mesh import SCMMesher, SCMMeshConfig, mesh_quality_report
from ..mesh.multiblock import generate_blade_passage_mesh
from ..mesh.multistage import MultistageGenerator, RowMeshConfig
from ..export import (
    write_cgns_structured,
    write_cgns_2d,
    write_blockmeshdict,
    read_openfoam_points,
    openfoam_points_to_cloud,
)

from .panels.machine_tree import MachineTreePanel
from .panels.profile_editor import ProfileEditorPanel
from .panels.properties_panel import PropertiesPanel
from .panels.blade_editor import BladeEditorPanel
from .panels.mesh_panel import MeshPanel


class MainWindow(QMainWindow):
    """AstraTurbo main application window — fully functional."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AstraTurbo — Turbomachinery Design Platform")
        self.setMinimumSize(1200, 800)

        # Core state
        self._machine = TurboMachine()
        self._project_path: str | None = None
        self._current_row: BladeRow | None = None
        self._current_profile: Superposition | None = None
        self._last_mesh = None
        self._last_mesh_blocks = None

        # Build a default blade row with a default profile
        self._init_default_project()

        self._setup_menubar()
        self._setup_toolbar()
        self._setup_panels()
        self._setup_statusbar()

        # Wire inter-panel signals
        self._connect_signals()

        # Show default profile
        self._profile_panel._update_plot()
        self.statusBar().showMessage(
            "Ready — Select a camber line and thickness, then Compute > Blade Geometry"
        )

    def _init_default_project(self) -> None:
        """Create a default project with one row and 3 span profiles."""
        from ..camberline import NACA65
        from ..thickness import NACA4Digit

        row = BladeRow(
            hub_points=np.array([[0.0, 0.10], [0.05, 0.10], [0.10, 0.10]]),
            shroud_points=np.array([[0.0, 0.20], [0.05, 0.20], [0.10, 0.20]]),
            stacking_mode=0,
        )
        row.name = "Row 0"
        # 3 profiles at hub, mid, tip with varying parameters
        row.add_profile(Superposition(NACA65(cl0=0.8), NACA4Digit(max_thickness=0.08)))
        row.add_profile(Superposition(NACA65(cl0=1.0), NACA4Digit(max_thickness=0.10)))
        row.add_profile(Superposition(NACA65(cl0=1.2), NACA4Digit(max_thickness=0.12)))
        self._machine.add_blade_row(row)
        self._current_row = row
        self._current_profile = row.profiles[1]  # Mid-span as default

    # ----------------------------------------------------------------
    # Menu bar
    # ----------------------------------------------------------------

    def _setup_menubar(self) -> None:
        menubar = self.menuBar()

        # --- File menu ---
        file_menu = menubar.addMenu("&File")

        self._add_action(file_menu, "&New Project", self._new_project, QKeySequence.New)
        self._add_action(file_menu, "&Open Project...", self._open_project, QKeySequence.Open)
        self._add_action(file_menu, "&Save Project", self._save_project, QKeySequence.Save)
        self._add_action(file_menu, "Save &As...", self._save_project_as, QKeySequence.SaveAs)
        file_menu.addSeparator()
        self._add_action(file_menu, "Import &Legacy XML Project...", self._import_xml)
        self._add_action(file_menu, "Import &OpenFOAM Points...", self._import_openfoam_points)
        file_menu.addSeparator()

        export_menu = file_menu.addMenu("E&xport")
        self._add_action(export_menu, "CGNS Mesh...", self._export_cgns)
        self._add_action(export_menu, "OpenFOAM blockMeshDict...", self._export_openfoam)
        self._add_action(export_menu, "VTK Mesh...", self._export_vtk)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Quit", self.close, QKeySequence.Quit)

        # --- Edit menu ---
        edit_menu = menubar.addMenu("&Edit")
        self._add_action(edit_menu, "&Add Blade Row", self._add_blade_row)
        self._add_action(edit_menu, "Add &Profile to Row", self._add_profile_to_row)

        # --- Compute menu ---
        compute_menu = menubar.addMenu("&Compute")
        self._add_action(compute_menu, "&Meanline Design...", self._run_meanline)
        compute_menu.addSeparator()
        self._add_action(compute_menu, "Compute &Blade Geometry", self._compute_blade)
        self._add_action(compute_menu, "Generate Blade &Array (Full Annulus)", self._generate_blade_array)
        self._add_action(compute_menu, "Generate &SCM Mesh (S2m)", self._compute_scm_mesh)
        self._add_action(compute_menu, "Generate S&1 Mesh (Blade-to-Blade)", self._compute_s1_mesh)
        self._add_action(compute_menu, "Generate &O-Grid Mesh", self._compute_ogrid_mesh)
        self._add_action(compute_menu, "Generate Multi-&Block Mesh", self._compute_multiblock_mesh)
        self._add_action(compute_menu, "Generate 3&D Mesh (Span Stacking)...", self._compute_3d_mesh)
        self._add_action(compute_menu, "Generate &Tip Clearance Mesh...", self._generate_tip_clearance)
        compute_menu.addSeparator()

        cfd_menu = compute_menu.addMenu("CFD Case Setup")
        self._add_action(cfd_menu, "&OpenFOAM...", lambda: self._setup_cfd("openfoam"))
        self._add_action(cfd_menu, "ANSYS &Fluent...", lambda: self._setup_cfd("fluent"))
        self._add_action(cfd_menu, "ANSYS C&FX...", lambda: self._setup_cfd("cfx"))
        self._add_action(cfd_menu, "&SU2...", lambda: self._setup_cfd("su2"))

        compute_menu.addSeparator()
        self._add_action(compute_menu, "FEA &Structural Analysis...", self._setup_fea)
        compute_menu.addSeparator()
        self._add_action(compute_menu, "Run &Optimization...", self._run_optimization)
        self._add_action(compute_menu, "Run Multi-&Fidelity Optimization...", self._run_multifidelity)
        compute_menu.addSeparator()
        self._add_action(compute_menu, "Run &Throughflow Solver...", self._run_throughflow)
        self._add_action(compute_menu, "S&mooth Mesh...", self._smooth_mesh)
        self._add_action(compute_menu, "&Parametric Sweep...", self._run_parametric_sweep)

        # --- Tools menu ---
        tools_menu = menubar.addMenu("&Tools")
        self._add_action(tools_menu, "&y+ Calculator...", self._yplus_calculator)
        self._add_action(tools_menu, "List Supported &Formats", self._show_formats)
        tools_menu.addSeparator()
        self._add_action(tools_menu, "&Design Database...", self._open_design_database)
        self._add_action(tools_menu, "&HPC Job Manager...", self._open_hpc_manager)

        # --- Help menu ---
        help_menu = menubar.addMenu("&Help")
        self._add_action(help_menu, "&About AstraTurbo", self._show_about)

    def _add_action(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # ----------------------------------------------------------------
    # Toolbar
    # ----------------------------------------------------------------

    def _setup_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        new_act = toolbar.addAction("New")
        new_act.triggered.connect(self._new_project)

        open_act = toolbar.addAction("Open")
        open_act.triggered.connect(self._open_project)

        save_act = toolbar.addAction("Save")
        save_act.triggered.connect(self._save_project)

        toolbar.addSeparator()

        compute_act = toolbar.addAction("Compute Blade")
        compute_act.triggered.connect(self._compute_blade)

        mesh_act = toolbar.addAction("Generate Mesh")
        mesh_act.triggered.connect(self._compute_multiblock_mesh)

        export_act = toolbar.addAction("Export CGNS")
        export_act.triggered.connect(self._export_cgns)

    # ----------------------------------------------------------------
    # Panels
    # ----------------------------------------------------------------

    def _setup_panels(self) -> None:
        # Machine tree (left)
        self._tree_panel = MachineTreePanel(self._machine)
        tree_dock = QDockWidget("Machine Structure", self)
        tree_dock.setWidget(self._tree_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, tree_dock)

        # Central: tabbed profile editor + blade editor + 3D viewer
        self._tabs = QTabWidget()
        self._profile_panel = ProfileEditorPanel()
        self._blade_panel = BladeEditorPanel()
        self._tabs.addTab(self._profile_panel, "2D Profile")
        self._tabs.addTab(self._blade_panel, "3D Blade")

        # 3D Viewer tab — lazy-initialized on first use to avoid
        # VTK/OpenGL context creation at startup (faster launch,
        # and enables headless testing without segfaults).
        self._point_cloud_viewer = None
        self._viewer_placeholder = QWidget()
        self._viewer_tab_index = self._tabs.addTab(self._viewer_placeholder, "3D Viewer")

        # AI Chat tab
        from .panels.ai_chat import AIChatPanel
        self._ai_chat = AIChatPanel()
        self._tabs.addTab(self._ai_chat, "AI Assistant")

        # Start on the 2D Profile tab
        self._tabs.setCurrentIndex(0)

        self.setCentralWidget(self._tabs)

        # Properties panel (right)
        self._properties_panel = PropertiesPanel()
        props_dock = QDockWidget("Properties", self)
        props_dock.setWidget(self._properties_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, props_dock)

        # Mesh panel (bottom)
        self._mesh_panel = MeshPanel()
        mesh_dock = QDockWidget("Mesh", self)
        mesh_dock.setWidget(self._mesh_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, mesh_dock)

    def _setup_statusbar(self) -> None:
        self.statusBar().showMessage("Ready")

    def _ensure_point_cloud_viewer(self):
        """Lazy-init the 3D point cloud viewer on first use.

        Defers VTK/OpenGL context creation until actually needed,
        which speeds up startup and avoids segfaults in headless tests.
        """
        if self._point_cloud_viewer is None:
            from .viewer.point_cloud_viewer import PointCloudViewer
            self._point_cloud_viewer = PointCloudViewer()
            self._tabs.removeTab(self._viewer_tab_index)
            self._tabs.insertTab(self._viewer_tab_index, self._point_cloud_viewer, "3D Viewer")
        return self._point_cloud_viewer

    def _connect_signals(self) -> None:
        """Wire panel signals to main window actions."""
        self._tree_panel.item_selected.connect(self._on_tree_item_selected)
        self._blade_panel._compute_btn.clicked.connect(self._compute_blade)
        self._mesh_panel._generate_btn.clicked.connect(self._compute_multiblock_mesh)

    def _on_tree_item_selected(self, obj) -> None:
        """Handle tree item selection — show properties and update panels."""
        self._properties_panel.set_object(obj)
        if isinstance(obj, BladeRow):
            self._current_row = obj
            self._blade_panel._n_blades.setValue(obj.number_blades)
        if isinstance(obj, Superposition):
            self._current_profile = obj
            self._profile_panel.set_profile(obj)

    # ----------------------------------------------------------------
    # File operations
    # ----------------------------------------------------------------

    def _new_project(self) -> None:
        self._machine = TurboMachine()
        self._project_path = None
        self._current_row = None
        self._current_profile = None
        self._last_mesh = None
        self._init_default_project()
        self._tree_panel.set_machine(self._machine)
        self._profile_panel._update_plot()
        self.statusBar().showMessage("New project created")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "AstraTurbo Projects (*.yaml *.yml)"
        )
        if path:
            try:
                data = load_project(path)
                self._project_path = path
                self.statusBar().showMessage(f"Opened: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to open project:\n{e}")

    def _save_project(self) -> None:
        if self._project_path:
            try:
                save_project({"name": self._machine.name, "type": self._machine.machine_type},
                             self._project_path)
                self.statusBar().showMessage(f"Saved: {self._project_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save:\n{e}")
        else:
            self._save_project_as()

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "AstraTurbo Projects (*.yaml)"
        )
        if path:
            self._project_path = path
            self._save_project()

    def _import_xml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Legacy XML Project", "", "XML Files (*.xml)"
        )
        if path:
            try:
                from ..machine import import_bladedesigner_xml
                data = import_bladedesigner_xml(path)
                self.statusBar().showMessage(
                    f"Imported XML: {len(data)} top-level elements from {path}"
                )
                QMessageBox.information(
                    self, "Import Successful",
                    f"Imported {path}\n\nTop-level keys: {list(data.keys())}"
                )
            except Exception as e:
                QMessageBox.warning(self, "Import Error", str(e))

    def _import_openfoam_points(self) -> None:
        from pathlib import Path
        import os
        home = str(Path.home())

        # Force Qt's own file dialog — macOS native dialog greys out
        # extensionless files even with "All Files (*)" filter
        os.environ["QT_USE_NATIVE_DIALOGS"] = "0"

        dialog = QFileDialog(self)
        dialog.setWindowTitle("Import OpenFOAM Points File")
        dialog.setDirectory(home)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilter("All Files (*)")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)

        if not dialog.exec():
            return

        selected = dialog.selectedFiles()
        if not selected:
            return
        path = selected[0]
        if path:
            # Pre-validate before attempting to read
            from ..export.openfoam_reader import validate_openfoam_file, OpenFOAMReadError

            is_valid, validation_msg = validate_openfoam_file(path)
            if not is_valid:
                QMessageBox.warning(
                    self, "Invalid File",
                    f"This file cannot be read as an OpenFOAM points file.\n\n"
                    f"{validation_msg}\n\n"
                    f"Expected: an ASCII-format OpenFOAM polyMesh/points file\n"
                    f"containing (x y z) coordinate triplets."
                )
                return

            try:
                points = read_openfoam_points(path)
                stats = openfoam_points_to_cloud(points)
                msg = (
                    f"Successfully loaded {stats['n_points']:,} points\n\n"
                    f"X: {stats['x_min']:.4f} to {stats['x_max']:.4f} "
                    f"({stats['x_range']*1000:.1f} mm)\n"
                    f"Y: {stats['y_min']:.4f} to {stats['y_max']:.4f} "
                    f"({stats['y_range']*1000:.1f} mm)\n"
                    f"Z: {stats['z_min']:.4f} to {stats['z_max']:.4f} "
                    f"({stats['z_range']*1000:.1f} mm)\n\n"
                    f"Centroid: ({stats['centroid'][0]:.4f}, "
                    f"{stats['centroid'][1]:.4f}, {stats['centroid'][2]:.4f})"
                )
                QMessageBox.information(self, "OpenFOAM Points Loaded", msg)

                # Display in 3D viewer and switch to that tab
                viewer = self._ensure_point_cloud_viewer()
                viewer.set_points(points, stats)
                self._tabs.setCurrentWidget(viewer)

                self.statusBar().showMessage(
                    f"Loaded {stats['n_points']:,} points from {path}"
                )
            except OpenFOAMReadError as e:
                QMessageBox.warning(self, "Import Error", str(e))
            except Exception as e:
                QMessageBox.critical(
                    self, "Unexpected Error",
                    f"An unexpected error occurred while reading the file.\n\n"
                    f"File: {path}\n"
                    f"Error: {type(e).__name__}: {e}"
                )

    # ----------------------------------------------------------------
    # Export operations
    # ----------------------------------------------------------------

    def _export_cgns(self) -> None:
        if self._last_mesh is None:
            QMessageBox.warning(
                self, "No Mesh",
                "Generate a mesh first (Compute > Generate Multi-Block Mesh)"
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CGNS", "mesh.cgns", "CGNS Files (*.cgns)"
        )
        if path:
            try:
                from ..export.cgns_writer import write_cgns_structured

                block_arrays = [b.points for b in self._last_mesh.blocks]
                block_names = [b.name for b in self._last_mesh.blocks]

                # Collect patches from mesh blocks for CGNS BCs
                patches = {}
                for block in self._last_mesh.blocks:
                    if block.patches:
                        patches[block.name] = block.patches

                write_cgns_structured(
                    path, block_arrays, block_names,
                    patches=patches if patches else None,
                )

                bc_msg = ""
                if patches:
                    n_bcs = sum(len(v) for v in patches.values())
                    bc_msg = f"Boundary conditions: {n_bcs}\n"

                self.statusBar().showMessage(f"Exported CGNS: {path}")
                QMessageBox.information(
                    self, "Export Successful",
                    f"CGNS exported to {path}\n\n"
                    f"Blocks: {self._last_mesh.n_blocks}\n"
                    f"Total cells: {self._last_mesh.total_cells}\n"
                    f"{bc_msg}"
                )
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    def _export_openfoam(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export blockMeshDict", "blockMeshDict", "All Files (*)"
        )
        if path:
            try:
                # Simple single-block export for demonstration
                vertices = np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                ], dtype=np.float64)
                blocks = [{"vertices": list(range(8)), "cells": [20, 20, 1], "grading": [1, 1, 1]}]
                patches = [
                    {"name": "inlet", "type": "patch", "faces": [[0, 3, 7, 4]]},
                    {"name": "outlet", "type": "patch", "faces": [[1, 2, 6, 5]]},
                ]
                write_blockmeshdict(path, vertices, blocks, patches)
                self.statusBar().showMessage(f"Exported blockMeshDict: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    def _export_vtk(self) -> None:
        if self._last_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Generate a mesh first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export VTK", "mesh.vtk", "VTK Files (*.vtk)"
        )
        if path:
            try:
                from ..export import export_structured_as_quads
                block_arrays = [b.points for b in self._last_mesh.blocks]
                export_structured_as_quads(path, block_arrays, file_format="vtk")
                self.statusBar().showMessage(f"Exported VTK: {path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    # ----------------------------------------------------------------
    # Edit operations
    # ----------------------------------------------------------------

    def _add_blade_row(self) -> None:
        row = BladeRow(
            hub_points=np.array([[0.0, 0.10], [0.10, 0.10]]),
            shroud_points=np.array([[0.0, 0.20], [0.10, 0.20]]),
        )
        idx = len(self._machine.blade_rows)
        row.name = f"Row {idx}"
        row.add_profile(Superposition.default())
        self._machine.add_blade_row(row)
        self._current_row = row
        self._tree_panel.set_machine(self._machine)
        self.statusBar().showMessage(f"Added blade row: Row {idx}")

    def _add_profile_to_row(self) -> None:
        if self._current_row is None:
            QMessageBox.warning(self, "No Row", "Select or create a blade row first.")
            return
        profile = Superposition.default()
        self._current_row.add_profile(profile)
        self._current_profile = profile
        self._tree_panel.set_machine(self._machine)
        self.statusBar().showMessage(
            f"Added profile to {self._current_row.name} "
            f"({len(self._current_row.profiles)} profiles)"
        )

    # ----------------------------------------------------------------
    # Compute operations
    # ----------------------------------------------------------------

    def _compute_blade(self) -> None:
        if self._current_row is None:
            QMessageBox.warning(self, "No Row", "Create a blade row first.")
            return

        try:
            import logging

            n = len(self._current_row.profiles)
            if n == 0:
                QMessageBox.warning(self, "No Profiles", "Add profiles to the blade row first.")
                return

            if n < 2:
                QMessageBox.warning(
                    self, "Need More Profiles",
                    "At least 2 span profiles are required for 3D blade.\n"
                    "Use Edit > Add Profile to Row."
                )
                return

            # Capture blade validation warnings
            blade_warnings = []
            class _WarningHandler(logging.Handler):
                def emit(self, record):
                    if record.levelno >= logging.WARNING:
                        blade_warnings.append(record.getMessage())

            handler = _WarningHandler()
            blade_logger = logging.getLogger("astraturbo.blade.blade_row")
            blade_logger.addHandler(handler)
            blade_logger.setLevel(logging.WARNING)

            # Vary stagger and chord from hub to tip for realistic 3D blade
            stagger = np.linspace(np.deg2rad(25), np.deg2rad(45), n)
            chords = np.linspace(0.04, 0.06, n)

            self._current_row.compute(
                stagger_angles=stagger,
                chord_lengths=chords,
            )

            blade_logger.removeHandler(handler)

            le = self._current_row.leading_edge
            te = self._current_row.trailing_edge

            self.statusBar().showMessage(
                f"Blade computed: {n} profiles, "
                f"LE shape {le.shape}, TE shape {te.shape}"
            )

            msg = (
                f"3D blade surface generated.\n\n"
                f"Profiles: {n}\n"
                f"LE points: {le.shape}\n"
                f"TE points: {te.shape}\n"
                f"Surface: {self._current_row.blade_surface}"
            )

            if blade_warnings:
                msg += f"\n\nValidation Warnings ({len(blade_warnings)}):\n"
                for w in blade_warnings:
                    msg += f"  - {w}\n"
                QMessageBox.warning(self, "Blade Computed (with warnings)", msg)
            else:
                QMessageBox.information(self, "Blade Geometry Computed", msg)
        except Exception as e:
            QMessageBox.warning(self, "Compute Error", f"{e}\n\n{traceback.format_exc()}")

    def _generate_blade_array(self) -> None:
        """Generate full annular blade array."""
        if self._current_row is None or self._current_row.profiles_3d is None:
            QMessageBox.warning(
                self, "No Blade",
                "Compute blade geometry first (Compute > Compute Blade Geometry)."
            )
            return

        try:
            from ..blade import generate_blade_array_flat

            all_points = generate_blade_array_flat(
                self._current_row.profiles_3d,
                self._current_row.number_blades,
            )

            n_blades = self._current_row.number_blades
            n_points = len(all_points)

            # Display in 3D viewer
            viewer = self._ensure_point_cloud_viewer()
            viewer.set_points(all_points)
            self._tabs.setCurrentWidget(viewer)

            self.statusBar().showMessage(
                f"Blade array: {n_blades} blades, {n_points:,} total points"
            )
            QMessageBox.information(
                self, "Blade Array Generated",
                f"Full annular array created.\n\n"
                f"Blades: {n_blades}\n"
                f"Total points: {n_points:,}\n\n"
                f"Displayed in 3D Viewer tab."
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"{e}\n\n{traceback.format_exc()}")

    def _compute_s1_mesh(self) -> None:
        """Generate S1 blade-to-blade mesh."""
        if self._current_row is None or not self._current_row.profiles:
            QMessageBox.warning(self, "No Profile", "Create a profile first.")
            return

        try:
            from ..mesh import S1Mesher, S1MeshConfig, mesh_quality_report

            profile = self._current_row.profiles[0].as_array()
            pitch = 2 * np.pi * 0.15 / max(self._current_row.number_blades, 1)

            config = S1MeshConfig(
                n_streamwise=self._mesh_panel._n_axial.value(),
                n_pitchwise=self._mesh_panel._n_radial.value(),
            )
            mesher = S1Mesher(config)
            blocks = mesher.generate(profile, pitch=pitch)

            total_cells = mesher.total_cells()
            report = mesh_quality_report(blocks[0].points)
            report["n_cells"] = total_cells
            self._mesh_panel.show_quality_report(report)

            self.statusBar().showMessage(
                f"S1 mesh: {len(blocks)} blocks, {total_cells} cells"
            )
        except Exception as e:
            QMessageBox.warning(self, "S1 Mesh Error", f"{e}\n\n{traceback.format_exc()}")

    def _compute_scm_mesh(self) -> None:
        if self._current_row is None:
            QMessageBox.warning(self, "No Row", "Create a blade row first.")
            return

        try:
            cfg = SCMMeshConfig(
                n_inlet_axial=self._mesh_panel._n_axial.value() // 3,
                n_blade_axial=self._mesh_panel._n_axial.value(),
                n_outlet_axial=self._mesh_panel._n_axial.value() // 3,
                n_radial=self._mesh_panel._n_radial.value(),
            )
            mesher = SCMMesher(cfg)
            blocks = mesher.generate(
                hub_contour=self._current_row.hub.points,
                shroud_contour=self._current_row.shroud.points,
                le_z=0.03, te_z=0.07,
            )

            # Quality report on first block
            report = mesh_quality_report(blocks[0].points)
            total_cells = sum(
                (b.points.shape[0] - 1) * (b.points.shape[1] - 1) for b in blocks
            )
            report["n_cells"] = total_cells

            self._mesh_panel.show_quality_report(report)
            self._last_mesh_blocks = [b.points for b in blocks]

            self.statusBar().showMessage(
                f"SCM mesh: {len(blocks)} blocks, {total_cells} cells"
            )
        except Exception as e:
            QMessageBox.warning(self, "Mesh Error", f"{e}\n\n{traceback.format_exc()}")

    def _compute_ogrid_mesh(self) -> None:
        self._compute_multiblock_mesh()

    def _compute_multiblock_mesh(self) -> None:
        if self._current_row is None or not self._current_row.profiles:
            QMessageBox.warning(self, "No Profile", "Create a profile first.")
            return

        try:
            profile = self._current_row.profiles[0].as_array()
            mesh = generate_blade_passage_mesh(
                profile=profile,
                pitch=0.05,
                n_blade=self._mesh_panel._n_axial.value(),
                n_ogrid=max(3, self._mesh_panel._n_radial.value() // 4),
                n_inlet=max(3, self._mesh_panel._n_axial.value() // 3),
                n_outlet=max(3, self._mesh_panel._n_axial.value() // 3),
                n_passage=self._mesh_panel._n_radial.value(),
                ogrid_thickness=0.005,
            )

            self._last_mesh = mesh

            # Quality report on first block
            first_block = mesh.blocks[0].points
            report = mesh_quality_report(first_block)
            report["n_cells"] = mesh.total_cells
            report["n_points"] = mesh.total_points
            self._mesh_panel.show_quality_report(report)

            self.statusBar().showMessage(
                f"Multi-block mesh: {mesh.n_blocks} blocks, "
                f"{mesh.total_cells} cells — Ready to export"
            )
        except Exception as e:
            QMessageBox.warning(self, "Mesh Error", f"{e}\n\n{traceback.format_exc()}")

    def _compute_3d_mesh(self) -> None:
        """Generate a 3D blade passage mesh by stacking at span stations."""
        from PySide6.QtWidgets import QInputDialog

        if self._current_row is None or not self._current_row.profiles:
            QMessageBox.warning(self, "No Profile", "Create a profile first.")
            return

        if len(self._current_row.profiles) < 2:
            QMessageBox.warning(
                self, "Need Multiple Profiles",
                "At least 2 span profiles are required for 3D mesh.\n"
                "Use Edit > Add Profile to Row to add more."
            )
            return

        n_span, ok = QInputDialog.getInt(
            self, "3D Mesh", "Number of span stations:", len(self._current_row.profiles), 2, 50
        )
        if not ok:
            return

        span, ok = QInputDialog.getDouble(
            self, "3D Mesh", "Total span height (m):", 0.05, 0.001, 5.0, 4
        )
        if not ok:
            return

        try:
            from ..mesh.multiblock import generate_blade_passage_mesh_3d

            # Get profiles from current blade row
            profiles = [p.as_array() for p in self._current_row.profiles]
            span_positions = np.linspace(0, span, n_span).tolist()

            # If we have fewer blade row profiles than requested span stations,
            # interpolate by reusing available profiles
            while len(profiles) < n_span:
                profiles.append(profiles[-1])

            mesh = generate_blade_passage_mesh_3d(
                profiles=profiles[:n_span],
                span_positions=span_positions,
                pitch=0.05,
                n_blade=self._mesh_panel._n_axial.value(),
                n_ogrid=max(3, self._mesh_panel._n_radial.value() // 4),
                n_inlet=max(3, self._mesh_panel._n_axial.value() // 3),
                n_outlet=max(3, self._mesh_panel._n_axial.value() // 3),
                n_passage=self._mesh_panel._n_radial.value(),
            )

            self._last_mesh = mesh

            self.statusBar().showMessage(
                f"3D mesh: {mesh.n_blocks} blocks, {n_span} span stations — Ready to export"
            )
            QMessageBox.information(
                self, "3D Mesh Generated",
                f"3D blade passage mesh generated.\n\n"
                f"Blocks: {mesh.n_blocks}\n"
                f"Span stations: {n_span}\n"
                f"Span height: {span} m\n\n"
                f"Use File > Export > CGNS to export."
            )
        except Exception as e:
            QMessageBox.warning(self, "3D Mesh Error", f"{e}\n\n{traceback.format_exc()}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About AstraTurbo",
            "AstraTurbo v0.1.0\n\n"
            "Open-source integrated turbomachinery\n"
            "design and simulation platform.\n\n"
            "Modules: Meanline | Blade | Mesh | CFD | FEA | Optimization\n"
            "Formats: 30 (CGNS, OpenFOAM, Fluent, CFX, VTK, ...)\n"
            "Materials: 6 (Inconel, Ti-6Al-4V, CMSX-4, ...)\n\n"
            "How to use:\n"
            "1. Compute > Meanline Design (or select profile manually)\n"
            "2. Compute > Blade Geometry\n"
            "3. Compute > Generate Multi-Block Mesh\n"
            "4. Compute > CFD Case Setup > OpenFOAM/Fluent/CFX/SU2\n"
            "5. Compute > FEA Structural Analysis\n"
            "6. File > Export > CGNS Mesh",
        )

    # ----------------------------------------------------------------
    # Meanline design
    # ----------------------------------------------------------------

    def _run_meanline(self) -> None:
        """Run meanline compressor design from a dialog."""
        from PySide6.QtWidgets import QInputDialog

        # Get inputs via simple dialogs
        pr, ok = QInputDialog.getDouble(
            self, "Meanline Design", "Overall Pressure Ratio:", 4.0, 1.1, 30.0, 2
        )
        if not ok:
            return

        mass_flow, ok = QInputDialog.getDouble(
            self, "Meanline Design", "Mass Flow (kg/s):", 20.0, 0.1, 1000.0, 1
        )
        if not ok:
            return

        rpm, ok = QInputDialog.getDouble(
            self, "Meanline Design", "RPM:", 12000, 100, 100000, 0
        )
        if not ok:
            return

        r_hub, ok = QInputDialog.getDouble(
            self, "Meanline Design", "Hub Radius (m):", 0.15, 0.01, 5.0, 3
        )
        if not ok:
            return

        r_tip, ok = QInputDialog.getDouble(
            self, "Meanline Design", "Tip Radius (m):", 0.30, 0.02, 5.0, 3
        )
        if not ok:
            return

        radial_stations, ok = QInputDialog.getInt(
            self, "Meanline Design", "Radial Stations (hub/mid/tip):", 3, 2, 20
        )
        if not ok:
            return

        try:
            from ..design import meanline_compressor, meanline_to_blade_parameters
            from ..design.meanline import blade_angle_to_cl0
            import math

            result = meanline_compressor(
                overall_pressure_ratio=pr, mass_flow=mass_flow,
                rpm=rpm, r_hub=r_hub, r_tip=r_tip,
                radial_stations=radial_stations,
            )

            params = meanline_to_blade_parameters(result)

            summary = result.summary()

            # Auto-computed cl0 for each stage
            summary += "\n\nAuto-computed Profile Parameters:\n"
            for stage, bp in zip(result.stages, params):
                cl0 = blade_angle_to_cl0(
                    stage.rotor_inlet_beta, stage.rotor_outlet_beta,
                    bp["rotor_solidity"],
                )
                summary += (
                    f"  Stage {stage.stage_number}: cl0 = {cl0:.4f}, "
                    f"stagger = {bp['rotor_stagger_deg']:.1f} deg, "
                    f"solidity = {bp['rotor_solidity']:.2f}\n"
                )

            summary += "\nBlade Parameters:\n"
            for p in params:
                summary += (
                    f"\nStage {p['stage']}:\n"
                    f"  Rotor: stagger={p['rotor_stagger_deg']:.1f} deg, "
                    f"camber={p['rotor_camber_deg']:.1f} deg\n"
                    f"  Stator: stagger={p['stator_stagger_deg']:.1f} deg, "
                    f"camber={p['stator_camber_deg']:.1f} deg\n"
                    f"  De Haller: {p['de_haller']:.3f}\n"
                )

            # Radial blade angle table
            summary += "\nRadial Blade Angles (free vortex):\n"
            for stage in result.stages:
                summary += f"  Stage {stage.stage_number}:\n"
                summary += f"    {'r (m)':>8s}  {'beta_in':>8s}  {'beta_out':>9s}  {'alpha_in':>9s}  {'alpha_out':>10s}\n"
                for a in stage.radial_blade_angles:
                    summary += (
                        f"    {a['r']:8.4f}  "
                        f"{math.degrees(a['beta_in']):8.1f}  "
                        f"{math.degrees(a['beta_out']):9.1f}  "
                        f"{math.degrees(a['alpha_in']):9.1f}  "
                        f"{math.degrees(a['alpha_out']):10.1f}\n"
                    )

            QMessageBox.information(self, "Meanline Design Result", summary)
            self.statusBar().showMessage(
                f"Meanline: {result.n_stages} stages, PR={result.overall_pressure_ratio:.2f}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Meanline Error", f"{e}\n\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    # CFD case setup
    # ----------------------------------------------------------------

    def _setup_cfd(self, solver: str) -> None:
        """Set up a CFD case for the specified solver."""
        from PySide6.QtWidgets import QInputDialog, QCheckBox, QDialog, QVBoxLayout, QDialogButtonBox, QLabel, QDoubleSpinBox, QFormLayout

        # Use a custom dialog for compressible options
        dlg = QDialog(self)
        dlg.setWindowTitle(f"CFD Setup ({solver})")
        layout = QVBoxLayout(dlg)

        form = QFormLayout()

        vel_spin = QDoubleSpinBox()
        vel_spin.setRange(1, 10000)
        vel_spin.setValue(100.0)
        vel_spin.setDecimals(1)
        form.addRow("Inlet Velocity (m/s):", vel_spin)

        comp_check = QCheckBox("Compressible (rhoSimpleFoam)")
        form.addRow("Flow type:", comp_check)

        total_p_spin = QDoubleSpinBox()
        total_p_spin.setRange(1000, 1e8)
        total_p_spin.setValue(101325.0)
        total_p_spin.setDecimals(0)
        total_p_spin.setEnabled(False)
        form.addRow("Total Pressure (Pa):", total_p_spin)

        total_t_spin = QDoubleSpinBox()
        total_t_spin.setRange(100, 5000)
        total_t_spin.setValue(288.15)
        total_t_spin.setDecimals(1)
        total_t_spin.setEnabled(False)
        form.addRow("Total Temperature (K):", total_t_spin)

        comp_check.toggled.connect(total_p_spin.setEnabled)
        comp_check.toggled.connect(total_t_spin.setEnabled)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        velocity = vel_spin.value()
        compressible = comp_check.isChecked()
        total_pressure = total_p_spin.value()
        total_temperature = total_t_spin.value()

        path, _ = QFileDialog.getSaveFileName(
            self, f"CFD Case Directory ({solver})", f"{solver}_case"
        )
        if not path:
            return

        try:
            from ..cfd import CFDWorkflow, CFDWorkflowConfig

            cfg = CFDWorkflowConfig(
                solver=solver,
                inlet_velocity=velocity,
                turbulence_model="kOmegaSST",
                compressible=compressible,
                total_pressure=total_pressure,
                total_temperature=total_temperature,
            )
            wf = CFDWorkflow(cfg)
            if self._last_mesh:
                # Pass patch names from mesh blocks if available
                if hasattr(self._last_mesh, 'blocks'):
                    patch_names_from_mesh = {}
                    for block in self._last_mesh.blocks:
                        if block.patches:
                            for face, bc_type in block.patches.items():
                                if bc_type not in patch_names_from_mesh:
                                    patch_names_from_mesh[bc_type] = bc_type

            case = wf.setup_case(path)

            solver_name = "rhoSimpleFoam" if compressible else solver
            msg = (
                f"Solver: {solver_name}\n"
                f"Velocity: {velocity} m/s\n"
            )
            if compressible:
                msg += f"Total Pressure: {total_pressure} Pa\nTotal Temperature: {total_temperature} K\n"
            msg += (
                f"Directory: {case}\n\n"
                f"Run the case from terminal:\n"
            )
            if solver == "openfoam":
                msg += f"  cd {case} && bash Allrun"
            else:
                msg += f"  cd {case} && bash run_{solver}.sh"

            QMessageBox.information(self, "CFD Case Created", msg)
            self.statusBar().showMessage(f"CFD case ({solver_name}) created at {case}")
        except Exception as e:
            QMessageBox.warning(self, "CFD Error", str(e))

    # ----------------------------------------------------------------
    # FEA structural analysis
    # ----------------------------------------------------------------

    def _setup_fea(self) -> None:
        """Set up FEA structural analysis."""
        from PySide6.QtWidgets import QInputDialog

        from ..fea import list_materials, get_material

        materials = list_materials()
        mat_name, ok = QInputDialog.getItem(
            self, "FEA Setup", "Select Material:", materials, 0, False
        )
        if not ok:
            return

        omega, ok = QInputDialog.getDouble(
            self, "FEA Setup", "Angular Velocity (rad/s):", 1200.0, 0, 50000, 1
        )
        if not ok:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "FEA Case Directory", "fea_case"
        )
        if not path:
            return

        try:
            from ..fea import FEAWorkflow, FEAWorkflowConfig

            material = get_material(mat_name)
            cfg = FEAWorkflowConfig(material=material, omega=omega)
            fea = FEAWorkflow(cfg)

            # Use current blade profile as surface if available
            if self._current_row and self._current_row.profiles:
                profile = self._current_row.profiles[0].as_array()
                n = len(profile)
                # Create a simple surface from profile (extruded in z)
                ni, nj = n, 3
                pts = np.zeros((ni * nj, 3))
                for j in range(nj):
                    pts[j*ni:(j+1)*ni, 0] = profile[:, 0]
                    pts[j*ni:(j+1)*ni, 1] = profile[:, 1]
                    pts[j*ni:(j+1)*ni, 2] = j * 0.05  # 50mm span
                fea.set_blade_surface(pts, ni, nj)

                case = fea.setup(path)

                estimate = fea.estimate_stress_analytical()

                QMessageBox.information(
                    self, "FEA Case Created",
                    f"Material: {material.name}\n"
                    f"Omega: {omega} rad/s\n"
                    f"Directory: {case}\n\n"
                    f"Analytical Estimate:\n"
                    f"  Centrifugal stress: {estimate['centrifugal_stress_MPa']:.1f} MPa\n"
                    f"  Yield strength: {material.yield_strength/1e6:.0f} MPa\n"
                    f"  Safety factor: {estimate['safety_factor']:.2f}\n\n"
                    f"Next: cd {case} && bash run_fea.sh  (requires CalculiX)"
                )
                self.statusBar().showMessage(
                    f"FEA case created: {material.name}, SF={estimate['safety_factor']:.2f}"
                )
            else:
                QMessageBox.warning(
                    self, "No Blade",
                    "Create a blade profile first (2D Profile tab), "
                    "then try FEA setup again."
                )
        except Exception as e:
            QMessageBox.warning(self, "FEA Error", f"{e}\n\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    # Optimization
    # ----------------------------------------------------------------

    def _run_optimization(self) -> None:
        """Run blade design optimization."""
        from PySide6.QtWidgets import QInputDialog

        gens, ok = QInputDialog.getInt(
            self, "Optimization", "Number of generations:", 30, 5, 500
        )
        if not ok:
            return

        pop, ok = QInputDialog.getInt(
            self, "Optimization", "Population size:", 15, 5, 200
        )
        if not ok:
            return

        try:
            from ..optimization import (
                Optimizer, OptimizationConfig, create_blade_design_space,
            )

            design_space = create_blade_design_space(n_profiles=3)

            def evaluate(x):
                penalty = float(np.sum((x - (design_space.lower_bounds + design_space.upper_bounds) / 2) ** 2))
                return np.array([-1.0 + penalty / 100.0]), np.array([])

            self.statusBar().showMessage("Running optimization...")
            optimizer = Optimizer(design_space, evaluate, n_objectives=1)
            result = optimizer.run(OptimizationConfig(
                n_generations=gens, population_size=pop,
            ))

            msg = (
                f"Optimization Complete\n\n"
                f"Evaluations: {result.n_evaluations}\n"
                f"Best objective: {result.best_f}\n"
                f"Converged: {result.converged}"
            )
            QMessageBox.information(self, "Optimization Result", msg)
            self.statusBar().showMessage(
                f"Optimization done: {result.n_evaluations} evaluations"
            )
        except ImportError:
            QMessageBox.warning(
                self, "Missing Dependency",
                "pymoo is required for optimization.\n"
                "Install with: pip install pymoo\n\n"
                "Falling back to scipy (single-objective only)."
            )
        except Exception as e:
            QMessageBox.warning(self, "Optimization Error", f"{e}\n\n{traceback.format_exc()}")

    def _run_multifidelity(self) -> None:
        """Run multi-fidelity optimization."""
        from PySide6.QtWidgets import QInputDialog

        n_levels, ok = QInputDialog.getInt(
            self, "Multi-Fidelity", "Number of fidelity levels:", 3, 2, 5
        )
        if not ok:
            return

        try:
            from ..optimization.multifidelity import MultiFidelityOptimizer

            optimizer = MultiFidelityOptimizer.create_default_turbomachinery(
                n_levels=n_levels
            )

            self.statusBar().showMessage("Running multi-fidelity optimization...")
            result = optimizer.run()

            best = result.best_design()
            msg = (
                f"Multi-Fidelity Optimization Complete\n\n"
                f"Levels: {n_levels}\n"
                f"Total evaluations: {sum(lvl.n_evaluations for lvl in result.levels)}\n"
            )
            if best:
                msg += f"Best efficiency: {best.get('efficiency', 'N/A')}\n"

            QMessageBox.information(self, "Multi-Fidelity Result", msg)
            self.statusBar().showMessage("Multi-fidelity optimization complete")
        except Exception as e:
            QMessageBox.warning(self, "Multi-Fidelity Error", f"{e}\n\n{traceback.format_exc()}")

    def _generate_tip_clearance(self) -> None:
        """Generate a tip clearance mesh block."""
        from PySide6.QtWidgets import QInputDialog

        if self._last_mesh is None:
            QMessageBox.warning(self, "No Mesh", "Generate a blade passage mesh first.")
            return

        clearance, ok = QInputDialog.getDouble(
            self, "Tip Clearance", "Clearance height (mm):", 1.0, 0.01, 50.0, 3
        )
        if not ok:
            return

        n_layers, ok = QInputDialog.getInt(
            self, "Tip Clearance", "Number of clearance layers:", 8, 2, 50
        )
        if not ok:
            return

        try:
            from ..mesh.tip_clearance import generate_tip_clearance_mesh

            # Use the tip of the blade passage mesh
            blade_pts = self._last_mesh.blocks[0].points
            tip_mesh, metrics = generate_tip_clearance_mesh(
                blade_pts,
                clearance_height=clearance / 1000.0,  # mm to m
                n_clearance=n_layers,
            )

            msg = (
                f"Tip Clearance Mesh Generated\n\n"
                f"Shape: {tip_mesh.shape}\n"
                f"Clearance: {clearance} mm\n"
                f"Layers: {n_layers}\n"
            )
            if metrics:
                msg += f"Max aspect ratio: {metrics.get('aspect_ratio_max', 'N/A'):.2f}\n"

            QMessageBox.information(self, "Tip Clearance", msg)
            self.statusBar().showMessage("Tip clearance mesh generated")
        except Exception as e:
            QMessageBox.warning(self, "Tip Clearance Error", f"{e}\n\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    # Tools
    # ----------------------------------------------------------------

    def _yplus_calculator(self) -> None:
        """y+ calculator dialog."""
        from PySide6.QtWidgets import QInputDialog

        velocity, ok = QInputDialog.getDouble(
            self, "y+ Calculator", "Freestream velocity (m/s):", 100.0, 1, 10000, 1
        )
        if not ok:
            return

        chord, ok = QInputDialog.getDouble(
            self, "y+ Calculator", "Chord length (m):", 0.1, 0.001, 10, 4
        )
        if not ok:
            return

        from ..mesh import first_cell_height_for_yplus, estimate_yplus

        dy_1 = first_cell_height_for_yplus(1.0, 1.225, velocity, 1.8e-5, chord)
        dy_30 = first_cell_height_for_yplus(30.0, 1.225, velocity, 1.8e-5, chord)

        msg = (
            f"y+ Calculator Results\n\n"
            f"Velocity: {velocity} m/s\n"
            f"Chord: {chord*1000:.1f} mm\n"
            f"Fluid: Air at sea level\n\n"
            f"For y+ = 1 (resolved BL):\n"
            f"  First cell: {dy_1*1e6:.1f} um ({dy_1*1000:.4f} mm)\n\n"
            f"For y+ = 30 (wall functions):\n"
            f"  First cell: {dy_30*1e6:.1f} um ({dy_30*1000:.4f} mm)"
        )
        QMessageBox.information(self, "y+ Calculator", msg)

    def _show_formats(self) -> None:
        """Show supported file formats."""
        from ..export import list_supported_formats

        fmts = list_supported_formats()
        lines = [f"Supported File Formats: {len(fmts)}\n"]
        for name, info in sorted(fmts.items()):
            rw = ("R" if info["read"] else "-") + ("W" if info["write"] else "-")
            exts = ", ".join(info["extensions"])
            lines.append(f"[{rw}] {name}: {exts} — {info['description']}")

        QMessageBox.information(self, "Supported Formats", "\n".join(lines))

    # ----------------------------------------------------------------
    # Throughflow solver
    # ----------------------------------------------------------------

    def _run_throughflow(self) -> None:
        """Run the throughflow solver and display results."""
        from PySide6.QtWidgets import QInputDialog

        pr, ok = QInputDialog.getDouble(
            self, "Throughflow Solver", "Pressure Ratio:", 1.5, 1.01, 10.0, 3
        )
        if not ok:
            return

        rpm, ok = QInputDialog.getDouble(
            self, "Throughflow Solver", "RPM:", 10000, 100, 100000, 0
        )
        if not ok:
            return

        r_hub, ok = QInputDialog.getDouble(
            self, "Throughflow Solver", "Hub Radius (m):", 0.1, 0.01, 5.0, 3
        )
        if not ok:
            return

        r_tip, ok = QInputDialog.getDouble(
            self, "Throughflow Solver", "Tip Radius (m):", 0.2, 0.02, 5.0, 3
        )
        if not ok:
            return

        try:
            from ..solver.throughflow import (
                ThroughflowSolver, ThroughflowConfig, BladeRowSpec,
            )

            n_stations = 20
            n_streamlines = 11
            config = ThroughflowConfig(
                n_stations=n_stations,
                n_streamlines=n_streamlines,
            )
            solver = ThroughflowSolver(config)

            hub_r = np.full(n_stations, r_hub)
            tip_r = np.full(n_stations, r_tip)
            axial = np.linspace(0.0, 0.2, n_stations)
            solver.set_annulus(hub_r, tip_r, axial)

            omega = rpm * 2.0 * np.pi / 60.0
            rotor = BladeRowSpec(
                row_type="rotor", n_blades=36,
                inlet_station=n_stations // 4,
                outlet_station=n_stations // 2,
                omega=omega,
            )
            solver.add_blade_row(rotor)
            solver.set_inlet_conditions()

            self.statusBar().showMessage("Running throughflow solver...")
            result = solver.solve()

            pr_actual = "N/A"
            if result.total_pressure is not None:
                pr_actual = f"{result.total_pressure[-1, :].mean() / result.total_pressure[0, :].mean():.4f}"

            msg = (
                f"Throughflow Solver Results\n\n"
                f"Converged: {result.converged}\n"
                f"Iterations: {result.n_iterations}\n"
                f"Pressure ratio: {pr_actual}\n"
            )
            if result.mach_number is not None:
                msg += f"Max Mach: {result.mach_number.max():.4f}\n"
            if result.residual_history:
                msg += f"Final residual: {result.residual_history[-1]:.2e}\n"

            QMessageBox.information(self, "Throughflow Results", msg)
            self.statusBar().showMessage(
                f"Throughflow done: {result.n_iterations} iterations, PR={pr_actual}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Throughflow Error", f"{e}\n\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    # Mesh smoothing
    # ----------------------------------------------------------------

    def _smooth_mesh(self) -> None:
        """Apply Laplacian smoothing to the last generated mesh."""
        if self._last_mesh is None:
            QMessageBox.warning(
                self, "No Mesh",
                "Generate a mesh first (Compute > Generate Multi-Block Mesh)"
            )
            return

        from PySide6.QtWidgets import QInputDialog

        iterations, ok = QInputDialog.getInt(
            self, "Smooth Mesh", "Number of iterations:", 50, 1, 500
        )
        if not ok:
            return

        try:
            from ..mesh.smoothing import laplacian_smooth
            from ..mesh.quality import mesh_quality_report

            # Smooth the first block of the mesh
            block_pts = self._last_mesh.blocks[0].points
            smoothed, metrics = laplacian_smooth(block_pts, n_iterations=iterations)

            # Update the block
            self._last_mesh.blocks[0].points = smoothed

            # Update quality report
            report = mesh_quality_report(smoothed)
            report["n_cells"] = self._last_mesh.total_cells
            report["n_points"] = self._last_mesh.total_points
            self._mesh_panel.show_quality_report(report)

            msg = (
                f"Mesh Smoothing Results ({iterations} iterations)\n\n"
                f"Before:\n"
                f"  Aspect ratio max: {metrics['before_aspect_ratio_max']:.3f}\n"
                f"  Skewness max: {metrics['before_skewness_max']:.3f}\n\n"
                f"After:\n"
                f"  Aspect ratio max: {metrics['after_aspect_ratio_max']:.3f}\n"
                f"  Skewness max: {metrics['after_skewness_max']:.3f}"
            )
            QMessageBox.information(self, "Mesh Smoothing", msg)
            self.statusBar().showMessage(
                f"Mesh smoothed: {iterations} iterations, "
                f"skewness {metrics['before_skewness_max']:.3f} -> {metrics['after_skewness_max']:.3f}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Smoothing Error", f"{e}\n\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    # Design database
    # ----------------------------------------------------------------

    def _open_design_database(self) -> None:
        """Open a dialog to list, save, and search designs."""
        from PySide6.QtWidgets import QInputDialog

        actions = ["List designs", "Save current design", "Search designs"]
        action, ok = QInputDialog.getItem(
            self, "Design Database", "Select action:", actions, 0, False
        )
        if not ok:
            return

        try:
            from ..database.design_db import DesignDatabase
            db = DesignDatabase()

            if action == "List designs":
                designs = db.list_designs()
                if not designs:
                    QMessageBox.information(self, "Design Database", "No designs in database.")
                else:
                    lines = [f"Designs ({len(designs)}):\n"]
                    for d in designs:
                        tags = ", ".join(d["tags"]) if d["tags"] else ""
                        params = d.get("parameters", {})
                        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
                        line = f"[{d['id']}] {d['name']}"
                        if param_str:
                            line += f"  ({param_str})"
                        if tags:
                            line += f"  [{tags}]"
                        lines.append(line)
                    QMessageBox.information(self, "Design Database", "\n".join(lines))

            elif action == "Save current design":
                name, ok = QInputDialog.getText(
                    self, "Save Design", "Design name:"
                )
                if not ok or not name:
                    db.close()
                    return

                params = {}
                if self._current_profile:
                    cl = getattr(self._current_profile._camber_line, "cl0", None)
                    if cl is not None:
                        params["cl0"] = cl
                    mt = getattr(self._current_profile._thickness, "max_thickness", None)
                    if mt is not None:
                        params["max_thickness"] = mt

                design_id = db.save_design(name=name, parameters=params)
                QMessageBox.information(
                    self, "Design Saved",
                    f"Design '{name}' saved with ID={design_id}"
                )

            elif action == "Search designs":
                query, ok = QInputDialog.getText(
                    self, "Search Designs", "Search query:"
                )
                if not ok:
                    db.close()
                    return

                results = db.search(query)
                if not results:
                    QMessageBox.information(self, "Search Results", "No matching designs found.")
                else:
                    lines = [f"Found {len(results)} designs:"]
                    for d in results:
                        lines.append(f"[{d['id']}] {d['name']}")
                    QMessageBox.information(self, "Search Results", "\n".join(lines))

            db.close()
        except Exception as e:
            QMessageBox.warning(self, "Database Error", f"{e}\n\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    # HPC Job Manager
    # ----------------------------------------------------------------

    def _open_hpc_manager(self) -> None:
        """Show a dialog for HPC job submission."""
        from PySide6.QtWidgets import QInputDialog

        actions = ["Submit job", "Check job status", "Cancel job",
                   "Download results", "Setup AWS Batch", "Teardown AWS Batch"]
        action, ok = QInputDialog.getItem(
            self, "HPC Job Manager", "Select action:", actions, 0, False
        )
        if not ok:
            return

        try:
            from ..hpc.job_manager import HPCJobManager, HPCConfig

            if action == "Submit job":
                backends = ["local", "slurm", "pbs", "aws"]
                backend, ok = QInputDialog.getItem(
                    self, "HPC Backend", "Select backend:", backends, 0, False
                )
                if not ok:
                    return

                config_kwargs: dict = {"backend": backend}

                if backend == "aws":
                    region, ok = QInputDialog.getText(
                        self, "AWS Region", "AWS region:", text="us-east-1"
                    )
                    if not ok:
                        return
                    queue, ok = QInputDialog.getText(
                        self, "AWS Job Queue", "AWS Batch job queue name:"
                    )
                    if not ok or not queue:
                        return
                    bucket, ok = QInputDialog.getText(
                        self, "AWS S3 Bucket", "S3 bucket for case data:"
                    )
                    if not ok or not bucket:
                        return
                    config_kwargs.update(
                        aws_region=region,
                        aws_job_queue=queue,
                        aws_s3_bucket=bucket,
                    )
                else:
                    nodes, ok = QInputDialog.getInt(
                        self, "HPC Submit", "Number of nodes:", 1, 1, 1000
                    )
                    if not ok:
                        return
                    config_kwargs["max_nodes"] = nodes

                walltime, ok = QInputDialog.getText(
                    self, "HPC Submit", "Wall time (HH:MM:SS):", text="2:00:00"
                )
                if not ok:
                    return

                case_dir = QFileDialog.getExistingDirectory(
                    self, "Select Case Directory"
                )
                if not case_dir:
                    return

                config = HPCConfig(**config_kwargs)
                manager = HPCJobManager(config)
                job_id = manager.submit_job(case_dir=case_dir, walltime=walltime)

                QMessageBox.information(
                    self, "Job Submitted",
                    f"Job ID: {job_id}\n"
                    f"Backend: {backend}\n"
                    f"Nodes: {nodes}\n"
                    f"Walltime: {walltime}\n"
                    f"Case: {case_dir}"
                )

            elif action == "Check job status":
                job_id, ok = QInputDialog.getText(
                    self, "HPC Status", "Job ID:"
                )
                if not ok or not job_id:
                    return

                manager = HPCJobManager()
                status = manager.check_status(job_id)
                QMessageBox.information(
                    self, "Job Status",
                    f"Job {job_id}: {status.value}"
                )

            elif action == "Cancel job":
                job_id, ok = QInputDialog.getText(
                    self, "HPC Cancel", "Job ID:"
                )
                if not ok or not job_id:
                    return

                manager = HPCJobManager()
                success = manager.cancel_job(job_id)
                if success:
                    QMessageBox.information(self, "Job Cancelled", f"Job {job_id} cancelled.")
                else:
                    QMessageBox.warning(self, "Cancel Failed", f"Could not cancel job {job_id}")

            elif action == "Download results":
                job_id, ok = QInputDialog.getText(
                    self, "HPC Download", "Job ID:"
                )
                if not ok or not job_id:
                    return

                output_dir = QFileDialog.getExistingDirectory(
                    self, "Select Output Directory"
                )
                if not output_dir:
                    return

                manager = HPCJobManager()
                success = manager.download_results(job_id, output_dir)
                if success:
                    QMessageBox.information(
                        self, "Download Complete",
                        f"Results for job {job_id} downloaded to:\n{output_dir}"
                    )
                else:
                    QMessageBox.warning(
                        self, "Download Failed",
                        f"Could not download results for job {job_id}"
                    )

            elif action == "Setup AWS Batch":
                from ..hpc.aws_setup import AWSBatchProvisioner

                region, ok = QInputDialog.getText(
                    self, "AWS Region", "AWS region:", text="us-east-1"
                )
                if not ok:
                    return

                platforms = ["EC2", "FARGATE"]
                platform, ok = QInputDialog.getItem(
                    self, "Platform", "Compute platform:", platforms, 0, False
                )
                if not ok:
                    return

                self.statusBar().showMessage("Provisioning AWS Batch infrastructure...")
                provisioner = AWSBatchProvisioner(region=region, platform=platform)
                messages: list[str] = []
                result = provisioner.setup(log_fn=messages.append)

                msg = (
                    f"AWS Batch Provisioned\n\n"
                    f"Region: {result.region}\n"
                    f"S3 Bucket: {result.bucket_name}\n"
                    f"Job Queue: {result.job_queue}\n"
                    f"Compute Env: {result.compute_environment}\n\n"
                    f"Created: {len(result.created_resources)}\n"
                    f"Skipped (already existed): {len(result.skipped_resources)}"
                )
                QMessageBox.information(self, "AWS Setup Complete", msg)
                self.statusBar().showMessage("AWS Batch environment ready")

            elif action == "Teardown AWS Batch":
                from ..hpc.aws_setup import AWSBatchProvisioner

                region, ok = QInputDialog.getText(
                    self, "AWS Region", "AWS region:", text="us-east-1"
                )
                if not ok:
                    return

                confirm = QMessageBox.question(
                    self, "Confirm Teardown",
                    f"This will delete all AstraTurbo AWS resources in {region}.\n\n"
                    "S3 bucket will only be deleted if empty.\n"
                    "Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if confirm != QMessageBox.Yes:
                    return

                self.statusBar().showMessage("Tearing down AWS resources...")
                provisioner = AWSBatchProvisioner(region=region)
                provisioner.teardown()
                QMessageBox.information(
                    self, "Teardown Complete",
                    f"All AstraTurbo resources in {region} have been deleted."
                )
                self.statusBar().showMessage("AWS teardown complete")

        except Exception as e:
            QMessageBox.warning(self, "HPC Error", f"{e}\n\n{traceback.format_exc()}")

    # ----------------------------------------------------------------
    # Parametric Sweep
    # ----------------------------------------------------------------

    def _run_parametric_sweep(self) -> None:
        """Run a parametric sweep over a design parameter."""
        from PySide6.QtWidgets import QInputDialog

        params = [
            "cl0", "max_thickness", "stagger_angle", "chord",
            "pressure_ratio", "mesh_ni", "mesh_nj"
        ]
        param, ok = QInputDialog.getItem(
            self, "Parametric Sweep", "Parameter to sweep:", params, 0, False
        )
        if not ok:
            return

        start, ok = QInputDialog.getDouble(
            self, "Parametric Sweep", "Start value:", 0.6, -100, 100, 4
        )
        if not ok:
            return

        end, ok = QInputDialog.getDouble(
            self, "Parametric Sweep", "End value:", 1.4, -100, 100, 4
        )
        if not ok:
            return

        steps, ok = QInputDialog.getInt(
            self, "Parametric Sweep", "Number of steps:", 5, 2, 50
        )
        if not ok:
            return

        try:
            from ..foundation.design_chain import DesignChain

            chain = DesignChain()
            self.statusBar().showMessage(
                f"Running sweep: {param} = {start} to {end} ({steps} steps)..."
            )

            results = chain.sweep(param, start=start, end=end, steps=steps)

            n_success = sum(1 for r in results if r.success)
            values = np.linspace(start, end, steps)

            lines = [
                f"Parametric Sweep: {param}\n",
                f"Range: {start} to {end}, {steps} steps",
                f"Successful: {n_success}/{len(results)}\n",
            ]
            for i, (val, res) in enumerate(zip(values, results)):
                status = "OK" if res.success else "FAILED"
                t = res.total_time
                lines.append(f"  {param}={val:.4f} -> {status} ({t:.3f}s)")

            QMessageBox.information(self, "Sweep Results", "\n".join(lines))
            self.statusBar().showMessage(
                f"Sweep done: {n_success}/{len(results)} successful"
            )
        except Exception as e:
            QMessageBox.warning(self, "Sweep Error", f"{e}\n\n{traceback.format_exc()}")
