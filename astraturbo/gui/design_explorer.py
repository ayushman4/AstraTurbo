"""Design Space Explorer — interactive slider-driven design tool.

Provides a QDialog with sliders for key design parameters that recompute
the solver in a background thread, giving near-instant feedback on how
changes to PR, mass flow, RPM, etc. affect performance.

Modes:
  - Compressor: axial compressor meanline
  - Turbine: axial turbine meanline
  - Engine Cycle: full turbojet/turboshaft cycle
"""

from __future__ import annotations

import traceback
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QDoubleSpinBox,
    QTextEdit,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QFileDialog,
    QMessageBox,
)


class _SolverWorker(QObject):
    """Runs the design solver in a background thread."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, mode: str, params: dict):
        super().__init__()
        self.mode = mode
        self.params = params

    def run(self) -> None:
        try:
            if self.mode == "Compressor":
                result = self._run_compressor()
            elif self.mode == "Turbine":
                result = self._run_turbine()
            elif self.mode == "Engine Cycle":
                result = self._run_cycle()
            else:
                result = "Unknown mode"
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")

    def _run_compressor(self) -> str:
        from ..design import meanline_compressor

        result = meanline_compressor(
            overall_pressure_ratio=self.params["pr"],
            mass_flow=self.params["mass_flow"],
            rpm=self.params["rpm"],
            r_hub=self.params.get("r_hub", 0.15),
            r_tip=self.params.get("r_tip", 0.30),
        )
        return result.summary()

    def _run_turbine(self) -> str:
        from ..design.turbine import meanline_turbine

        result = meanline_turbine(
            overall_expansion_ratio=self.params["pr"],
            mass_flow=self.params["mass_flow"],
            rpm=self.params["rpm"],
            r_hub=self.params.get("r_hub", 0.25),
            r_tip=self.params.get("r_tip", 0.35),
            T_inlet=self.params.get("tit", 1500.0),
        )
        return result.summary()

    def _run_cycle(self) -> str:
        from ..design.engine_cycle import engine_cycle

        kwargs = dict(
            overall_pressure_ratio=self.params["pr"],
            turbine_inlet_temp=self.params.get("tit", 1400.0),
            mass_flow=self.params["mass_flow"],
            rpm=self.params["rpm"],
            r_hub=self.params.get("r_hub", 0.15),
            r_tip=self.params.get("r_tip", 0.30),
            altitude=self.params.get("altitude", 0.0),
            mach_flight=self.params.get("mach", 0.0),
        )
        n_spools = self.params.get("n_spools", 1)
        if n_spools == 2:
            kwargs["n_spools"] = 2
            if "hp_pr" in self.params and self.params["hp_pr"] > 0:
                kwargs["hp_pressure_ratio"] = self.params["hp_pr"]
            if "hp_rpm" in self.params and self.params["hp_rpm"] > 0:
                kwargs["hp_rpm"] = self.params["hp_rpm"]

        result = engine_cycle(**kwargs)
        return result.summary()


class DesignExplorerDialog(QDialog):
    """Interactive design space explorer with live-updating sliders."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Design Space Explorer")
        self.setMinimumSize(700, 600)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_SolverWorker] = None

        # Debounce timer — avoids recomputing on every pixel of slider drag
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._compute)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self._mode = QComboBox()
        self._mode.addItems(["Compressor", "Turbine", "Engine Cycle"])
        self._mode.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Sliders group
        slider_group = QGroupBox("Design Parameters")
        self._slider_grid = QGridLayout(slider_group)

        self._sliders: dict[str, tuple[QSlider, QDoubleSpinBox, QLabel]] = {}
        self._slider_defs = {
            "pr": ("PR / ER / OPR", 1.5, 30.0, 4.0, 0.1),
            "mass_flow": ("Mass Flow (kg/s)", 1.0, 100.0, 20.0, 0.5),
            "rpm": ("RPM", 1000.0, 60000.0, 15000.0, 100.0),
            "tit": ("TIT (K)", 800.0, 2200.0, 1500.0, 10.0),
            "altitude": ("Altitude (m)", 0.0, 15000.0, 0.0, 100.0),
            "mach": ("Mach Number", 0.0, 2.0, 0.0, 0.01),
            "hp_pr": ("HP Pressure Ratio", 1.5, 15.0, 4.0, 0.1),
            "hp_rpm": ("HP RPM", 1000.0, 60000.0, 20000.0, 100.0),
        }

        for row, (key, (label, vmin, vmax, default, step)) in enumerate(self._slider_defs.items()):
            lbl = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            n_steps = int((vmax - vmin) / step)
            slider.setRange(0, n_steps)
            slider.setValue(int((default - vmin) / step))
            slider.valueChanged.connect(self._on_slider_moved)

            spin = QDoubleSpinBox()
            spin.setRange(vmin, vmax)
            spin.setSingleStep(step)
            spin.setDecimals(2 if step < 1 else 0)
            spin.setValue(default)
            spin.valueChanged.connect(lambda val, s=slider, mn=vmin, st=step: s.setValue(int((val - mn) / st)))

            slider.valueChanged.connect(
                lambda val, sp=spin, mn=vmin, st=step: sp.setValue(mn + val * st)
            )

            self._sliders[key] = (slider, spin, lbl)
            self._slider_grid.addWidget(lbl, row, 0)
            self._slider_grid.addWidget(slider, row, 1)
            self._slider_grid.addWidget(spin, row, 2)

        layout.addWidget(slider_group)

        # N-spools selector (Engine Cycle only)
        spool_layout = QHBoxLayout()
        self._spool_label = QLabel("Spools:")
        spool_layout.addWidget(self._spool_label)
        self._spool_combo = QComboBox()
        self._spool_combo.addItems(["1 (Single)", "2 (Twin)"])
        self._spool_combo.currentIndexChanged.connect(self._on_spool_changed)
        spool_layout.addWidget(self._spool_combo)
        spool_layout.addStretch()
        layout.addLayout(spool_layout)

        # Results panel
        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setFontFamily("monospace")
        self._results.setPlaceholderText("Adjust sliders to see results...")
        layout.addWidget(self._results)

        # Buttons
        btn_layout = QHBoxLayout()
        compute_btn = QPushButton("Compute Now")
        compute_btn.clicked.connect(self._compute)
        btn_layout.addWidget(compute_btn)

        export_btn = QPushButton("Export Report...")
        export_btn.clicked.connect(self._export_report)
        btn_layout.addWidget(export_btn)

        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._on_mode_changed(self._mode.currentText())

    def _on_mode_changed(self, mode: str) -> None:
        """Show/hide sliders based on mode."""
        cycle_only = {"tit", "altitude", "mach"}
        turbine_only = {"tit"}
        twin_spool_only = {"hp_pr", "hp_rpm"}

        is_cycle = mode == "Engine Cycle"
        is_twin = is_cycle and self._spool_combo.currentIndex() == 1

        for key, (slider, spin, lbl) in self._sliders.items():
            if key in twin_spool_only:
                visible = is_twin
            elif mode == "Engine Cycle":
                visible = True
            elif mode == "Turbine":
                visible = key not in cycle_only or key in turbine_only
            else:  # Compressor
                visible = key not in cycle_only
            slider.setVisible(visible)
            spin.setVisible(visible)
            lbl.setVisible(visible)

        # Show/hide spool selector
        self._spool_label.setVisible(is_cycle)
        self._spool_combo.setVisible(is_cycle)

        # Update PR label
        lbl = self._sliders["pr"][2]
        if mode == "Compressor":
            lbl.setText("Pressure Ratio")
        elif mode == "Turbine":
            lbl.setText("Expansion Ratio")
        else:
            lbl.setText("OPR")

        self._schedule_compute()

    def _on_spool_changed(self, _index: int) -> None:
        """Show/hide HP spool sliders when spool count changes."""
        self._on_mode_changed(self._mode.currentText())

    def _on_slider_moved(self) -> None:
        self._schedule_compute()

    def _schedule_compute(self) -> None:
        self._debounce.start()

    def _get_params(self) -> dict:
        params = {}
        for key, (_, spin, _) in self._sliders.items():
            params[key] = spin.value()
        params["n_spools"] = self._spool_combo.currentIndex() + 1
        return params

    def _compute(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return  # already computing

        mode = self._mode.currentText()
        params = self._get_params()
        self._results.setPlainText("Computing...")

        self._worker = _SolverWorker(mode, params)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_result(self, text: str) -> None:
        self._results.setPlainText(text)

    def _on_error(self, text: str) -> None:
        self._results.setPlainText(f"Error: {text}")

    def _export_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "explorer_report.html", "HTML (*.html)"
        )
        if not path:
            return

        mode = self._mode.currentText()
        params = self._get_params()

        try:
            from ..reports import generate_report, ReportConfig

            cfg = ReportConfig(
                title=f"Design Explorer — {mode}",
                output_path=path,
            )

            kwargs = {}
            if mode == "Compressor":
                from ..design import meanline_compressor
                result = meanline_compressor(
                    overall_pressure_ratio=params["pr"],
                    mass_flow=params["mass_flow"],
                    rpm=params["rpm"],
                    r_hub=0.15, r_tip=0.30,
                )
                kwargs["meanline_result"] = result
            elif mode == "Turbine":
                from ..design.turbine import meanline_turbine
                result = meanline_turbine(
                    overall_expansion_ratio=params["pr"],
                    mass_flow=params["mass_flow"],
                    rpm=params["rpm"],
                    r_hub=0.25, r_tip=0.35,
                    T_inlet=params.get("tit", 1500.0),
                )
                kwargs["turbine_result"] = result
            elif mode == "Engine Cycle":
                from ..design.engine_cycle import engine_cycle
                ec_kwargs = dict(
                    overall_pressure_ratio=params["pr"],
                    turbine_inlet_temp=params.get("tit", 1400.0),
                    mass_flow=params["mass_flow"],
                    rpm=params["rpm"],
                    r_hub=0.15, r_tip=0.30,
                )
                n_spools = params.get("n_spools", 1)
                if n_spools == 2:
                    ec_kwargs["n_spools"] = 2
                    if "hp_pr" in params and params["hp_pr"] > 0:
                        ec_kwargs["hp_pressure_ratio"] = params["hp_pr"]
                    if "hp_rpm" in params and params["hp_rpm"] > 0:
                        ec_kwargs["hp_rpm"] = params["hp_rpm"]
                result = engine_cycle(**ec_kwargs)
                kwargs["engine_cycle_result"] = result

            generate_report(config=cfg, **kwargs)
            QMessageBox.information(self, "Export", f"Report saved: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"{e}\n\n{traceback.format_exc()}")
