"""Tests for CFD workflow integration."""

import pytest
from pathlib import Path

from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig


class TestCFDWorkflow:
    def test_openfoam_setup(self, tmp_path):
        cfg = CFDWorkflowConfig(
            solver="openfoam", inlet_velocity=100, turbulence_model="kOmegaSST",
        )
        wf = CFDWorkflow(cfg)
        wf.set_mesh("test.cgns")
        case = wf.setup_case(tmp_path / "of_case")

        assert (case / "system" / "controlDict").exists()
        assert (case / "system" / "fvSchemes").exists()
        assert (case / "0" / "U").exists()
        assert (case / "Allrun").exists()

        # Check Allrun contains cgnsToFoam
        allrun = (case / "Allrun").read_text()
        assert "cgnsToFoam" in allrun

    def test_openfoam_rotating(self, tmp_path):
        cfg = CFDWorkflowConfig(
            solver="openfoam", is_rotating=True, omega=1200.0,
        )
        wf = CFDWorkflow(cfg)
        case = wf.setup_case(tmp_path / "rotating_case")

        assert (case / "constant" / "MRFProperties").exists()
        mrf = (case / "constant" / "MRFProperties").read_text()
        assert "1200" in mrf

    def test_fluent_setup(self, tmp_path):
        cfg = CFDWorkflowConfig(solver="fluent", inlet_velocity=80)
        wf = CFDWorkflow(cfg)
        wf.set_mesh("blade.msh")
        case = wf.setup_case(tmp_path / "fluent_case")

        assert (case / "run.jou").exists()
        journal = (case / "run.jou").read_text()
        assert "kw-sst" in journal
        assert "80" in journal

    def test_cfx_setup(self, tmp_path):
        cfg = CFDWorkflowConfig(
            solver="cfx", inlet_velocity=120, is_rotating=True, omega=1500,
        )
        wf = CFDWorkflow(cfg)
        case = wf.setup_case(tmp_path / "cfx_case")

        assert (case / "setup.ccl").exists()
        ccl = (case / "setup.ccl").read_text()
        assert "Rotating" in ccl
        assert "1500" in ccl
        assert "SST" in ccl

    def test_su2_setup(self, tmp_path):
        cfg = CFDWorkflowConfig(solver="su2", inlet_velocity=100)
        wf = CFDWorkflow(cfg)
        case = wf.setup_case(tmp_path / "su2_case")

        assert (case / "astraturbo.cfg").exists()
        assert (case / "run_su2.sh").exists()

    def test_run_without_setup(self):
        wf = CFDWorkflow()
        result = wf.run()
        assert not result.success
        assert "not set up" in result.error_message
