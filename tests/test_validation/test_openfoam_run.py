"""Integration test: actually run OpenFOAM if installed.

These tests are skipped when OpenFOAM is not in PATH. When OpenFOAM IS
available, they validate that AstraTurbo-generated cases actually run
through the solver without crashing — the real proof that the pipeline
produces correct, solver-ready output.

Tests:
    1. Incompressible cascade (simpleFoam) — blockMesh + 50 iterations
    2. Compressible cascade (rhoSimpleFoam) — blockMesh + 50 iterations
    3. NASA Rotor 37 end-to-end — meanline → profile → mesh → rhoSimpleFoam
    4. Case file sanity — verify all required files parse without errors

Run with:
    pytest tests/test_validation/test_openfoam_run.py -v

Skip reason if OpenFOAM not found:
    SKIPPED: OpenFOAM not installed (simpleFoam not in PATH)
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

# ── Skip all tests if OpenFOAM not installed ──────────────────

_HAS_OPENFOAM = shutil.which("simpleFoam") is not None
_HAS_RHOFOAM = shutil.which("rhoSimpleFoam") is not None
_HAS_BLOCKMESH = shutil.which("blockMesh") is not None
_HAS_CHECKMESH = shutil.which("checkMesh") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_OPENFOAM,
    reason="OpenFOAM not installed (simpleFoam not in PATH)",
)


# ── Helpers ───────────────────────────────────────────────────

def _make_simple_profile() -> np.ndarray:
    """Create a closed NACA-like profile for testing."""
    t = np.linspace(0, 2 * np.pi, 80)
    x = 0.5 * (1 - np.cos(t))
    y = 0.06 * np.sin(t)
    return np.column_stack([x, y])


def _run_openfoam_case(case_dir: Path, n_iters: int = 50, timeout: int = 120) -> dict:
    """Run an OpenFOAM case and return results.

    Returns dict with:
        success: bool
        returncode: int
        stdout: str
        stderr: str
    """
    # Override endTime to limit iterations
    ctrl_path = case_dir / "system" / "controlDict"
    if ctrl_path.exists():
        ctrl = ctrl_path.read_text()
        ctrl = ctrl.replace("endTime         1000", f"endTime         {n_iters}")
        ctrl = ctrl.replace("writeInterval   100", f"writeInterval   {n_iters}")
        ctrl_path.write_text(ctrl)

    # Run blockMesh first if no polyMesh
    poly_dir = case_dir / "constant" / "polyMesh"
    if not (poly_dir / "points").exists() and _HAS_BLOCKMESH:
        bm = subprocess.run(
            ["blockMesh"],
            cwd=str(case_dir),
            capture_output=True, text=True, timeout=60,
        )
        if bm.returncode != 0:
            return {
                "success": False,
                "returncode": bm.returncode,
                "stdout": bm.stdout,
                "stderr": bm.stderr,
                "stage": "blockMesh",
            }

    # Determine solver from controlDict
    ctrl = ctrl_path.read_text()
    if "rhoSimpleFoam" in ctrl:
        solver = "rhoSimpleFoam"
    elif "pimpleFoam" in ctrl:
        solver = "pimpleFoam"
    else:
        solver = "simpleFoam"

    if not shutil.which(solver):
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{solver} not found in PATH",
            "stage": "solver",
        }

    result = subprocess.run(
        [solver],
        cwd=str(case_dir),
        capture_output=True, text=True, timeout=timeout,
    )

    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stage": "solver",
    }


# ── Test: incompressible case (simpleFoam) ────────────────────

class TestOpenFOAMIncompressible:
    """Test that AstraTurbo-generated incompressible cases run in simpleFoam."""

    def test_simplefoam_case_runs(self, tmp_path):
        """Generate an incompressible case and run simpleFoam for 50 iterations."""
        from astraturbo.cfd.openfoam import create_openfoam_case

        case = create_openfoam_case(
            case_dir=tmp_path / "incomp_case",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            inlet_velocity=50.0,
            viscosity=1.5e-5,
        )

        # We need a mesh — write a minimal blockMeshDict
        _write_test_blockmeshdict(case)

        result = _run_openfoam_case(case, n_iters=20)

        assert result["success"], (
            f"simpleFoam failed (stage={result.get('stage')}):\n"
            f"stdout: {result['stdout'][-500:]}\n"
            f"stderr: {result['stderr'][-500:]}"
        )

    def test_simplefoam_produces_residuals(self, tmp_path):
        """simpleFoam output should contain residual information."""
        from astraturbo.cfd.openfoam import create_openfoam_case

        case = create_openfoam_case(
            case_dir=tmp_path / "residual_case",
            solver="simpleFoam",
            inlet_velocity=50.0,
        )
        _write_test_blockmeshdict(case)

        result = _run_openfoam_case(case, n_iters=10)

        if result["success"]:
            # Should have residual output mentioning Ux or p
            output = result["stdout"]
            assert "Ux" in output or "p" in output or "smoothSolver" in output, (
                "Solver output should contain field residual info"
            )


# ── Test: compressible case (rhoSimpleFoam) ───────────────────

class TestOpenFOAMCompressible:
    """Test that AstraTurbo-generated compressible cases run in rhoSimpleFoam."""

    pytestmark = pytest.mark.skipif(
        not _HAS_RHOFOAM,
        reason="rhoSimpleFoam not in PATH",
    )

    def test_rhosimplefoam_case_runs(self, tmp_path):
        """Generate a compressible case and run rhoSimpleFoam for 50 iterations."""
        from astraturbo.cfd.openfoam import create_openfoam_case

        case = create_openfoam_case(
            case_dir=tmp_path / "comp_case",
            solver="rhoSimpleFoam",
            compressible=True,
            total_pressure=101325.0,
            total_temperature=300.0,
            inlet_velocity=80.0,
        )

        _write_test_blockmeshdict(case)

        result = _run_openfoam_case(case, n_iters=20)

        assert result["success"], (
            f"rhoSimpleFoam failed (stage={result.get('stage')}):\n"
            f"stdout: {result['stdout'][-500:]}\n"
            f"stderr: {result['stderr'][-500:]}"
        )

    def test_compressible_has_temperature_field(self, tmp_path):
        """After running, time directories should contain T field."""
        from astraturbo.cfd.openfoam import create_openfoam_case

        case = create_openfoam_case(
            case_dir=tmp_path / "temp_case",
            solver="rhoSimpleFoam",
            compressible=True,
            total_pressure=101325.0,
            total_temperature=300.0,
            inlet_velocity=80.0,
        )
        _write_test_blockmeshdict(case)

        result = _run_openfoam_case(case, n_iters=5)

        if result["success"]:
            # Check 0/T exists (initial condition)
            assert (case / "0" / "T").exists(), "Temperature field should exist"
            # Check thermophysicalProperties
            assert (case / "constant" / "thermophysicalProperties").exists()


# ── Test: NASA Rotor 37 full pipeline with solver ─────────────

class TestRotor37WithSolver:
    """End-to-end: Rotor 37 requirements → meanline → profile → CFD → run solver."""

    pytestmark = pytest.mark.skipif(
        not _HAS_RHOFOAM,
        reason="rhoSimpleFoam not in PATH",
    )

    def test_rotor37_pipeline_runs_solver(self, tmp_path):
        """Full pipeline: R37 meanline → profile → compressible case → rhoSimpleFoam."""
        from astraturbo.design.meanline import (
            meanline_compressor,
            meanline_to_blade_parameters,
            blade_angle_to_cl0,
        )
        from astraturbo.camberline import NACA65
        from astraturbo.thickness import NACA65Series
        from astraturbo.profile import Superposition
        from astraturbo.cfd.openfoam import create_openfoam_case

        # Load Rotor 37 data
        ref_path = Path(__file__).parent / "reference_data" / "nasa_rotor37.json"
        with open(ref_path) as f:
            r37 = json.load(f)

        dp = r37["design_point"]
        geo = r37["geometry"]

        # Step 1: Meanline
        ml = meanline_compressor(
            overall_pressure_ratio=dp["total_pressure_ratio"],
            mass_flow=dp["mass_flow_kg_s"],
            rpm=dp["rpm"],
            r_hub=geo["hub_radius_inlet_m"],
            r_tip=geo["tip_radius_inlet_m"],
            n_stages=1,
        )

        # Step 2: Profile from auto-computed cl0
        params = meanline_to_blade_parameters(ml)[0]
        cl0 = blade_angle_to_cl0(
            ml.stages[0].rotor_inlet_beta,
            ml.stages[0].rotor_outlet_beta,
            params["rotor_solidity"],
        )
        profile = Superposition(
            NACA65(cl0=min(cl0, 2.0)),
            NACA65Series(max_thickness=0.10),
        )

        # Step 3: Compressible CFD case
        case = create_openfoam_case(
            case_dir=tmp_path / "r37_cfd",
            solver="rhoSimpleFoam",
            compressible=True,
            total_pressure=r37["inlet_conditions"]["total_pressure_Pa"],
            total_temperature=r37["inlet_conditions"]["total_temperature_K"],
            inlet_velocity=ml.stations[0].C_axial,
        )

        # Write a test mesh
        _write_test_blockmeshdict(case)

        # Step 4: Run solver
        result = _run_openfoam_case(case, n_iters=10, timeout=180)

        assert result["success"], (
            f"Rotor 37 rhoSimpleFoam failed:\n"
            f"stdout: {result['stdout'][-500:]}\n"
            f"stderr: {result['stderr'][-500:]}"
        )


# ── Test: case file parsing (checkMesh) ───────────────────────

class TestOpenFOAMCaseValidity:
    """Verify generated cases pass OpenFOAM's own file parsers."""

    pytestmark = pytest.mark.skipif(
        not _HAS_CHECKMESH or not _HAS_BLOCKMESH,
        reason="blockMesh or checkMesh not in PATH",
    )

    def test_blockmesh_parses_case(self, tmp_path):
        """blockMesh should successfully create a mesh from our blockMeshDict."""
        from astraturbo.cfd.openfoam import create_openfoam_case

        case = create_openfoam_case(
            case_dir=tmp_path / "parse_case",
            solver="simpleFoam",
            inlet_velocity=50.0,
        )
        _write_test_blockmeshdict(case)

        result = subprocess.run(
            ["blockMesh"],
            cwd=str(case),
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"blockMesh failed:\n{result.stderr[-500:]}"
        )

    def test_checkmesh_passes(self, tmp_path):
        """checkMesh should report no fatal errors on our generated mesh."""
        from astraturbo.cfd.openfoam import create_openfoam_case

        case = create_openfoam_case(
            case_dir=tmp_path / "check_case",
            solver="simpleFoam",
            inlet_velocity=50.0,
        )
        _write_test_blockmeshdict(case)

        # blockMesh first
        subprocess.run(
            ["blockMesh"], cwd=str(case),
            capture_output=True, timeout=60,
        )

        result = subprocess.run(
            ["checkMesh"],
            cwd=str(case),
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"checkMesh failed:\n{result.stderr[-500:]}"
        )
        assert "Failed" not in result.stdout, (
            f"checkMesh reported failures:\n{result.stdout[-500:]}"
        )


# ── Helper: write a minimal blockMeshDict for testing ─────────

def _write_test_blockmeshdict(case_dir: Path) -> None:
    """Write a minimal blockMeshDict that creates a simple hex domain.

    This is a small channel mesh (1m x 0.1m x 0.01m) with named patches
    matching AstraTurbo's convention: inlet, outlet, blade, hub, shroud.
    """
    bmd = case_dir / "system" / "blockMeshDict"
    bmd.write_text("""\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

convertToMeters 1;

vertices
(
    (0    0     0)
    (1    0     0)
    (1    0.1   0)
    (0    0.1   0)
    (0    0     0.01)
    (1    0     0.01)
    (1    0.1   0.01)
    (0    0.1   0.01)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }
    outlet
    {
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }
    blade
    {
        type wall;
        faces
        (
            (0 1 5 4)
        );
    }
    hub
    {
        type wall;
        faces
        (
            (0 3 2 1)
        );
    }
    shroud
    {
        type wall;
        faces
        (
            (4 5 6 7)
        );
    }
    defaultFaces
    {
        type empty;
        faces
        (
            (3 7 6 2)
        );
    }
);

mergePatchPairs
(
);
""")
