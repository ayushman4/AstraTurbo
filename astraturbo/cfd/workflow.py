"""End-to-end CFD workflow orchestration.

Manages the complete pipeline: geometry → mesh → case setup → solver → post-process.
Supports OpenFOAM, Fluent, and CFX through a unified interface.
"""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class CFDWorkflowConfig:
    """Configuration for a complete CFD workflow."""

    # Solver
    solver: Literal["openfoam", "fluent", "cfx", "su2"] = "openfoam"

    # Flow conditions
    inlet_velocity: float = 100.0        # m/s
    inlet_temperature: float = 288.15    # K
    inlet_pressure: float = 101325.0     # Pa
    outlet_pressure: float = 101325.0    # Pa

    # Compressible flow
    compressible: bool = False
    total_pressure: float = 101325.0     # Pa (inlet total pressure)
    total_temperature: float = 288.15    # K (inlet total temperature)

    # Fluid
    density: float = 1.225               # kg/m³
    viscosity: float = 1.8e-5            # Pa·s (dynamic)
    kinematic_viscosity: float = 1.47e-5 # m²/s

    # Turbulence
    turbulence_model: str = "kOmegaSST"
    turbulent_intensity: float = 0.05    # 5%
    turbulent_length_scale: float = 0.01 # m

    # Solver settings
    n_iterations: int = 2000
    convergence_target: float = 1e-5
    n_procs: int = 1

    # Rotating frame (for rotors)
    is_rotating: bool = False
    omega: float = 0.0                   # rad/s
    rotation_axis: list[float] = field(default_factory=lambda: [0, 0, 1])
    rotation_origin: list[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class CFDWorkflowResult:
    """Result from a CFD workflow execution."""

    success: bool = False
    case_dir: str = ""
    log_file: str = ""
    error_message: str = ""

    # Extracted performance (populated after post-processing)
    pressure_ratio: float | None = None
    mass_flow: float | None = None
    efficiency: float | None = None
    residuals: dict | None = None


class CFDWorkflow:
    """Orchestrates the full CFD pipeline.

    Usage::

        wf = CFDWorkflow(config)
        wf.set_mesh("mesh.cgns")
        wf.setup_case("my_case/")
        result = wf.run()
        print(result.pressure_ratio)
    """

    def __init__(self, config: CFDWorkflowConfig | None = None) -> None:
        self.config = config or CFDWorkflowConfig()
        self._mesh_path: str | None = None
        self._case_dir: Path | None = None

    def set_mesh(self, mesh_path: str | Path) -> None:
        """Set the mesh file to use."""
        self._mesh_path = str(Path(mesh_path).resolve())

    def setup_case(self, case_dir: str | Path) -> Path:
        """Set up the complete CFD case directory.

        Creates all necessary files for the selected solver.

        Returns:
            Path to the case directory.
        """
        self._case_dir = Path(case_dir)
        cfg = self.config

        # Verify mesh file exists if specified
        if self._mesh_path and not Path(self._mesh_path).exists():
            import warnings
            warnings.warn(
                f"Mesh file not found: {self._mesh_path}. "
                "Case will be set up but mesh import may fail.",
                stacklevel=2,
            )

        if cfg.solver == "openfoam":
            return self._setup_openfoam()
        elif cfg.solver == "fluent":
            return self._setup_fluent()
        elif cfg.solver == "cfx":
            return self._setup_cfx()
        elif cfg.solver == "su2":
            return self._setup_su2()
        else:
            raise ValueError(f"Unknown solver: {cfg.solver}")

    def run(self) -> CFDWorkflowResult:
        """Execute the CFD solver."""
        if self._case_dir is None:
            return CFDWorkflowResult(
                success=False, error_message="Case not set up. Call setup_case() first."
            )

        cfg = self.config

        # Check solver availability before attempting to run
        solver_check = self._check_solver_available()
        if solver_check:
            return CFDWorkflowResult(
                success=False,
                case_dir=str(self._case_dir),
                error_message=solver_check,
            )

        if cfg.solver == "openfoam":
            return self._run_openfoam()
        elif cfg.solver == "su2":
            return self._run_su2()
        else:
            return CFDWorkflowResult(
                success=False,
                error_message=f"Automated execution not supported for {cfg.solver}. "
                              f"Case files are ready at {self._case_dir}",
                case_dir=str(self._case_dir),
            )

    def postprocess(self, result: CFDWorkflowResult) -> CFDWorkflowResult:
        """Extract performance metrics from solver output."""
        if not result.success:
            return result

        cfg = self.config
        if cfg.solver == "openfoam":
            return self._postprocess_openfoam(result)

        return result

    # ── Solver checks ──────────────────────────────────────

    def _check_solver_available(self) -> str | None:
        """Check if the solver executable is available in PATH.

        Returns:
            Error message string if solver not found, None if available.
        """
        cfg = self.config
        solver_executables = {
            "openfoam": "simpleFoam",
            "su2": "SU2_CFD",
            "fluent": "fluent",
            "cfx": "cfx5solve",
        }
        exe = solver_executables.get(cfg.solver)
        if exe is None:
            return None

        if shutil.which(exe) is None:
            return (
                f"{cfg.solver} solver not found in PATH (looked for '{exe}'). "
                f"Install {cfg.solver} or add it to your PATH. "
                f"Case files are ready at {self._case_dir}."
            )
        return None

    # ── OpenFOAM ──────────────────────────────────────────

    def _setup_openfoam(self) -> Path:
        from ..cfd.openfoam import create_openfoam_case
        cfg = self.config

        # Determine solver based on compressible flag
        if cfg.compressible:
            of_solver = "rhoSimpleFoam"
        elif cfg.is_rotating:
            of_solver = "pimpleFoam"
        else:
            of_solver = "simpleFoam"

        # Collect patch names from mesh if generated by AstraTurbo
        patch_names = None
        # (patch_names can be set externally before calling setup_case)

        case = create_openfoam_case(
            case_dir=self._case_dir,
            solver=of_solver,
            turbulence_model=cfg.turbulence_model,
            inlet_velocity=cfg.inlet_velocity,
            viscosity=cfg.kinematic_viscosity,
            patch_names=patch_names,
            compressible=cfg.compressible,
            total_pressure=cfg.total_pressure if cfg.compressible else 101325.0,
            total_temperature=cfg.total_temperature if cfg.compressible else 288.15,
        )

        # If mesh is CGNS, write conversion script with fallback
        if self._mesh_path and self._mesh_path.endswith(".cgns"):
            # Security: quote path to prevent shell injection
            import shlex
            safe_path = shlex.quote(self._mesh_path)
            script = self._case_dir / "import_mesh.sh"
            with open(script, "w") as f:
                f.write("#!/bin/bash\nset -e\n")
                f.write(f"if command -v cgnsToFoam &>/dev/null; then\n")
                f.write(f"    cgnsToFoam {safe_path}\n")
                f.write(f"elif [ -f system/blockMeshDict ]; then\n")
                f.write(f"    blockMesh\n")
                f.write(f"else\n")
                f.write(f"    echo 'ERROR: cgnsToFoam not available and no blockMeshDict found.'\n")
                f.write(f"    exit 1\n")
                f.write(f"fi\n")
                f.write("checkMesh\n")
            script.chmod(0o755)

        # If rotating, add MRF zone + topoSetDict to create the cellZone
        if cfg.is_rotating:
            mrf_path = self._case_dir / "constant" / "MRFProperties"
            with open(mrf_path, "w") as f:
                f.write("FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\n")
                f.write("MRF1\n{\n")
                f.write("    cellZone    rotatingZone;\n")
                f.write("    active      yes;\n")
                f.write(f"    origin      ({cfg.rotation_origin[0]} {cfg.rotation_origin[1]} {cfg.rotation_origin[2]});\n")
                f.write(f"    axis        ({cfg.rotation_axis[0]} {cfg.rotation_axis[1]} {cfg.rotation_axis[2]});\n")
                f.write(f"    omega       {cfg.omega};\n")
                f.write("}\n")

            # Write topoSetDict: creates "rotatingZone" cellZone from ALL cells.
            # This is required because blockMesh doesn't create cellZones.
            topo_path = self._case_dir / "system" / "topoSetDict"
            with open(topo_path, "w") as f:
                f.write("FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }\n\n")
                f.write("actions\n(\n")
                f.write("    {\n")
                f.write("        name    rotatingZone;\n")
                f.write("        type    cellSet;\n")
                f.write("        action  new;\n")
                f.write("        source  boxToCell;\n")
                f.write("        box     (-1e10 -1e10 -1e10) (1e10 1e10 1e10);\n")
                f.write("    }\n")
                f.write("    {\n")
                f.write("        name    rotatingZone;\n")
                f.write("        type    cellZoneSet;\n")
                f.write("        action  new;\n")
                f.write("        source  setToCellZone;\n")
                f.write("        set     rotatingZone;\n")
                f.write("    }\n")
                f.write(");\n")

        # Write Allrun script
        # Security: validate solver name against whitelist, quote paths
        import shlex
        allowed_solvers = {"simpleFoam", "pimpleFoam", "rhoSimpleFoam", "rhoPimpleFoam", "sonicFoam"}
        solver = of_solver
        if solver not in allowed_solvers:
            solver = "simpleFoam"

        n_procs = max(1, min(int(cfg.n_procs), 1024))  # Bound to sane range

        allrun = self._case_dir / "Allrun"
        with open(allrun, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            f.write('cd "${0%/*}" 2>/dev/null || true\n\n')

            # Mesh import strategy: try multiple methods
            if self._mesh_path:
                safe_path = shlex.quote(self._mesh_path)
                if self._mesh_path.endswith(".cgns"):
                    # Try cgnsToFoam first, fall back to blockMesh if not available
                    f.write("# Mesh import: CGNS with fallback to blockMeshDict\n")
                    f.write(f"if command -v cgnsToFoam &>/dev/null; then\n")
                    f.write(f"    echo 'Importing CGNS mesh...'\n")
                    f.write(f"    cgnsToFoam {safe_path}\n")
                    f.write(f"elif [ -f system/blockMeshDict ]; then\n")
                    f.write(f"    echo 'cgnsToFoam not found, using blockMeshDict...'\n")
                    f.write(f"    blockMesh\n")
                    f.write(f"else\n")
                    f.write(f"    echo 'ERROR: Neither cgnsToFoam nor blockMeshDict found.'\n")
                    f.write(f"    echo 'Install cgnsToFoam or export mesh as blockMeshDict:'\n")
                    f.write(f"    echo '  mesh.export_openfoam(\"system/blockMeshDict\")'\n")
                    f.write(f"    exit 1\n")
                    f.write(f"fi\n\n")
                elif self._mesh_path.endswith(".msh"):
                    # Fluent mesh
                    f.write(f"echo 'Importing Fluent mesh...'\n")
                    f.write(f"fluentMeshToFoam {safe_path}\n\n")
                else:
                    f.write("blockMesh\n\n")
            else:
                f.write("blockMesh\n\n")

            f.write("checkMesh\n\n")

            # Create cellZone for MRF if rotating
            if cfg.is_rotating:
                f.write("# Create rotatingZone cellZone for MRF\n")
                f.write("topoSet\n\n")

            if n_procs > 1:
                f.write("decomposePar -force\n")
                f.write(f"mpirun -np {n_procs} {solver} -parallel\n")
                f.write("reconstructPar\n")
            else:
                f.write(f"{solver}\n")
        allrun.chmod(0o755)

        # If mesh is CGNS, also generate blockMeshDict as fallback
        # so the case works even without cgnsToFoam
        if self._mesh_path and self._mesh_path.endswith(".cgns"):
            bmd_note = self._case_dir / "system" / "README_mesh.txt"
            bmd_note.parent.mkdir(parents=True, exist_ok=True)
            with open(bmd_note, "w") as f:
                f.write("# Mesh import note\n")
                f.write("# If cgnsToFoam is not available, generate blockMeshDict:\n")
                f.write("#   from astraturbo.mesh.multiblock import ...\n")
                f.write("#   mesh.export_openfoam('system/blockMeshDict')\n")
                f.write("# Then re-run: bash Allrun\n")

        return self._case_dir

    def _run_openfoam(self) -> CFDWorkflowResult:
        allrun = self._case_dir / "Allrun"
        log = self._case_dir / "log.Allrun"

        if not allrun.exists():
            return CFDWorkflowResult(
                success=False, case_dir=str(self._case_dir),
                error_message="Allrun script not found. Run setup_case() first.",
            )

        try:
            with open(log, "w") as f:
                proc = subprocess.run(
                    ["bash", str(allrun)], stdout=f, stderr=subprocess.STDOUT,
                    cwd=str(self._case_dir), timeout=7200,
                )
            return CFDWorkflowResult(
                success=(proc.returncode == 0),
                case_dir=str(self._case_dir),
                log_file=str(log),
            )
        except FileNotFoundError:
            return CFDWorkflowResult(
                success=False, case_dir=str(self._case_dir),
                error_message="OpenFOAM not found in PATH. Install OpenFOAM first.",
            )
        except subprocess.TimeoutExpired:
            return CFDWorkflowResult(
                success=False, case_dir=str(self._case_dir),
                log_file=str(log), error_message="Solver timed out.",
            )

    def _postprocess_openfoam(self, result: CFDWorkflowResult) -> CFDWorkflowResult:
        from ..cfd.postprocess import read_openfoam_residuals
        if result.log_file:
            result.residuals = read_openfoam_residuals(result.log_file)
        return result

    # ── Fluent ────────────────────────────────────────────

    def _setup_fluent(self) -> Path:
        """Generate ANSYS Fluent journal file."""
        self._case_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        journal = self._case_dir / "run.jou"

        with open(journal, "w") as f:
            f.write("; AstraTurbo Fluent Journal\n")
            f.write(f"; Generated for turbomachinery simulation\n\n")

            # Read mesh
            if self._mesh_path:
                if self._mesh_path.endswith(".cgns"):
                    f.write(f'/file/read-case "{self._mesh_path}"\n')
                elif self._mesh_path.endswith(".msh"):
                    f.write(f'/file/read-case "{self._mesh_path}"\n')

            # Solver settings
            f.write("/define/models/viscous/kw-sst? yes\n")

            # Boundary conditions
            f.write(f'/define/boundary-conditions/velocity-inlet inlet no no yes yes no {cfg.inlet_velocity} no 0 no {cfg.inlet_temperature} no no yes {cfg.turbulent_intensity} {cfg.turbulent_length_scale}\n')
            f.write(f'/define/boundary-conditions/pressure-outlet outlet yes no {cfg.outlet_pressure} no {cfg.inlet_temperature} no yes no no yes {cfg.turbulent_intensity} {cfg.turbulent_length_scale}\n')

            # Rotating frame
            if cfg.is_rotating:
                f.write(f"/define/boundary-conditions/fluid fluid yes no no yes {cfg.omega} 0 0 1 0 0 0\n")

            # Solution methods
            f.write("/solve/set/p-v-coupling 24\n")  # Coupled scheme
            f.write("/solve/set/discretization-scheme/pressure 14\n")  # Second order
            f.write("/solve/set/discretization-scheme/mom 1\n")
            f.write("/solve/set/discretization-scheme/k 1\n")
            f.write("/solve/set/discretization-scheme/omega 1\n")

            # Convergence monitors
            f.write(f"/solve/monitors/residual/convergence-criteria {cfg.convergence_target} {cfg.convergence_target} {cfg.convergence_target} {cfg.convergence_target} {cfg.convergence_target} {cfg.convergence_target}\n")

            # Iterate
            f.write(f"/solve/iterate {cfg.n_iterations}\n")

            # Save
            f.write(f'/file/write-case-data "{self._case_dir / "result"}"\n')
            f.write("/exit yes\n")

        # Write run script
        run_script = self._case_dir / "run_fluent.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"fluent 3ddp -t{cfg.n_procs} -i run.jou > fluent.log 2>&1\n")
        run_script.chmod(0o755)

        return self._case_dir

    # ── CFX ───────────────────────────────────────────────

    def _setup_cfx(self) -> Path:
        """Generate ANSYS CFX .def file (CCL format)."""
        self._case_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        ccl_file = self._case_dir / "setup.ccl"

        with open(ccl_file, "w") as f:
            f.write("# AstraTurbo CFX Setup (CCL)\n\n")

            f.write("FLOW: Flow Analysis 1\n")
            f.write("  SOLUTION UNITS:\n")
            f.write("    Length Units = m\n")
            f.write("    Mass Units = kg\n")
            f.write("    Pressure Units = Pa\n")
            f.write("    Temperature Units = K\n")
            f.write("  END\n\n")

            f.write("  DOMAIN: Passage\n")
            if cfg.is_rotating:
                f.write("    Domain Motion = Rotating\n")
                f.write(f"    Angular Velocity = {cfg.omega} [radian s^-1]\n")
                f.write(f"    Axis Definition = Coordinate Axis\n")
                f.write(f"    Rotation Axis = Global Z\n")
            else:
                f.write("    Domain Motion = Stationary\n")
            f.write("    Fluid Definition = Air at 25 C\n")

            # Turbulence
            f.write(f"    TURBULENCE MODEL:\n")
            if "sst" in cfg.turbulence_model.lower():
                f.write("      Option = SST\n")
            else:
                f.write("      Option = k epsilon\n")
            f.write("    END\n\n")

            # Boundaries
            f.write("    BOUNDARY: Inlet\n")
            f.write("      Boundary Type = INLET\n")
            f.write(f"      Flow Direction = Normal to Boundary Condition\n")
            f.write(f"      Mass and Momentum = Normal Speed\n")
            f.write(f"      Normal Speed = {cfg.inlet_velocity} [m s^-1]\n")
            f.write(f"      Static Temperature = {cfg.inlet_temperature} [K]\n")
            f.write("    END\n\n")

            f.write("    BOUNDARY: Outlet\n")
            f.write("      Boundary Type = OUTLET\n")
            f.write(f"      Mass and Momentum = Average Static Pressure\n")
            f.write(f"      Relative Pressure = {cfg.outlet_pressure} [Pa]\n")
            f.write("    END\n\n")

            f.write("    BOUNDARY: Blade\n")
            f.write("      Boundary Type = WALL\n")
            f.write("      Wall Influence On Flow = No Slip\n")
            f.write("    END\n\n")

            f.write("    BOUNDARY: Hub\n")
            f.write("      Boundary Type = WALL\n")
            if cfg.is_rotating:
                f.write("      Wall Influence On Flow = Counter Rotating Wall\n")
            f.write("    END\n\n")

            f.write("  END  # Domain\n\n")

            # Solver control
            f.write("  SOLVER CONTROL:\n")
            f.write(f"    Maximum Number of Iterations = {cfg.n_iterations}\n")
            f.write(f"    Residual Target = {cfg.convergence_target}\n")
            f.write("    Advection Scheme = High Resolution\n")
            f.write("    Turbulence Numerics = High Resolution\n")
            f.write("  END\n\n")

            f.write("END  # Flow\n")

        # Write run script
        run_script = self._case_dir / "run_cfx.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            if self._mesh_path:
                f.write(f"cfx5pre -batch setup.ccl -mesh {self._mesh_path}\n")
            f.write(f"cfx5solve -def setup.def -par-local {cfg.n_procs}\n")
        run_script.chmod(0o755)

        return self._case_dir

    # ── SU2 ───────────────────────────────────────────────

    def _setup_su2(self) -> Path:
        from ..cfd.su2 import write_su2_config
        self._case_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config

        mesh_file = self._mesh_path or "mesh.su2"
        write_su2_config(
            self._case_dir / "astraturbo.cfg",
            mesh_file=str(mesh_file),
            mach_number=cfg.inlet_velocity / 343.0,  # Approximate
            n_iterations=cfg.n_iterations,
        )

        # Run script
        run_script = self._case_dir / "run_su2.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            if cfg.n_procs > 1:
                f.write(f"mpirun -np {cfg.n_procs} SU2_CFD astraturbo.cfg > su2.log 2>&1\n")
            else:
                f.write("SU2_CFD astraturbo.cfg > su2.log 2>&1\n")
        run_script.chmod(0o755)

        return self._case_dir

    def _run_su2(self) -> CFDWorkflowResult:
        from ..cfd.runner import run_su2 as _run_su2
        cfg_file = self._case_dir / "astraturbo.cfg"
        result = _run_su2(cfg_file, self.config.n_procs)
        return CFDWorkflowResult(
            success=result.success,
            case_dir=str(self._case_dir),
            log_file=result.log_file,
            error_message=result.error_message,
        )
