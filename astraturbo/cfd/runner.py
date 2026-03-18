"""CFD solver runner — manages subprocess execution of CFD solvers."""

from __future__ import annotations

import subprocess
import shutil
import platform
from pathlib import Path
from dataclasses import dataclass, field


def _solver_install_hint(solver_name: str) -> str:
    """Return platform-specific install instructions for a solver."""
    system = platform.system()
    hints = {
        "simpleFoam": {
            "Darwin": (
                "Install OpenFOAM on macOS:\n"
                "  brew install open-mpi\n"
                "  Download from https://openfoam.org/download/\n"
                "  Or use Docker: docker pull openfoam/openfoam2312-default"
            ),
            "Linux": (
                "Install OpenFOAM on Linux:\n"
                "  Ubuntu/Debian: sudo apt install openfoam\n"
                "  Or: wget -q -O - https://dl.openfoam.org/gpg.key | sudo apt-key add -\n"
                "  See: https://openfoam.org/download/"
            ),
            "Windows": (
                "Install OpenFOAM on Windows:\n"
                "  Use WSL2: wsl --install, then apt install openfoam\n"
                "  Or Docker: docker pull openfoam/openfoam2312-default\n"
                "  See: https://openfoam.org/download/windows/"
            ),
        },
        "rhoSimpleFoam": None,  # same as simpleFoam
        "SU2_CFD": {
            "Darwin": "Install SU2: brew install su2 or https://su2code.github.io/download.html",
            "Linux": "Install SU2: pip install SU2 or https://su2code.github.io/download.html",
            "Windows": "Install SU2: https://su2code.github.io/download.html",
        },
        "ccx": {
            "Darwin": "Install CalculiX: brew install calculix",
            "Linux": "Install CalculiX: sudo apt install calculix-ccx",
            "Windows": "Install CalculiX: http://www.calculix.de/",
        },
    }
    solver_hints = hints.get(solver_name, hints.get("simpleFoam"))
    if solver_hints is None:
        solver_hints = hints["simpleFoam"]
    if isinstance(solver_hints, dict):
        return solver_hints.get(system, solver_hints.get("Linux", ""))
    return solver_hints


@dataclass
class RunConfig:
    """Configuration for a CFD run."""

    solver: str = "simpleFoam"
    case_dir: str = "."
    n_procs: int = 1
    log_file: str = "solver.log"


@dataclass
class RunResult:
    """Result of a CFD run."""

    success: bool = False
    return_code: int = -1
    log_file: str = ""
    error_message: str = ""


def run_openfoam(config: RunConfig) -> RunResult:
    """Run an OpenFOAM solver as a subprocess.

    Args:
        config: Run configuration.

    Returns:
        RunResult with status and log path.
    """
    case_dir = Path(config.case_dir)
    log_path = case_dir / config.log_file

    # Check solver exists
    solver_path = shutil.which(config.solver)
    if solver_path is None:
        hint = _solver_install_hint(config.solver)
        return RunResult(
            success=False, return_code=-1,
            error_message=(
                f"Solver '{config.solver}' not found in PATH.\n\n{hint}"
            ),
        )

    cmd = [config.solver, "-case", str(case_dir)]

    if config.n_procs > 1:
        mpi_path = shutil.which("mpirun")
        if mpi_path is None:
            return RunResult(
                success=False, return_code=-1,
                error_message="mpirun not found for parallel execution",
            )
        cmd = ["mpirun", "-np", str(config.n_procs)] + cmd

    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=str(case_dir), timeout=3600,
            )
        return RunResult(
            success=(result.returncode == 0),
            return_code=result.returncode,
            log_file=str(log_path),
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False, return_code=-1,
            log_file=str(log_path),
            error_message="Solver timed out after 3600s",
        )
    except Exception as e:
        return RunResult(
            success=False, return_code=-1,
            error_message=str(e),
        )


def run_su2(config_file: str | Path, n_procs: int = 1) -> RunResult:
    """Run the SU2_CFD solver.

    Args:
        config_file: Path to the .cfg file.
        n_procs: Number of MPI processes.

    Returns:
        RunResult with status.
    """
    config_file = Path(config_file)
    case_dir = config_file.parent
    log_path = case_dir / "su2.log"

    solver_path = shutil.which("SU2_CFD")
    if solver_path is None:
        hint = _solver_install_hint("SU2_CFD")
        return RunResult(
            success=False, return_code=-1,
            error_message=(
                f"SU2_CFD not found in PATH.\n\n{hint}"
            ),
        )

    cmd = ["SU2_CFD", str(config_file)]
    if n_procs > 1:
        cmd = ["mpirun", "-np", str(n_procs)] + cmd

    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=str(case_dir), timeout=7200,
            )
        return RunResult(
            success=(result.returncode == 0),
            return_code=result.returncode,
            log_file=str(log_path),
        )
    except Exception as e:
        return RunResult(
            success=False, return_code=-1,
            error_message=str(e),
        )
