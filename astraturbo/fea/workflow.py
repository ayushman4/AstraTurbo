"""Coupled CFD-FEA workflow for turbomachinery blades.

Orchestrates the fluid-structure interaction (FSI) pipeline:
  1. Generate blade geometry (AstraTurbo blade/)
  2. Generate CFD mesh and run CFD (AstraTurbo mesh/ + cfd/)
  3. Extract surface pressures from CFD
  4. Map pressures onto FEA mesh
  5. Run structural analysis (CalculiX)
  6. Extract deformations
  7. (Optional) Update geometry and iterate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess

import numpy as np
from numpy.typing import NDArray

from .material import Material, INCONEL_718
from .calculix import write_calculix_input
from .mesh_export import (
    blade_surface_to_solid_mesh,
    map_cfd_pressure_to_fea,
    identify_root_nodes,
)


@dataclass
class FEAWorkflowConfig:
    """Configuration for FEA structural analysis."""

    material: Material = field(default_factory=lambda: INCONEL_718)
    omega: float = 0.0                      # Angular velocity (rad/s)
    rotation_axis: tuple = (0.0, 0.0, 1.0)
    blade_thickness: float = 0.002          # m
    analysis_type: str = "static"           # 'static', 'frequency', 'buckle'
    solver: str = "calculix"                # 'calculix' or 'abaqus'


@dataclass
class FEAResult:
    """Result from FEA analysis."""

    success: bool = False
    case_dir: str = ""
    log_file: str = ""
    error_message: str = ""

    # Results (populated after post-processing)
    max_stress: float | None = None         # Pa (von Mises)
    max_displacement: float | None = None   # m
    natural_frequencies: list[float] | None = None  # Hz (for modal analysis)
    safety_factor: float | None = None


class FEAWorkflow:
    """Manages the FEA structural analysis pipeline.

    Usage::

        wf = FEAWorkflow(config)
        wf.set_blade_surface(surface_points, ni, nj)
        wf.set_cfd_pressure(cfd_points, cfd_pressure)  # Optional
        wf.setup("fea_case/")
        result = wf.run()
    """

    def __init__(self, config: FEAWorkflowConfig | None = None) -> None:
        self.config = config or FEAWorkflowConfig()
        self._surface_points: NDArray | None = None
        self._ni: int = 0
        self._nj: int = 0
        self._cfd_points: NDArray | None = None
        self._cfd_pressure: NDArray | None = None
        self._case_dir: Path | None = None

        # Generated mesh
        self._nodes: NDArray | None = None
        self._elements: NDArray | None = None

    def set_blade_surface(
        self, surface_points: NDArray[np.float64], ni: int, nj: int
    ) -> None:
        """Set the blade surface from CFD mesh or geometry.

        Args:
            surface_points: (ni*nj, 3) blade surface coordinates.
            ni: Points in streamwise direction.
            nj: Points in spanwise direction.
        """
        self._surface_points = surface_points
        self._ni = ni
        self._nj = nj

    def set_cfd_pressure(
        self,
        cfd_points: NDArray[np.float64],
        cfd_pressure: NDArray[np.float64],
    ) -> None:
        """Set CFD surface pressure for load mapping.

        Args:
            cfd_points: (N, 3) CFD surface node coordinates.
            cfd_pressure: (N,) pressure at each CFD node.
        """
        self._cfd_points = cfd_points
        self._cfd_pressure = cfd_pressure

    def setup(self, case_dir: str | Path) -> Path:
        """Generate the FEA input files.

        Creates:
          - Solid mesh from blade surface
          - Maps CFD pressure loads (if provided)
          - Writes CalculiX/Abaqus input file
          - Writes run script

        Returns:
            Path to the case directory.
        """
        self._case_dir = Path(case_dir)
        self._case_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config

        if self._surface_points is None:
            raise ValueError("Set blade surface first with set_blade_surface()")

        # Generate solid mesh
        self._nodes, self._elements = blade_surface_to_solid_mesh(
            self._surface_points, self._ni, self._nj, cfg.blade_thickness
        )

        # Find root nodes for BC
        root_nodes = identify_root_nodes(self._nodes)

        # Map CFD pressure if available
        pressure_loads = None
        if self._cfd_points is not None and self._cfd_pressure is not None:
            n_surface = self._ni * self._nj
            fea_surface = self._nodes[:n_surface]
            mapped_pressure = map_cfd_pressure_to_fea(
                self._cfd_points, self._cfd_pressure, fea_surface
            )
            pressure_loads = {"blade_surface": mapped_pressure}

        # Write input file
        inp_file = self._case_dir / "blade.inp"
        write_calculix_input(
            inp_file,
            nodes=self._nodes,
            elements=self._elements,
            material=cfg.material,
            omega=cfg.omega,
            rotation_axis=cfg.rotation_axis,
            pressure_loads=pressure_loads,
            fixed_nodes=root_nodes,
            element_type="C3D8",
            analysis_type=cfg.analysis_type,
        )

        # Write run script
        run_script = self._case_dir / "run_fea.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            if cfg.solver == "calculix":
                f.write("ccx blade > calculix.log 2>&1\n")
            elif cfg.solver == "abaqus":
                f.write("abaqus job=blade interactive > abaqus.log 2>&1\n")
        run_script.chmod(0o755)

        return self._case_dir

    def run(self) -> FEAResult:
        """Execute the FEA solver."""
        if self._case_dir is None:
            return FEAResult(
                success=False, error_message="Case not set up. Call setup() first."
            )

        run_script = self._case_dir / "run_fea.sh"
        log_file = self._case_dir / "fea.log"

        try:
            with open(log_file, "w") as f:
                proc = subprocess.run(
                    ["bash", str(run_script)],
                    stdout=f, stderr=subprocess.STDOUT,
                    cwd=str(self._case_dir),
                    timeout=3600,
                )
            return FEAResult(
                success=(proc.returncode == 0),
                case_dir=str(self._case_dir),
                log_file=str(log_file),
            )
        except FileNotFoundError:
            return FEAResult(
                success=False, case_dir=str(self._case_dir),
                error_message=(
                    f"FEA solver not found. Install "
                    f"{'CalculiX (ccx)' if self.config.solver == 'calculix' else 'Abaqus'}."
                ),
            )
        except subprocess.TimeoutExpired:
            return FEAResult(
                success=False, case_dir=str(self._case_dir),
                log_file=str(log_file),
                error_message="FEA solver timed out.",
            )

    def estimate_stress_analytical(self) -> dict:
        """Quick analytical stress estimate without running FEA.

        Uses simplified formulas for centrifugal stress in a tapered blade.
        Useful for sanity checking before running full FEA.

        Returns:
            Dict with estimated stresses and safety factor.
        """
        cfg = self.config
        mat = cfg.material

        if self._surface_points is None:
            return {"error": "No blade surface set"}

        # Blade dimensions from surface
        pts = self._surface_points
        span = pts[:, 2].max() - pts[:, 2].min()  # Approximate span
        r_hub = pts[:, 2].min()  # Approximate hub radius
        r_tip = r_hub + span

        # Centrifugal stress: sigma = rho * omega^2 * A_tip/A_root * span²/2
        # Simplified for uniform blade: sigma = rho * omega^2 * (r_tip² - r_hub²) / 2
        sigma_centrifugal = mat.density * cfg.omega**2 * (r_tip**2 - r_hub**2) / 2.0

        # Safety factor
        sf = mat.yield_strength / sigma_centrifugal if sigma_centrifugal > 0 else float("inf")

        return {
            "centrifugal_stress_MPa": sigma_centrifugal / 1e6,
            "yield_strength_MPa": mat.yield_strength / 1e6,
            "safety_factor": sf,
            "span_m": span,
            "r_hub_m": r_hub,
            "r_tip_m": r_tip,
            "omega_rad_s": cfg.omega,
            "material": mat.name,
            "acceptable": sf > 1.5,
        }
