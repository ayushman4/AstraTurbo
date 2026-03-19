"""CFD solver interface for AstraTurbo.

Provides case setup, execution, and post-processing for:
  - OpenFOAM (simpleFoam, pimpleFoam, with MRF for rotors)
  - ANSYS Fluent (journal file generation)
  - ANSYS CFX (CCL definition file generation)
  - SU2

Workflow orchestration via CFDWorkflow class handles the full pipeline:
  mesh → case setup → solver execution → post-processing
"""

from .openfoam import create_openfoam_case, write_simpleFoam_case
from .su2 import write_su2_config
from .runner import run_openfoam, run_su2, RunConfig, RunResult
from .postprocess import read_openfoam_residuals, compute_performance_map
from .workflow import CFDWorkflow, CFDWorkflowConfig, CFDWorkflowResult

__all__ = [
    "create_openfoam_case",
    "write_simpleFoam_case",
    "write_su2_config",
    "run_openfoam",
    "run_su2",
    "RunConfig",
    "RunResult",
    "read_openfoam_residuals",
    "compute_performance_map",
    "CFDWorkflow",
    "CFDWorkflowConfig",
    "CFDWorkflowResult",
]
