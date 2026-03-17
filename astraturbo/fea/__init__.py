"""FEA (Finite Element Analysis) integration for AstraTurbo.

Provides structural analysis workflow for turbomachinery blades:
  - material: Material property database (Inconel, Ti-6Al-4V, etc.)
  - calculix: CalculiX/Abaqus input file generation
  - mesh_export: Blade surface to solid mesh, CFD pressure mapping
  - workflow: Coupled CFD-FEA pipeline orchestration
"""

from .material import Material, get_material, list_materials, MATERIAL_DATABASE
from .calculix import write_calculix_input
from .mesh_export import (
    blade_surface_to_solid_mesh,
    map_cfd_pressure_to_fea,
    identify_root_nodes,
    export_fea_mesh_abaqus,
)
from .workflow import FEAWorkflow, FEAWorkflowConfig, FEAResult

__all__ = [
    "Material",
    "get_material",
    "list_materials",
    "MATERIAL_DATABASE",
    "write_calculix_input",
    "blade_surface_to_solid_mesh",
    "map_cfd_pressure_to_fea",
    "identify_root_nodes",
    "export_fea_mesh_abaqus",
    "FEAWorkflow",
    "FEAWorkflowConfig",
    "FEAResult",
]
