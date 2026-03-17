"""Export module for AstraTurbo.

Comprehensive format support for mesh, geometry, and simulation data:

Native readers/writers:
  - CGNS structured (read + write via h5py)
  - OpenFOAM polyMesh points (read) and blockMeshDict (write)
  - PLOT3D structured grid (read + write)
  - STL (read + write, no dependencies)
  - CSV profile data (read + write)

Via meshio (~40 formats):
  - VTK, VTU, PVTU, Gmsh, Nastran, UNV, SU2, XDMF,
    Exodus, Abaqus, Fluent, Medit, OBJ, PLY, and more

Via cadquery (optional):
  - STEP (read + write)
  - IGES (write)
"""

# CGNS
from .cgns_writer import write_cgns_structured, write_cgns_2d
from .cgns_reader import read_cgns, cgns_to_points, cgns_info, CGNSReadError

# OpenFOAM
from .openfoam_writer import write_blockmeshdict
from .openfoam_reader import (
    OpenFOAMReadError,
    validate_openfoam_file,
    read_openfoam_points,
    read_openfoam_boundary,
    read_openfoam_polymesh,
    openfoam_points_to_cloud,
)

# Multi-format
from .formats import (
    FormatError,
    detect_format,
    read_mesh,
    write_mesh,
    export_structured_as_quads,
    list_supported_formats,
)

# CAD
from .cad_export import (
    CADExportError,
    write_stl_ascii,
    write_stl_from_surface,
    read_stl,
)

__all__ = [
    # CGNS
    "write_cgns_structured", "write_cgns_2d",
    "read_cgns", "cgns_to_points", "cgns_info", "CGNSReadError",
    # OpenFOAM
    "write_blockmeshdict",
    "OpenFOAMReadError", "validate_openfoam_file",
    "read_openfoam_points", "read_openfoam_boundary",
    "read_openfoam_polymesh", "openfoam_points_to_cloud",
    # Multi-format
    "FormatError", "detect_format", "read_mesh", "write_mesh",
    "export_structured_as_quads", "list_supported_formats",
    # CAD
    "CADExportError", "write_stl_ascii", "write_stl_from_surface", "read_stl",
]
