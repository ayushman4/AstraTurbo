"""Mesh generation module for AstraTurbo.

Provides structured mesh generation for turbomachinery:
  - transfinite: 2D transfinite interpolation with grading
  - scm_mesher: Streamline Curvature Method (S2m plane)
  - ogrid: O10H topology O-grid around blades
  - quality: Mesh quality metrics (aspect ratio, skewness, y+)
  - polyline: Polyline and Arc geometry for block edges
  - grading: Edge grading projection onto curves
  - vertex_extraction: Block topology builder from profiles
  - multiblock: Multi-block structured mesh generator (GridZ replacement)
  - multistage: Multi-row rotor+stator mesh orchestration
  - s1_mesher: S1 blade-to-blade surface mesh
"""

from .transfinite import tfi_2d, tfi_2d_vectorized, tfi_2d_graded, apply_grading
from .scm_mesher import SCMMesher, SCMMeshConfig
from .s1_mesher import S1Mesher, S1MeshConfig
from .ogrid import OGridGenerator, OGridMesh, OGridMeshConfig
from .quality import (
    compute_aspect_ratio,
    compute_skewness,
    estimate_yplus,
    first_cell_height_for_yplus,
    auto_first_cell_height,
    mesh_quality_report,
)
from .tip_clearance import generate_tip_clearance_mesh
from .smoothing import (
    laplacian_smooth,
    laplacian_smooth_vectorized,
    orthogonality_correction,
    combined_smooth,
)
from .polyline import Polyline, Arc, BlockEdge
from .grading import (
    compute_graded_parameters,
    compute_double_sided_grading,
    compute_boundary_layer_grading,
    project_grading_onto_polyline,
    project_grading_onto_arc,
    project_boundary_layer_onto_polyline,
)
from .vertex_extraction import (
    BlockVertex,
    BlockFace,
    BlockTopology,
    extract_vertices_from_profile,
    extract_polylines_from_profile,
    build_passage_vertices,
    build_passage_edges,
    build_block_topology_from_profile,
    import_profile_points_from_xml,
)
from .multiblock import (
    StructuredBlock,
    MultiBlockMesh,
    MultiBlockGenerator,
    generate_blade_passage_mesh,
)
from .multistage import (
    RowMeshConfig,
    StageConfig,
    MultistageMesh,
    MultistageGenerator,
)

__all__ = [
    # Transfinite
    "tfi_2d", "tfi_2d_vectorized", "tfi_2d_graded", "apply_grading",
    # SCM
    "SCMMesher", "SCMMeshConfig",
    # O-Grid
    "OGridGenerator", "OGridMesh", "OGridMeshConfig",
    # Quality
    "compute_aspect_ratio", "compute_skewness",
    "estimate_yplus", "first_cell_height_for_yplus",
    "auto_first_cell_height", "mesh_quality_report",
    # Tip clearance
    "generate_tip_clearance_mesh",
    # Smoothing
    "laplacian_smooth", "laplacian_smooth_vectorized",
    "orthogonality_correction", "combined_smooth",
    # Polyline/Arc
    "Polyline", "Arc", "BlockEdge",
    # Grading
    "compute_graded_parameters", "compute_double_sided_grading",
    "compute_boundary_layer_grading",
    "project_grading_onto_polyline", "project_grading_onto_arc",
    "project_boundary_layer_onto_polyline",
    # Vertex extraction
    "BlockVertex", "BlockFace", "BlockTopology",
    "extract_vertices_from_profile", "extract_polylines_from_profile",
    "build_passage_vertices", "build_passage_edges",
    "build_block_topology_from_profile", "import_profile_points_from_xml",
    # Multi-block
    "StructuredBlock", "MultiBlockMesh", "MultiBlockGenerator",
    "generate_blade_passage_mesh",
    # Multi-stage
    "RowMeshConfig", "StageConfig", "MultistageMesh", "MultistageGenerator",
]
