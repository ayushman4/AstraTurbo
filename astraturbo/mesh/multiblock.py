"""Multi-block structured mesh generator.

Native Python replacement for GridZ scripting. Generates structured
multi-block CGNS meshes from polyline-defined block boundaries.

This fills the adaptation requirement:
  "Use this data, script it using GridZ to generate a multi-block
   structured CGNS mesh"

Instead of scripting GridZ externally, this module performs the same
function natively: takes polyline boundaries for each block, applies
grading, generates structured interior points via TFI, and exports
to CGNS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .polyline import Polyline, BlockEdge
from .grading import (
    compute_graded_parameters,
    project_grading_onto_polyline,
    compute_boundary_layer_grading,
)
from .transfinite import tfi_2d_vectorized
from .vertex_extraction import BlockTopology, BlockVertex


@dataclass
class StructuredBlock:
    """A single block of a multi-block structured mesh.

    Contains the computed grid points and metadata.
    """

    name: str
    points: NDArray[np.float64]  # (Ni, Nj, D) grid coordinates
    n_cells_i: int = 0
    n_cells_j: int = 0
    grading_i: float = 1.0
    grading_j: float = 1.0

    # Boundary patches on this block
    patches: dict[str, str] = field(default_factory=dict)
    # e.g. {"bottom": "blade", "top": "passage", "left": "inlet", "right": "outlet"}


@dataclass
class MultiBlockMesh:
    """Complete multi-block structured mesh.

    Contains all blocks and their connectivity, ready for CGNS export.
    """

    blocks: list[StructuredBlock] = field(default_factory=list)
    name: str = "AstraTurbo_Mesh"

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def total_cells(self) -> int:
        total = 0
        for b in self.blocks:
            ni, nj = b.points.shape[0] - 1, b.points.shape[1] - 1
            total += ni * nj
        return total

    @property
    def total_points(self) -> int:
        return sum(b.points.shape[0] * b.points.shape[1] for b in self.blocks)

    def get_block(self, name: str) -> StructuredBlock | None:
        for b in self.blocks:
            if b.name == name:
                return b
        return None

    def export_cgns(self, filepath: str | Path) -> None:
        """Export the multi-block mesh to CGNS format."""
        from ..export.cgns_writer import write_cgns_structured

        block_arrays = [b.points for b in self.blocks]
        block_names = [b.name for b in self.blocks]
        write_cgns_structured(filepath, block_arrays, block_names, self.name)

    def export_openfoam(self, filepath: str | Path) -> None:
        """Export to OpenFOAM blockMeshDict format."""
        from ..export.openfoam_writer import write_blockmeshdict

        all_vertices = []
        all_blocks_def = []
        all_patches = []
        vertex_offset = 0

        for block in self.blocks:
            ni, nj = block.points.shape[0], block.points.shape[1]
            dim = block.points.shape[2]

            # Corner vertices (for 2D blocks, extend to 3D)
            corners_2d = [
                block.points[0, 0],      # 0: bottom-left
                block.points[-1, 0],     # 1: bottom-right
                block.points[-1, -1],    # 2: top-right
                block.points[0, -1],     # 3: top-left
            ]

            for c in corners_2d:
                if dim == 2:
                    all_vertices.append(np.array([c[0], c[1], 0.0]))
                    all_vertices.append(np.array([c[0], c[1], 1.0]))
                else:
                    all_vertices.append(c)

            if dim == 2:
                # 8 vertices for hex block from 4 corner 2D points
                v = vertex_offset
                all_blocks_def.append({
                    "vertices": [v, v+2, v+4, v+6, v+1, v+3, v+5, v+7],
                    "cells": [ni - 1, nj - 1, 1],
                    "grading": [block.grading_i, block.grading_j, 1],
                })
                vertex_offset += 8
            else:
                v = vertex_offset
                all_blocks_def.append({
                    "vertices": list(range(v, v + len(corners_2d))),
                    "cells": [ni - 1, nj - 1, 1],
                    "grading": [block.grading_i, block.grading_j, 1],
                })
                vertex_offset += len(corners_2d)

        vertices_array = np.array(all_vertices, dtype=np.float64)
        write_blockmeshdict(filepath, vertices_array, all_blocks_def, all_patches)


class MultiBlockGenerator:
    """Generates multi-block structured meshes from polyline boundaries.

    This is the native Python replacement for GridZ. The workflow:
    1. Define blocks by their 4 boundary polylines (bottom, top, left, right)
    2. Apply grading to each edge
    3. Generate interior points via transfinite interpolation
    4. Assemble into a MultiBlockMesh
    5. Export to CGNS or OpenFOAM

    Usage::

        gen = MultiBlockGenerator()
        gen.add_block("inlet", bottom, top, left, right,
                      n_i=20, n_j=15, grading_i=0.5)
        gen.add_block("blade", bottom2, top2, left2, right2,
                      n_i=40, n_j=15, grading_j=1.3)
        mesh = gen.generate()
        mesh.export_cgns("output.cgns")
    """

    def __init__(self) -> None:
        self._block_defs: list[dict] = []

    def add_block(
        self,
        name: str,
        bottom: Polyline | NDArray[np.float64],
        top: Polyline | NDArray[np.float64],
        left: Polyline | NDArray[np.float64],
        right: Polyline | NDArray[np.float64],
        n_i: int = 20,
        n_j: int = 20,
        grading_i: float = 1.0,
        grading_j: float = 1.0,
        patches: dict[str, str] | None = None,
    ) -> None:
        """Add a block definition.

        Args:
            name: Block name.
            bottom: Bottom boundary (i-direction, j=0).
            top: Top boundary (i-direction, j=max).
            left: Left boundary (j-direction, i=0).
            right: Right boundary (j-direction, i=max).
            n_i: Number of cells in i-direction.
            n_j: Number of cells in j-direction.
            grading_i: Grading ratio in i-direction.
            grading_j: Grading ratio in j-direction.
            patches: Dict mapping face names to boundary types.
        """
        def _to_polyline(data):
            if isinstance(data, Polyline):
                return data
            return Polyline(np.asarray(data, dtype=np.float64))

        self._block_defs.append({
            "name": name,
            "bottom": _to_polyline(bottom),
            "top": _to_polyline(top),
            "left": _to_polyline(left),
            "right": _to_polyline(right),
            "n_i": n_i,
            "n_j": n_j,
            "grading_i": grading_i,
            "grading_j": grading_j,
            "patches": patches or {},
        })

    def add_block_from_topology(
        self,
        name: str,
        topology: BlockTopology,
        edge_bottom: int,
        edge_top: int,
        edge_left: int,
        edge_right: int,
        n_i: int = 20,
        n_j: int = 20,
        grading_i: float = 1.0,
        grading_j: float = 1.0,
    ) -> None:
        """Add a block using edge indices from a BlockTopology."""
        edges = topology.edges
        self.add_block(
            name=name,
            bottom=edges[edge_bottom].to_polyline(),
            top=edges[edge_top].to_polyline(),
            left=edges[edge_left].to_polyline(),
            right=edges[edge_right].to_polyline(),
            n_i=n_i, n_j=n_j,
            grading_i=grading_i, grading_j=grading_j,
        )

    def generate(self) -> MultiBlockMesh:
        """Generate the multi-block structured mesh.

        For each block:
        1. Resample boundary polylines to required point counts with grading
        2. Run transfinite interpolation
        3. Assemble into StructuredBlock

        Returns:
            MultiBlockMesh ready for export.
        """
        mesh = MultiBlockMesh()

        for bdef in self._block_defs:
            ni = bdef["n_i"] + 1  # Points = cells + 1
            nj = bdef["n_j"] + 1

            # Resample boundaries with grading
            t_i = compute_graded_parameters(ni, bdef["grading_i"])
            t_j = compute_graded_parameters(nj, bdef["grading_j"])

            bottom = self._resample_at(bdef["bottom"], t_i)
            top = self._resample_at(bdef["top"], t_i)
            left = self._resample_at(bdef["left"], t_j)
            right = self._resample_at(bdef["right"], t_j)

            # TFI
            points = tfi_2d_vectorized(bottom, top, left, right)

            block = StructuredBlock(
                name=bdef["name"],
                points=points,
                n_cells_i=bdef["n_i"],
                n_cells_j=bdef["n_j"],
                grading_i=bdef["grading_i"],
                grading_j=bdef["grading_j"],
                patches=bdef["patches"],
            )
            mesh.blocks.append(block)

        return mesh

    @staticmethod
    def _resample_at(
        polyline: Polyline, t_params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Resample a polyline at given normalized parameters."""
        params = polyline.normalized_parameters()
        n = len(t_params)
        dim = polyline.dim
        new_pts = np.empty((n, dim), dtype=np.float64)
        for d in range(dim):
            new_pts[:, d] = np.interp(t_params, params, polyline.points[:, d])
        return new_pts


def generate_blade_passage_mesh(
    profile: NDArray[np.float64],
    pitch: float,
    n_blade: int = 40,
    n_ogrid: int = 10,
    n_inlet: int = 15,
    n_outlet: int = 15,
    n_passage: int = 20,
    ogrid_thickness: float = 0.01,
    grading_ogrid: float = 1.3,
    grading_inlet: float = 0.5,
    grading_outlet: float = 2.0,
    inlet_offset: float | None = None,
    outlet_offset: float | None = None,
) -> MultiBlockMesh:
    """High-level function to generate a complete blade passage mesh.

    Creates a multi-block structured mesh suitable for CGNS export,
    combining O-grid around the blade with H-grid blocks for the
    inlet, outlet, and passage.

    Args:
        profile: (N, 2) closed blade profile.
        pitch: Blade-to-blade pitch.
        n_blade: Cells around the blade (O-grid circumferential).
        n_ogrid: Cells in O-grid wall-normal direction.
        n_inlet: Cells in inlet block.
        n_outlet: Cells in outlet block.
        n_passage: Cells in passage pitchwise direction.
        ogrid_thickness: O-grid layer thickness.
        grading_ogrid: O-grid wall-normal grading.
        grading_inlet: Inlet block grading.
        grading_outlet: Outlet block grading.
        inlet_offset: Distance upstream of LE.
        outlet_offset: Distance downstream of TE.

    Returns:
        MultiBlockMesh ready for export to CGNS or OpenFOAM.
    """
    from .vertex_extraction import build_block_topology_from_profile

    le_idx = int(np.argmin(profile[:, 0]))
    x_min, x_max = profile[:, 0].min(), profile[:, 0].max()
    chord = x_max - x_min
    y_mid = profile[:, 1].mean()

    if inlet_offset is None:
        inlet_offset = 0.5 * chord
    if outlet_offset is None:
        outlet_offset = 0.5 * chord

    x_inlet = x_min - inlet_offset
    x_outlet = x_max + outlet_offset
    y_lower = y_mid - pitch / 2
    y_upper = y_mid + pitch / 2

    # Split profile
    suction = profile[:le_idx + 1]
    pressure = profile[le_idx:]

    # Offset curves for O-grid outer boundary
    def _offset_curve(pts, thickness):
        n = len(pts)
        normals = np.zeros_like(pts)
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            tangent = pts[ip] - pts[im]
            normals[i] = np.array([-tangent[1], tangent[0]])
            nl = np.linalg.norm(normals[i])
            if nl > 1e-15:
                normals[i] /= nl
        return pts + normals * thickness

    suction_outer = _offset_curve(suction, ogrid_thickness)
    pressure_outer = _offset_curve(pressure, ogrid_thickness)

    gen = MultiBlockGenerator()

    # Block 1: O-grid suction side
    n_ss = len(suction)
    ss_left = np.column_stack((
        np.linspace(suction[0, 0], suction_outer[0, 0], n_ogrid + 1),
        np.linspace(suction[0, 1], suction_outer[0, 1], n_ogrid + 1),
    ))
    ss_right = np.column_stack((
        np.linspace(suction[-1, 0], suction_outer[-1, 0], n_ogrid + 1),
        np.linspace(suction[-1, 1], suction_outer[-1, 1], n_ogrid + 1),
    ))
    gen.add_block(
        "ogrid_suction",
        Polyline(suction), Polyline(suction_outer),
        Polyline(ss_left), Polyline(ss_right),
        n_i=n_blade // 2, n_j=n_ogrid,
        grading_j=grading_ogrid,
        patches={"bottom": "blade"},
    )

    # Block 2: O-grid pressure side
    ps_left = np.column_stack((
        np.linspace(pressure[0, 0], pressure_outer[0, 0], n_ogrid + 1),
        np.linspace(pressure[0, 1], pressure_outer[0, 1], n_ogrid + 1),
    ))
    ps_right = np.column_stack((
        np.linspace(pressure[-1, 0], pressure_outer[-1, 0], n_ogrid + 1),
        np.linspace(pressure[-1, 1], pressure_outer[-1, 1], n_ogrid + 1),
    ))
    gen.add_block(
        "ogrid_pressure",
        Polyline(pressure), Polyline(pressure_outer),
        Polyline(ps_left), Polyline(ps_right),
        n_i=n_blade // 2, n_j=n_ogrid,
        grading_j=grading_ogrid,
        patches={"bottom": "blade"},
    )

    # Block 3: Inlet (upper half)
    inlet_bl = np.array([[x_inlet, y_mid], [x_min, y_mid]])
    inlet_tl = np.array([[x_inlet, y_upper], [x_min, y_upper]])
    inlet_left = np.array([[x_inlet, y_mid], [x_inlet, y_upper]])
    inlet_right = np.array([[x_min, y_mid], [x_min, y_upper]])
    gen.add_block(
        "inlet_upper", Polyline(inlet_bl), Polyline(inlet_tl),
        Polyline(inlet_left), Polyline(inlet_right),
        n_i=n_inlet, n_j=n_passage // 2,
        grading_i=grading_inlet,
        patches={"left": "inlet", "top": "periodic_upper"},
    )

    # Block 4: Inlet (lower half)
    inlet_bl2 = np.array([[x_inlet, y_lower], [x_min, y_lower]])
    inlet_tl2 = np.array([[x_inlet, y_mid], [x_min, y_mid]])
    inlet_left2 = np.array([[x_inlet, y_lower], [x_inlet, y_mid]])
    inlet_right2 = np.array([[x_min, y_lower], [x_min, y_mid]])
    gen.add_block(
        "inlet_lower", Polyline(inlet_bl2), Polyline(inlet_tl2),
        Polyline(inlet_left2), Polyline(inlet_right2),
        n_i=n_inlet, n_j=n_passage // 2,
        grading_i=grading_inlet,
        patches={"left": "inlet", "bottom": "periodic_lower"},
    )

    # Block 5: Outlet (upper half)
    outlet_bl = np.array([[x_max, y_mid], [x_outlet, y_mid]])
    outlet_tl = np.array([[x_max, y_upper], [x_outlet, y_upper]])
    outlet_left = np.array([[x_max, y_mid], [x_max, y_upper]])
    outlet_right = np.array([[x_outlet, y_mid], [x_outlet, y_upper]])
    gen.add_block(
        "outlet_upper", Polyline(outlet_bl), Polyline(outlet_tl),
        Polyline(outlet_left), Polyline(outlet_right),
        n_i=n_outlet, n_j=n_passage // 2,
        grading_i=grading_outlet,
        patches={"right": "outlet", "top": "periodic_upper"},
    )

    # Block 6: Outlet (lower half)
    outlet_bl2 = np.array([[x_max, y_lower], [x_outlet, y_lower]])
    outlet_tl2 = np.array([[x_max, y_mid], [x_outlet, y_mid]])
    outlet_left2 = np.array([[x_max, y_lower], [x_max, y_mid]])
    outlet_right2 = np.array([[x_outlet, y_lower], [x_outlet, y_mid]])
    gen.add_block(
        "outlet_lower", Polyline(outlet_bl2), Polyline(outlet_tl2),
        Polyline(outlet_left2), Polyline(outlet_right2),
        n_i=n_outlet, n_j=n_passage // 2,
        grading_i=grading_outlet,
        patches={"right": "outlet", "bottom": "periodic_lower"},
    )

    mesh = gen.generate()

    # Run quality checks and log warnings
    _logger = logging.getLogger(__name__)
    from .quality import mesh_quality_report
    for block in mesh.blocks:
        report = mesh_quality_report(block.points)
        if report.get("aspect_ratio_max", 0) > 100:
            _logger.warning(
                "Block '%s': max aspect ratio %.1f exceeds threshold of 100",
                block.name, report["aspect_ratio_max"],
            )
        if report.get("skewness_max", 0) > 0.95:
            _logger.warning(
                "Block '%s': max skewness %.3f exceeds threshold of 0.95",
                block.name, report["skewness_max"],
            )

    return mesh


def generate_blade_passage_mesh_3d(
    profiles: list[NDArray[np.float64]],
    span_positions: list[float],
    pitch: float,
    n_blade: int = 40,
    n_ogrid: int = 10,
    n_inlet: int = 15,
    n_outlet: int = 15,
    n_passage: int = 20,
    ogrid_thickness: float = 0.01,
    grading_ogrid: float = 1.3,
    grading_inlet: float = 0.5,
    grading_outlet: float = 2.0,
) -> MultiBlockMesh:
    """Generate a 3D blade passage mesh by stacking 2D meshes at span stations.

    Takes 2D profiles at different span stations, generates a 2D mesh at each
    station, then stacks them with interpolation in the span direction.

    Args:
        profiles: List of (N, 2) closed 2D blade profiles at different span stations.
        span_positions: List of span coordinates (z-values) for each profile.
        pitch: Blade-to-blade pitch.
        n_blade: Cells around blade.
        n_ogrid: O-grid cells.
        n_inlet: Inlet cells.
        n_outlet: Outlet cells.
        n_passage: Passage pitchwise cells.
        ogrid_thickness: O-grid layer thickness.
        grading_ogrid: O-grid grading.
        grading_inlet: Inlet grading.
        grading_outlet: Outlet grading.

    Returns:
        MultiBlockMesh with 3D blocks (Ni, Nj, Nk, 3).
    """
    logger = logging.getLogger(__name__)

    if len(profiles) != len(span_positions):
        raise ValueError("Number of profiles must match number of span positions")
    if len(profiles) < 2:
        raise ValueError("At least 2 span stations required for 3D mesh")

    # Generate 2D mesh at each span station
    meshes_2d = []
    for i, profile in enumerate(profiles):
        mesh_2d = generate_blade_passage_mesh(
            profile=profile,
            pitch=pitch,
            n_blade=n_blade,
            n_ogrid=n_ogrid,
            n_inlet=n_inlet,
            n_outlet=n_outlet,
            n_passage=n_passage,
            ogrid_thickness=ogrid_thickness,
            grading_ogrid=grading_ogrid,
            grading_inlet=grading_inlet,
            grading_outlet=grading_outlet,
        )
        meshes_2d.append(mesh_2d)

    # Stack 2D meshes into 3D blocks
    # All meshes should have the same block structure
    n_blocks = meshes_2d[0].n_blocks
    n_span = len(span_positions)

    mesh_3d = MultiBlockMesh(name="AstraTurbo_3D_Mesh")

    for block_idx in range(n_blocks):
        block_name = meshes_2d[0].blocks[block_idx].name
        # Get the 2D block from each span station
        blocks_at_stations = []
        for mesh in meshes_2d:
            if block_idx < len(mesh.blocks):
                blocks_at_stations.append(mesh.blocks[block_idx].points)

        if not blocks_at_stations:
            continue

        ni, nj = blocks_at_stations[0].shape[0], blocks_at_stations[0].shape[1]
        dim_2d = blocks_at_stations[0].shape[2] if blocks_at_stations[0].ndim == 3 else 2

        # Create 3D block: (Ni, Nj, Nk, 3)
        nk = n_span
        block_3d = np.zeros((ni, nj, nk, 3), dtype=np.float64)

        for k, (pts_2d, z_pos) in enumerate(zip(blocks_at_stations, span_positions)):
            block_3d[:, :, k, 0] = pts_2d[:, :, 0] if dim_2d >= 2 else pts_2d[:, :, 0]
            block_3d[:, :, k, 1] = pts_2d[:, :, 1] if dim_2d >= 2 else 0.0
            block_3d[:, :, k, 2] = z_pos

        # If we have fewer mesh stations than span positions, interpolate
        # (for now we require one mesh per span station)

        structured = StructuredBlock(
            name=block_name,
            points=block_3d,
            n_cells_i=ni - 1,
            n_cells_j=nj - 1,
            patches=meshes_2d[0].blocks[block_idx].patches,
        )
        mesh_3d.blocks.append(structured)

    logger.info(
        "Generated 3D mesh: %d blocks, %d total points, %d span stations",
        mesh_3d.n_blocks, mesh_3d.total_points, n_span,
    )
    return mesh_3d
