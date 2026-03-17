"""O-grid topology definition for turbomachinery blade passages.

Ported from V1 blockMeshDictGen.py topology section. Defines the O10H
blocking structure — a combination of O-grid around the blade with
H-grid blocks in the inlet, outlet, and passage.

The O10H topology creates 10 blocks per blade layer:
  - 4 O-grid blocks around the blade (suction, pressure, LE, TE)
  - 2 H-grid blocks in the inlet
  - 2 H-grid blocks in the outlet
  - 2 H-grid blocks in the passage (above/below blade)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class BlockDefinition:
    """Definition of a single hex block in the O-grid topology."""

    name: str
    # Vertex indices (8 corners for a hex block)
    vertices: list[int] = field(default_factory=list)
    # Cell counts in i, j, k directions
    n_cells: list[int] = field(default_factory=lambda: [10, 10, 1])
    # Grading in i, j, k
    grading: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class PatchDefinition:
    """Definition of a boundary patch."""

    name: str
    patch_type: str  # 'wall', 'inlet', 'outlet', 'cyclic', 'empty'
    faces: list[list[int]] = field(default_factory=list)


@dataclass
class O10HTopology:
    """Complete O10H topology for a single blade passage.

    This topology wraps an O-grid around the blade with H-grid
    extensions for inlet, outlet, and passage.
    """

    blocks: list[BlockDefinition] = field(default_factory=list)
    patches: list[PatchDefinition] = field(default_factory=list)
    vertices: NDArray[np.float64] | None = None

    # Block names for reference
    BLOCK_NAMES = [
        "o_suction",      # O-grid suction side
        "o_pressure",     # O-grid pressure side
        "o_leading",      # O-grid leading edge
        "o_trailing",     # O-grid trailing edge
        "h_inlet_upper",  # H-grid inlet upper
        "h_inlet_lower",  # H-grid inlet lower
        "h_outlet_upper", # H-grid outlet upper
        "h_outlet_lower", # H-grid outlet lower
        "h_pass_upper",   # H-grid passage upper
        "h_pass_lower",   # H-grid passage lower
    ]


def create_o10h_topology(
    n_ogrid_normal: int = 10,
    n_blade_wrap: int = 40,
    n_inlet: int = 15,
    n_outlet: int = 15,
    n_passage: int = 20,
    n_span: int = 1,
) -> O10HTopology:
    """Create an O10H topology with specified cell counts.

    Args:
        n_ogrid_normal: Cells in O-grid normal direction (wall-normal).
        n_blade_wrap: Cells wrapping around the blade in the O-grid.
        n_inlet: Cells in inlet H-block (streamwise).
        n_outlet: Cells in outlet H-block (streamwise).
        n_passage: Cells in passage H-block (pitchwise).
        n_span: Cells in spanwise direction (1 for 2D).

    Returns:
        O10HTopology with block definitions.
    """
    topo = O10HTopology()

    # O-grid blocks (around blade)
    for i, name in enumerate(O10HTopology.BLOCK_NAMES[:4]):
        topo.blocks.append(BlockDefinition(
            name=name,
            n_cells=[n_blade_wrap // 4, n_ogrid_normal, n_span],
            grading=[1.0, 1.2, 1.0],  # Cluster toward blade wall
        ))

    # H-grid blocks (inlet/outlet/passage)
    for name in O10HTopology.BLOCK_NAMES[4:6]:
        topo.blocks.append(BlockDefinition(
            name=name,
            n_cells=[n_inlet, n_passage // 2, n_span],
        ))

    for name in O10HTopology.BLOCK_NAMES[6:8]:
        topo.blocks.append(BlockDefinition(
            name=name,
            n_cells=[n_outlet, n_passage // 2, n_span],
        ))

    for name in O10HTopology.BLOCK_NAMES[8:10]:
        topo.blocks.append(BlockDefinition(
            name=name,
            n_cells=[n_blade_wrap // 2, n_passage // 2, n_span],
        ))

    # Define patches
    topo.patches = [
        PatchDefinition(name="inlet", patch_type="inlet"),
        PatchDefinition(name="outlet", patch_type="outlet"),
        PatchDefinition(name="blade", patch_type="wall"),
        PatchDefinition(name="hub", patch_type="wall"),
        PatchDefinition(name="shroud", patch_type="wall"),
        PatchDefinition(name="periodic_upper", patch_type="cyclic"),
        PatchDefinition(name="periodic_lower", patch_type="cyclic"),
    ]

    return topo
