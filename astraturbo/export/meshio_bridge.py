"""Multi-format mesh export via meshio.

Bridges AstraTurbo mesh data to any format supported by meshio,
including VTK, Fluent .msh, Gmsh, SU2, and more.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def export_structured_as_quads(
    filepath: str | Path,
    blocks: list[NDArray[np.float64]],
    file_format: str | None = None,
) -> None:
    """Export structured mesh blocks as quad elements via meshio.

    Converts structured blocks to unstructured quad mesh for export
    to formats that don't support structured grids.

    Args:
        filepath: Output file path.
        blocks: List of (Ni, Nj, 2) or (Ni, Nj, 3) blocks.
        file_format: Format string (e.g. 'vtk', 'su2'). None = auto-detect.
    """
    import meshio

    all_points = []
    all_cells = []
    point_offset = 0

    for block in blocks:
        ni, nj = block.shape[0], block.shape[1]
        dim = block.shape[2]

        # Flatten points
        if dim == 2:
            pts = np.zeros((ni * nj, 3), dtype=np.float64)
            pts[:, :2] = block.reshape(-1, 2)
        else:
            pts = block.reshape(-1, 3)

        all_points.append(pts)

        # Build quad connectivity
        quads = []
        for i in range(ni - 1):
            for j in range(nj - 1):
                p0 = point_offset + i * nj + j
                p1 = point_offset + (i + 1) * nj + j
                p2 = point_offset + (i + 1) * nj + (j + 1)
                p3 = point_offset + i * nj + (j + 1)
                quads.append([p0, p1, p2, p3])

        all_cells.extend(quads)
        point_offset += ni * nj

    points = np.concatenate(all_points, axis=0)
    cells = [("quad", np.array(all_cells, dtype=np.int64))]

    mesh = meshio.Mesh(points=points, cells=cells)
    meshio.write(filepath, mesh, file_format=file_format)
