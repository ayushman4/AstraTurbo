"""CAD geometry export: STEP, IGES, STL.

Uses cadquery/OCP for STEP and IGES (optional dependency).
STL can be written from point arrays without CAD dependencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class CADExportError(Exception):
    """Raised when CAD export fails."""


def export_step(
    filepath: str | Path,
    points_or_surface: NDArray[np.float64] | Any,
    **kwargs,
) -> None:
    """Export geometry to STEP format.

    Args:
        filepath: Output .step file path.
        points_or_surface: Either a (N, 3) point array (creates a B-spline surface)
            or a cadquery/OCP shape object.

    Raises:
        CADExportError: If cadquery is not installed.
    """
    filepath = Path(filepath)

    try:
        import cadquery as cq
    except ImportError:
        raise CADExportError(
            "cadquery is required for STEP export.\n"
            "Install with: pip install cadquery"
        )

    if isinstance(points_or_surface, np.ndarray):
        # Create a simple B-spline from points
        pts = points_or_surface
        if pts.ndim == 2 and pts.shape[1] == 3:
            spline = cq.Workplane("XY").spline(
                [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
            )
            cq.exporters.export(spline, str(filepath), exportType="STEP")
        else:
            raise CADExportError(f"Expected (N, 3) array, got shape {pts.shape}")
    else:
        # Assume it's already a cadquery object
        cq.exporters.export(points_or_surface, str(filepath), exportType="STEP")


def export_iges(
    filepath: str | Path,
    points_or_surface: NDArray[np.float64] | Any,
) -> None:
    """Export geometry to IGES format.

    Requires cadquery with IGES support.
    """
    filepath = Path(filepath)

    try:
        import cadquery as cq
    except ImportError:
        raise CADExportError(
            "cadquery is required for IGES export.\n"
            "Install with: pip install cadquery"
        )

    if isinstance(points_or_surface, np.ndarray):
        pts = points_or_surface
        if pts.ndim == 2 and pts.shape[1] == 3:
            spline = cq.Workplane("XY").spline(
                [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
            )
            # IGES export via OCP directly
            try:
                from OCP.IGESControl import IGESControl_Writer
                from OCP.IFSelect import IFSelect_RetDone

                writer = IGESControl_Writer()
                writer.AddShape(spline.val().wrapped)
                if writer.Write(str(filepath)) != IFSelect_RetDone:
                    raise CADExportError("IGES write failed")
            except ImportError:
                raise CADExportError(
                    "IGES export requires OCP (OpenCASCADE Python).\n"
                    "Install with: pip install cadquery"
                )
    else:
        raise CADExportError("IGES export requires a point array or cadquery shape.")


def write_stl_ascii(
    filepath: str | Path,
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    solid_name: str = "astraturbo",
) -> None:
    """Write an ASCII STL file from vertices and triangle indices.

    No external dependencies required.

    Args:
        filepath: Output .stl file path.
        vertices: (N, 3) vertex coordinates.
        triangles: (M, 3) triangle vertex indices.
        solid_name: Name for the STL solid.
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        f.write(f"solid {solid_name}\n")
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            n_len = np.linalg.norm(normal)
            if n_len > 1e-15:
                normal /= n_len

            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def write_stl_from_surface(
    filepath: str | Path,
    surface_points: NDArray[np.float64],
    ni: int,
    nj: int,
) -> None:
    """Write an STL file from a structured surface grid.

    Triangulates the structured grid and writes STL.
    No external dependencies required.

    Args:
        filepath: Output .stl file path.
        surface_points: (ni*nj, 3) or (ni, nj, 3) surface points.
        ni, nj: Grid dimensions.
    """
    if surface_points.ndim == 3:
        surface_points = surface_points.reshape(-1, 3)

    # Triangulate structured grid
    triangles = []
    for i in range(ni - 1):
        for j in range(nj - 1):
            p00 = i * nj + j
            p10 = (i + 1) * nj + j
            p01 = i * nj + (j + 1)
            p11 = (i + 1) * nj + (j + 1)
            triangles.append([p00, p10, p11])
            triangles.append([p00, p11, p01])

    triangles = np.array(triangles, dtype=np.int64)
    write_stl_ascii(filepath, surface_points, triangles)


def read_stl(filepath: str | Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Read an ASCII or binary STL file.

    Uses meshio for robust reading.

    Args:
        filepath: Path to .stl file.

    Returns:
        (vertices, triangles) tuple.
    """
    try:
        import meshio
        mesh = meshio.read(filepath, file_format="stl")
        vertices = np.asarray(mesh.points, dtype=np.float64)
        triangles = np.asarray(mesh.cells[0].data, dtype=np.int64)
        return vertices, triangles
    except ImportError:
        # Fallback: simple ASCII STL reader
        return _read_stl_ascii(filepath)


def _read_stl_ascii(filepath: Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Simple ASCII STL reader (no dependencies)."""
    filepath = Path(filepath)
    vertices = []
    triangles = []
    current_tri = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                v = [float(parts[1]), float(parts[2]), float(parts[3])]
                idx = len(vertices)
                vertices.append(v)
                current_tri.append(idx)
                if len(current_tri) == 3:
                    triangles.append(current_tri)
                    current_tri = []

    return (
        np.array(vertices, dtype=np.float64),
        np.array(triangles, dtype=np.int64),
    )
