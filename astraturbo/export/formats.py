"""Comprehensive format support bridge using meshio.

meshio supports ~40 mesh formats. This module provides a unified
read/write interface for all of them, plus format detection.

Formats enabled (read + write):
  VTK, VTU, PVTU, Fluent .msh, Gmsh .msh, Nastran, UNV,
  XDMF/HDF5, Abaqus, Exodus, STL, PLY, OBJ, SU2, and more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# Format extension mapping
FORMAT_MAP = {
    # Mesh formats
    ".vtk": "vtk",
    ".vtu": "vtu",
    ".pvtu": "pvtu",
    ".msh": "gmsh",         # Gmsh (also Fluent, detected by content)
    ".unv": "ideas-unv",
    ".nas": "nastran",
    ".bdf": "nastran",
    ".su2": "su2",
    ".xdmf": "xdmf",
    ".exo": "exodus",
    ".e": "exodus",
    ".med": "med",
    ".mesh": "medit",
    ".stl": "stl",
    ".ply": "ply",
    ".obj": "obj",
    ".off": "off",
    ".cas": "fluentcas",     # Fluent case
    ".cgns": "cgns",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".csv": "csv",
    ".xml": "dolfin-xml",
    ".inp": "abaqus",
    # Tecplot
    ".plt": "tecplot",
    ".tec": "tecplot",
    # EnSight
    ".case": "ensight",
    ".encas": "ensight",
    # UGRID
    ".ugrid": "ugrid",
    # PLOT3D
    ".xyz": "plot3d",
    ".p3d": "plot3d",
    ".q": "plot3d-q",
    # Additional
    ".dat": "ambiguous",     # Tecplot or Fluent — detected by content
}

# Human-readable descriptions
FORMAT_DESCRIPTIONS = {
    "vtk": "VTK Legacy",
    "vtu": "VTK Unstructured XML",
    "pvtu": "VTK Parallel Unstructured",
    "gmsh": "Gmsh",
    "ideas-unv": "I-DEAS Universal (UNV)",
    "nastran": "Nastran",
    "su2": "SU2",
    "xdmf": "XDMF + HDF5",
    "exodus": "Exodus II",
    "medit": "Medit",
    "stl": "STL (Stereolithography)",
    "ply": "Stanford PLY",
    "obj": "Wavefront OBJ",
    "fluentcas": "ANSYS Fluent Case",
    "cgns": "CFD General Notation System",
    "csv": "Comma-Separated Values",
    "tecplot": "Tecplot",
    "ensight": "EnSight Gold",
    "ugrid": "UGRID (NASA)",
    "hdf5": "HDF5 Generic",
    "plot3d": "PLOT3D Structured Grid",
    "plot3d-q": "PLOT3D Solution",
    "abaqus": "Abaqus",
}


class FormatError(Exception):
    """Raised when a file format cannot be detected or is unsupported."""


def detect_format(filepath: str | Path) -> str:
    """Detect mesh file format from extension and content.

    Args:
        filepath: Path to the file.

    Returns:
        Format string compatible with meshio.

    Raises:
        FormatError: If format cannot be determined.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FormatError(f"File not found: {filepath}")

    ext = filepath.suffix.lower()

    # Check extension mapping
    if ext in FORMAT_MAP:
        fmt = FORMAT_MAP[ext]

        # Disambiguate .msh (Gmsh vs Fluent)
        if ext == ".msh":
            try:
                with open(filepath, "rb") as f:
                    header = f.read(256)
                if b"$MeshFormat" in header:
                    return "gmsh"
                elif b"(0 " in header or b"(2 " in header:
                    return "ansys"
            except Exception:
                pass
            return "gmsh"

        # Disambiguate .dat (Tecplot vs Fluent vs generic)
        if ext == ".dat" or fmt == "ambiguous":
            try:
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    first_lines = f.read(512)
                upper = first_lines.upper()
                if "TITLE" in upper or "VARIABLES" in upper or "ZONE" in upper:
                    return "tecplot"
                elif "(" in first_lines and ")" in first_lines:
                    return "fluentdat"
            except Exception:
                pass
            return "tecplot"  # Default .dat to Tecplot

        return fmt

    # Check content for headerless formats
    try:
        with open(filepath, "rb") as f:
            header = f.read(min(4096, filepath.stat().st_size))

        if b"FoamFile" in header or b"OpenFOAM" in header:
            return "openfoam"
        if b"$MeshFormat" in header:
            return "gmsh"
        if b"solid" in header[:10]:
            return "stl"
    except Exception:
        pass

    raise FormatError(
        f"Cannot detect format for: {filepath}\n"
        f"Supported extensions: {', '.join(sorted(FORMAT_MAP.keys()))}"
    )


def read_mesh(filepath: str | Path, file_format: str | None = None) -> dict:
    """Read a mesh file in any supported format.

    Uses meshio for most formats, with custom readers for OpenFOAM
    and other special cases.

    Args:
        filepath: Path to the mesh file.
        file_format: Format override (None = auto-detect).

    Returns:
        Dict with 'points' (NDArray), 'cells' (list), 'point_data' (dict),
        'cell_data' (dict), 'format' (str).

    Raises:
        FormatError: If the format is unsupported.
    """
    filepath = Path(filepath)
    fmt = file_format or detect_format(filepath)

    # Special case: OpenFOAM
    if fmt == "openfoam":
        from .openfoam_reader import read_openfoam_points
        points = read_openfoam_points(filepath)
        return {
            "points": points,
            "cells": [],
            "point_data": {},
            "cell_data": {},
            "format": "openfoam",
        }

    # Special case: PLOT3D
    if fmt in ("plot3d", "plot3d-q") or filepath.suffix in (".xyz", ".p3d"):
        return _read_plot3d(filepath)

    # Special case: Tecplot
    if fmt == "tecplot":
        return _read_tecplot(filepath)

    # Special case: EnSight
    if fmt == "ensight":
        return _read_ensight(filepath)

    # Special case: UGRID
    if fmt == "ugrid":
        return _read_ugrid(filepath)

    # Special case: Generic HDF5
    if fmt == "hdf5":
        return _read_hdf5_generic(filepath)

    # Special case: CGNS (prefer native reader for structured data)
    if fmt == "cgns":
        try:
            from .cgns_reader import read_cgns
            data = read_cgns(filepath)
            all_pts = []
            for zone in data["zones"]:
                if len(zone["points"]) > 0:
                    all_pts.append(zone["points"])
            points = np.concatenate(all_pts) if all_pts else np.empty((0, 3))
            return {
                "points": points,
                "cells": [],
                "point_data": {},
                "cell_data": {},
                "format": "cgns",
                "zones": data["zones"],
            }
        except Exception:
            pass  # Fall through to meshio

    # All other formats via meshio
    try:
        import meshio
        mesh = meshio.read(filepath, file_format=fmt if fmt != FORMAT_MAP.get(filepath.suffix.lower()) else None)
        return {
            "points": np.asarray(mesh.points, dtype=np.float64),
            "cells": [(c.type, c.data) for c in mesh.cells],
            "point_data": dict(mesh.point_data),
            "cell_data": dict(mesh.cell_data),
            "format": fmt,
        }
    except ImportError:
        raise FormatError("meshio is required for this format. Install with: pip install meshio")
    except Exception as e:
        raise FormatError(f"Error reading {filepath} as {fmt}: {e}")


def write_mesh(
    filepath: str | Path,
    points: NDArray[np.float64],
    cells: list | None = None,
    point_data: dict | None = None,
    cell_data: dict | None = None,
    file_format: str | None = None,
) -> None:
    """Write a mesh file in any supported format.

    Args:
        filepath: Output file path.
        points: (N, 3) array of point coordinates.
        cells: List of (cell_type, connectivity_array) tuples.
        point_data: Dict of point-associated data arrays.
        cell_data: Dict of cell-associated data arrays.
        file_format: Format override (None = detect from extension).
    """
    filepath = Path(filepath)
    fmt = file_format or FORMAT_MAP.get(filepath.suffix.lower())

    if fmt is None:
        raise FormatError(
            f"Cannot determine format from extension '{filepath.suffix}'.\n"
            f"Supported: {', '.join(sorted(FORMAT_MAP.keys()))}"
        )

    # Special case: PLOT3D
    if fmt == "plot3d" or filepath.suffix in (".xyz", ".p3d"):
        _write_plot3d(filepath, points)
        return

    # All other formats via meshio
    import meshio

    if cells is None:
        # Default to point cloud (vertex cells)
        cells = [("vertex", np.arange(len(points)).reshape(-1, 1))]

    mesh = meshio.Mesh(
        points=points,
        cells=[meshio.CellBlock(ctype, cdata) for ctype, cdata in cells],
        point_data=point_data or {},
        cell_data=cell_data or {},
    )
    meshio.write(filepath, mesh, file_format=fmt)


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
        file_format: Format string. None = auto-detect from extension.
    """
    import meshio

    all_points = []
    all_cells = []
    point_offset = 0

    for block in blocks:
        ni, nj = block.shape[0], block.shape[1]
        dim = block.shape[2]

        if dim == 2:
            pts = np.zeros((ni * nj, 3), dtype=np.float64)
            pts[:, :2] = block.reshape(-1, 2)
        else:
            pts = block.reshape(-1, 3)

        all_points.append(pts)

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


# ──────────────────────────────────────────────────────────
# PLOT3D support (native — not in meshio)
# ──────────────────────────────────────────────────────────

def _read_plot3d(filepath: Path) -> dict:
    """Read a PLOT3D structured grid file (.xyz).

    Format:
        nblocks
        ni1 nj1 nk1
        ni2 nj2 nk2
        ...
        x-coords block1
        y-coords block1
        z-coords block1
        x-coords block2
        ...
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        n_blocks = int(f.readline().strip())
        dims = []
        for _ in range(n_blocks):
            parts = f.readline().strip().split()
            dims.append([int(p) for p in parts])

        all_points = []
        for ni, nj, nk in dims:
            n_pts = ni * nj * nk
            coords = []
            for _ in range(3):  # x, y, z
                values = []
                while len(values) < n_pts:
                    line = f.readline().strip()
                    values.extend(float(v) for v in line.split())
                coords.append(np.array(values[:n_pts]))

            x, y, z = coords
            block_pts = np.column_stack((x, y, z))
            all_points.append(block_pts)

    points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3))
    return {
        "points": points,
        "cells": [],
        "point_data": {},
        "cell_data": {},
        "format": "plot3d",
        "block_dims": dims,
    }


def _write_plot3d(filepath: Path, points: NDArray[np.float64], block_dims: list | None = None) -> None:
    """Write a PLOT3D structured grid file (.xyz).

    Args:
        filepath: Output path.
        points: (N, 3) point array.
        block_dims: List of [ni, nj, nk] per block. None = single block.
    """
    filepath = Path(filepath)

    if block_dims is None:
        n = len(points)
        block_dims = [[n, 1, 1]]

    with open(filepath, "w") as f:
        f.write(f"{len(block_dims)}\n")
        for dims in block_dims:
            f.write(f"{dims[0]} {dims[1]} {dims[2]}\n")

        offset = 0
        for dims in block_dims:
            n_pts = dims[0] * dims[1] * dims[2]
            block_pts = points[offset:offset + n_pts]
            for d in range(3):
                vals = block_pts[:, d]
                for i in range(0, len(vals), 6):
                    chunk = vals[i:i+6]
                    f.write(" ".join(f"{v:.10e}" for v in chunk) + "\n")
            offset += n_pts


# ──────────────────────────────────────────────────────────
# Tecplot support (native reader/writer)
# ──────────────────────────────────────────────────────────

def _read_tecplot(filepath: Path) -> dict:
    """Read a Tecplot ASCII data file (.plt, .dat).

    Supports the common POINT and BLOCK data packing formats:
        TITLE = "..."
        VARIABLES = "x" "y" "z" ...
        ZONE T="...", I=N, J=M, K=L, F=POINT
        x1 y1 z1 var1 ...
        x2 y2 z2 var2 ...
    """
    filepath = Path(filepath)
    variables = []
    zones = []
    current_zone = None
    current_data = []
    n_i, n_j, n_k = 0, 0, 0
    data_format = "POINT"

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            upper = line.upper()

            if not line or line.startswith("#"):
                continue

            # Parse TITLE
            if upper.startswith("TITLE"):
                continue

            # Parse VARIABLES
            if upper.startswith("VARIABLES"):
                # Extract variable names from quoted strings
                import re
                variables = re.findall(r'"([^"]*)"', line)
                if not variables:
                    # Try space-separated after =
                    parts = line.split("=", 1)
                    if len(parts) > 1:
                        variables = parts[1].replace(",", " ").split()
                continue

            # Parse ZONE header
            if upper.startswith("ZONE"):
                # Save previous zone
                if current_zone is not None and current_data:
                    current_zone["data"] = np.array(current_data, dtype=np.float64)
                    zones.append(current_zone)
                    current_data = []

                current_zone = {"name": ""}
                # Parse zone attributes
                import re
                t_match = re.search(r'T\s*=\s*"([^"]*)"', line, re.IGNORECASE)
                if t_match:
                    current_zone["name"] = t_match.group(1)
                i_match = re.search(r'I\s*=\s*(\d+)', line, re.IGNORECASE)
                j_match = re.search(r'J\s*=\s*(\d+)', line, re.IGNORECASE)
                k_match = re.search(r'K\s*=\s*(\d+)', line, re.IGNORECASE)
                f_match = re.search(r'F\s*=\s*(\w+)', line, re.IGNORECASE)
                n_i = int(i_match.group(1)) if i_match else 0
                n_j = int(j_match.group(1)) if j_match else 0
                n_k = int(k_match.group(1)) if k_match else 0
                data_format = f_match.group(1).upper() if f_match else "POINT"
                current_zone["dims"] = [n_i, n_j, n_k]
                current_zone["format"] = data_format
                continue

            # Parse data lines
            try:
                values = [float(v) for v in line.split()]
                if values:
                    current_data.append(values)
            except ValueError:
                continue

    # Save last zone
    if current_zone is not None and current_data:
        current_zone["data"] = np.array(current_data, dtype=np.float64)
        zones.append(current_zone)

    # Build result
    all_points = []
    point_data = {}

    for zone in zones:
        data = zone.get("data")
        if data is None or data.size == 0:
            continue
        n_cols = data.shape[1]
        # First 2 or 3 columns are coordinates
        n_coord = min(3, n_cols)
        pts = data[:, :n_coord]
        if n_coord == 2:
            pts = np.column_stack((pts, np.zeros(len(pts))))
        all_points.append(pts)

        # Remaining columns are field data
        for col_idx in range(n_coord, n_cols):
            var_name = variables[col_idx] if col_idx < len(variables) else f"var{col_idx}"
            point_data.setdefault(var_name, []).append(data[:, col_idx])

    points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3))

    # Flatten point_data lists
    for key in point_data:
        point_data[key] = np.concatenate(point_data[key])

    return {
        "points": points,
        "cells": [],
        "point_data": point_data,
        "cell_data": {},
        "format": "tecplot",
        "variables": variables,
        "zones": zones,
    }


def _write_tecplot(
    filepath: Path,
    points: NDArray[np.float64],
    variables: list[str] | None = None,
    point_data: dict | None = None,
    zone_name: str = "Zone1",
    dims: list[int] | None = None,
) -> None:
    """Write a Tecplot ASCII data file."""
    filepath = Path(filepath)
    n_pts = len(points)
    dim = points.shape[1]

    if variables is None:
        variables = ["X", "Y", "Z"][:dim]
    if point_data:
        variables = variables + list(point_data.keys())

    if dims is None:
        dims = [n_pts, 1, 1]

    with open(filepath, "w") as f:
        f.write('TITLE = "AstraTurbo Export"\n')
        var_str = " ".join(f'"{v}"' for v in variables)
        f.write(f"VARIABLES = {var_str}\n")
        f.write(f'ZONE T="{zone_name}", I={dims[0]}, J={dims[1]}, K={dims[2]}, F=POINT\n')
        for i in range(n_pts):
            vals = list(points[i])
            if point_data:
                for key in point_data:
                    vals.append(float(point_data[key][i]))
            f.write(" ".join(f"{v:.10e}" for v in vals) + "\n")


# ──────────────────────────────────────────────────────────
# EnSight Gold support
# ──────────────────────────────────────────────────────────

def _read_ensight(filepath: Path) -> dict:
    """Read an EnSight Gold case file (.case).

    Parses the .case file to find the geometry file, then reads coordinates.
    """
    filepath = Path(filepath)
    case_dir = filepath.parent
    geo_file = None

    # Parse .case file for geometry reference
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("model:"):
                parts = line.split()
                if len(parts) >= 2:
                    geo_file = parts[-1]

    if geo_file is None:
        raise FormatError(f"No geometry file referenced in EnSight case: {filepath}")

    geo_path = case_dir / geo_file
    if not geo_path.exists():
        raise FormatError(f"EnSight geometry file not found: {geo_path}")

    # Try meshio for EnSight geo reading
    try:
        import meshio
        mesh = meshio.read(geo_path, file_format="ensight")
        return {
            "points": np.asarray(mesh.points, dtype=np.float64),
            "cells": [(c.type, c.data) for c in mesh.cells],
            "point_data": dict(mesh.point_data),
            "cell_data": dict(mesh.cell_data),
            "format": "ensight",
        }
    except Exception:
        # Fallback: parse EnSight geo ASCII
        return _read_ensight_geo_ascii(geo_path)


def _read_ensight_geo_ascii(filepath: Path) -> dict:
    """Read an EnSight Gold ASCII geometry file."""
    points = []
    reading_coords = False
    n_points = 0
    coord_count = 0

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "coordinates":
                reading_coords = True
                continue
            if reading_coords and n_points == 0:
                try:
                    n_points = int(line)
                except ValueError:
                    reading_coords = False
                continue
            if reading_coords and n_points > 0:
                try:
                    val = float(line)
                    points.append(val)
                    coord_count += 1
                    if coord_count >= n_points * 3:
                        reading_coords = False
                except ValueError:
                    if coord_count >= n_points * 3:
                        reading_coords = False

    if points:
        arr = np.array(points, dtype=np.float64)
        pts = arr.reshape(3, -1).T  # EnSight stores x,x,x...,y,y,y...,z,z,z...
    else:
        pts = np.empty((0, 3))

    return {
        "points": pts,
        "cells": [],
        "point_data": {},
        "cell_data": {},
        "format": "ensight",
    }


# ──────────────────────────────────────────────────────────
# UGRID support (NASA format)
# ──────────────────────────────────────────────────────────

def _read_ugrid(filepath: Path) -> dict:
    """Read a UGRID file (NASA unstructured grid format).

    ASCII format:
        n_nodes n_tris n_quads n_tets n_pyramids n_prisms n_hexes
        x1 y1 z1
        x2 y2 z2
        ...
        tri_connectivity (3 per line)
        quad_connectivity (4 per line)
        ...
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        # First line: counts
        counts = f.readline().strip().split()
        n_nodes = int(counts[0])
        n_tris = int(counts[1]) if len(counts) > 1 else 0
        n_quads = int(counts[2]) if len(counts) > 2 else 0
        n_tets = int(counts[3]) if len(counts) > 3 else 0

        # Read nodes
        points = np.empty((n_nodes, 3), dtype=np.float64)
        for i in range(n_nodes):
            parts = f.readline().strip().split()
            points[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

        # Read triangles
        cells = []
        if n_tris > 0:
            tris = np.empty((n_tris, 3), dtype=np.int64)
            for i in range(n_tris):
                parts = f.readline().strip().split()
                tris[i] = [int(p) - 1 for p in parts[:3]]  # 1-indexed to 0-indexed
            cells.append(("triangle", tris))

        # Read quads
        if n_quads > 0:
            quads = np.empty((n_quads, 4), dtype=np.int64)
            for i in range(n_quads):
                parts = f.readline().strip().split()
                quads[i] = [int(p) - 1 for p in parts[:4]]
            cells.append(("quad", quads))

    return {
        "points": points,
        "cells": cells,
        "point_data": {},
        "cell_data": {},
        "format": "ugrid",
    }


# ──────────────────────────────────────────────────────────
# Generic HDF5 reader
# ──────────────────────────────────────────────────────────

def _read_hdf5_generic(filepath: Path) -> dict:
    """Read a generic HDF5 file and extract coordinate-like datasets.

    Searches for datasets named like coordinates (x, y, z, X, Y, Z,
    CoordinateX, etc.) and combines them. Falls back to returning all
    datasets as point_data.
    """
    filepath = Path(filepath)

    try:
        import h5py
    except ImportError:
        raise FormatError("h5py is required for HDF5 reading. Install with: pip install h5py")

    coordinate_names = {
        "x", "y", "z", "X", "Y", "Z",
        "CoordinateX", "CoordinateY", "CoordinateZ",
        "coordinatex", "coordinatey", "coordinatez",
        "Points", "points", "Coordinates", "coordinates",
    }

    found_coords = {}
    all_datasets = {}

    def _scan(group, prefix=""):
        for key in group.keys():
            item = group[key]
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                all_datasets[full_key] = np.array(item, dtype=np.float64)
                if key in coordinate_names:
                    found_coords[key] = np.array(item, dtype=np.float64)
            elif isinstance(item, h5py.Group):
                _scan(item, full_key)

    with h5py.File(filepath, "r") as f:
        _scan(f)

    # Try to assemble coordinates
    points = np.empty((0, 3))

    # Check for combined array (Points, Coordinates)
    for name in ("Points", "points", "Coordinates", "coordinates"):
        if name in found_coords:
            arr = found_coords[name]
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    points = np.column_stack((arr, np.zeros(len(arr))))
                else:
                    points = arr[:, :3]
                break

    # Check for separate x, y, z arrays
    if len(points) == 0:
        x = found_coords.get("x", found_coords.get("X", found_coords.get("CoordinateX")))
        y = found_coords.get("y", found_coords.get("Y", found_coords.get("CoordinateY")))
        z = found_coords.get("z", found_coords.get("Z", found_coords.get("CoordinateZ")))

        if x is not None and y is not None:
            if z is not None:
                points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            else:
                points = np.column_stack((x.ravel(), y.ravel(), np.zeros_like(x.ravel())))

    # Remaining datasets as point_data
    point_data = {}
    for key, arr in all_datasets.items():
        name = key.split("/")[-1]
        if name not in coordinate_names and arr.ndim <= 2:
            point_data[name] = arr

    return {
        "points": points,
        "cells": [],
        "point_data": point_data,
        "cell_data": {},
        "format": "hdf5",
        "all_datasets": list(all_datasets.keys()),
    }


def list_supported_formats() -> dict[str, dict]:
    """Return a dictionary of all supported formats with capabilities.

    Returns:
        Dict mapping format name to {extensions, read, write, description}.
    """
    formats = {}

    # Formats with native support
    formats["openfoam_points"] = {
        "extensions": ["points"],
        "read": True, "write": False,
        "description": "OpenFOAM polyMesh points (ASCII)",
    }
    formats["openfoam_blockmesh"] = {
        "extensions": ["blockMeshDict"],
        "read": False, "write": True,
        "description": "OpenFOAM blockMeshDict",
    }
    formats["cgns_structured"] = {
        "extensions": [".cgns"],
        "read": True, "write": True,
        "description": "CGNS structured zones (HDF5)",
    }
    formats["plot3d"] = {
        "extensions": [".xyz", ".p3d", ".q"],
        "read": True, "write": True,
        "description": "PLOT3D structured grid",
    }
    formats["tecplot"] = {
        "extensions": [".plt", ".dat", ".tec"],
        "read": True, "write": True,
        "description": "Tecplot ASCII data",
    }
    formats["ensight"] = {
        "extensions": [".case", ".encas"],
        "read": True, "write": False,
        "description": "EnSight Gold case",
    }
    formats["ugrid"] = {
        "extensions": [".ugrid"],
        "read": True, "write": False,
        "description": "UGRID (NASA unstructured grid)",
    }
    formats["hdf5_generic"] = {
        "extensions": [".h5", ".hdf5"],
        "read": True, "write": False,
        "description": "HDF5 generic (auto-detect coordinate datasets)",
    }
    formats["yaml_project"] = {
        "extensions": [".yaml", ".yml"],
        "read": True, "write": True,
        "description": "AstraTurbo project file",
    }
    formats["bladedesigner_xml"] = {
        "extensions": [".xml"],
        "read": True, "write": False,
        "description": "Legacy XML project import",
    }
    formats["csv_profile"] = {
        "extensions": [".csv"],
        "read": True, "write": True,
        "description": "Profile point data (x,y CSV)",
    }
    formats["stl_native"] = {
        "extensions": [".stl"],
        "read": True, "write": True,
        "description": "STL Stereolithography (native, no dependencies)",
    }

    # Formats via meshio
    meshio_formats = {
        "vtk": ([".vtk"], "VTK Legacy"),
        "vtu": ([".vtu"], "VTK Unstructured XML"),
        "pvtu": ([".pvtu"], "VTK Parallel Unstructured"),
        "gmsh": ([".msh"], "Gmsh"),
        "nastran": ([".nas", ".bdf"], "Nastran"),
        "ideas-unv": ([".unv"], "I-DEAS Universal (UNV)"),
        "su2": ([".su2"], "SU2 mesh"),
        "stl_meshio": ([".stl"], "STL (via meshio)"),
        "fluent": ([".cas", ".msh"], "ANSYS Fluent"),
        "xdmf": ([".xdmf"], "XDMF + HDF5"),
        "exodus": ([".exo", ".e"], "Exodus II"),
        "abaqus": ([".inp"], "Abaqus"),
        "obj": ([".obj"], "Wavefront OBJ"),
        "ply": ([".ply"], "Stanford PLY"),
        "medit": ([".mesh"], "Medit"),
        "med": ([".med"], "MED (Salome)"),
    }
    for name, (exts, desc) in meshio_formats.items():
        formats[name] = {
            "extensions": exts,
            "read": True, "write": True,
            "description": f"{desc} (via meshio)",
        }

    # Optional CAD formats (cadquery required)
    formats["step"] = {
        "extensions": [".step", ".stp"],
        "read": False, "write": True,
        "description": "STEP (requires cadquery)",
    }
    formats["iges"] = {
        "extensions": [".iges", ".igs"],
        "read": False, "write": True,
        "description": "IGES (requires cadquery)",
    }

    return formats
