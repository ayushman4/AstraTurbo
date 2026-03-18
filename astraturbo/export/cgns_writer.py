"""CGNS mesh writer using h5py.

Writes structured mesh data to CGNS-HDF5 format. This replaces the
V1 CgnsCreator.py which used the old CGNS Python binding that was
nearly impossible to install cross-platform.

CGNS uses HDF5 internally, so we write the tree structure directly
with h5py following the CGNS/SIDS specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import h5py


def write_cgns_structured(
    filepath: str | Path,
    blocks: list[NDArray[np.float64]],
    block_names: list[str] | None = None,
    base_name: str = "AstraTurbo",
    patches: dict[str, dict[str, str]] | None = None,
) -> None:
    """Write structured mesh blocks to a CGNS-HDF5 file.

    Args:
        filepath: Output .cgns file path.
        blocks: List of structured blocks, each (Ni, Nj, Nk, 3) or (Ni, Nj, 3).
            For 2D blocks (Ni, Nj, 2), z-coordinates are set to 0.
        block_names: Optional names for each block.
        base_name: Name of the CGNS base.
        patches: Optional dict of zone_name -> {face_name: bc_type} for writing BCs.
    """
    filepath = Path(filepath)
    n_blocks = len(blocks)

    if block_names is None:
        block_names = [f"Zone_{i}" for i in range(n_blocks)]

    with h5py.File(filepath, "w") as f:
        # Root node attributes
        f.attrs["label"] = np.bytes_("CGNSLibraryVersion_t")
        f.attrs["type"] = np.bytes_("R4")

        # Library version
        f.create_dataset(
            " CGNSLibraryVersion",
            data=np.array([3.4], dtype=np.float32),
        )

        # Create base node
        base = f.create_group(base_name)
        base.attrs["label"] = np.bytes_("CGNSBase_t")
        # Cell dimension and physical dimension
        cell_dim = 3
        phys_dim = 3
        base.create_dataset(" data", data=np.array([cell_dim, phys_dim], dtype=np.int32))

        for idx, (block, name) in enumerate(zip(blocks, block_names)):
            _write_zone(base, block, name)

        # Write boundary conditions if patches provided
        if patches:
            for zone_name, zone_patches in patches.items():
                if zone_name in base:
                    zone_grp = base[zone_name]
                    write_cgns_boundary_conditions(zone_grp, zone_patches)


def _write_zone(
    base_group: h5py.Group,
    block: NDArray[np.float64],
    zone_name: str,
) -> None:
    """Write a single structured zone to the CGNS base."""
    # Handle 2D blocks by adding z=0
    if block.ndim == 3 and block.shape[-1] == 2:
        # (Ni, Nj, 2) -> (Ni, Nj, 1, 3)
        ni, nj = block.shape[0], block.shape[1]
        block_3d = np.zeros((ni, nj, 1, 3), dtype=np.float64)
        block_3d[:, :, 0, 0] = block[:, :, 0]
        block_3d[:, :, 0, 1] = block[:, :, 1]
        block = block_3d
    elif block.ndim == 3 and block.shape[-1] == 3:
        # (Ni, Nj, 3) -> (Ni, Nj, 1, 3)
        ni, nj = block.shape[0], block.shape[1]
        block = block.reshape(ni, nj, 1, 3)

    ni, nj, nk = block.shape[0], block.shape[1], block.shape[2]

    zone = base_group.create_group(zone_name)
    zone.attrs["label"] = np.bytes_("Zone_t")

    # Zone type: structured
    zt = zone.create_group("ZoneType")
    zt.attrs["label"] = np.bytes_("ZoneType_t")
    zt.create_dataset(" data", data=np.bytes_("Structured"))

    # Zone size: [[Ni, Nj, Nk], [Ni-1, Nj-1, Nk-1], [0, 0, 0]]
    zone_size = np.array([
        [ni, nj, nk],
        [ni - 1, nj - 1, max(nk - 1, 1)],
        [0, 0, 0],
    ], dtype=np.int32)
    zone.create_dataset(" data", data=zone_size)

    # Grid coordinates
    grid = zone.create_group("GridCoordinates")
    grid.attrs["label"] = np.bytes_("GridCoordinates_t")

    for coord_idx, coord_name in enumerate(["CoordinateX", "CoordinateY", "CoordinateZ"]):
        coord_grp = grid.create_group(coord_name)
        coord_grp.attrs["label"] = np.bytes_("DataArray_t")
        coord_grp.create_dataset(
            " data",
            data=block[:, :, :, coord_idx].astype(np.float64),
        )


def write_cgns_2d(
    filepath: str | Path,
    blocks: list[NDArray[np.float64]],
    block_names: list[str] | None = None,
    flow_data: dict[str, list[NDArray[np.float64]]] | None = None,
) -> None:
    """Write 2D structured mesh (S2m plane) to CGNS.

    Simplified interface for the SCM mesher output.

    Args:
        filepath: Output .cgns file path.
        blocks: List of (Ni, Nj, 2) blocks in (z, r) coordinates.
        block_names: Optional zone names.
        flow_data: Optional dict mapping field names to lists of
            (Ni, Nj) arrays per block (e.g. {'beta': [arr1, arr2, ...]}).
    """
    filepath = Path(filepath)
    n_blocks = len(blocks)

    if block_names is None:
        block_names = [f"Block_{i}" for i in range(n_blocks)]

    # Convert 2D (z, r) to 3D (z, r, 0)
    blocks_3d = []
    for block in blocks:
        ni, nj = block.shape[0], block.shape[1]
        b3d = np.zeros((ni, nj, 3), dtype=np.float64)
        b3d[:, :, 0] = block[:, :, 0]  # z -> x
        b3d[:, :, 1] = block[:, :, 1]  # r -> y
        blocks_3d.append(b3d)

    write_cgns_structured(filepath, blocks_3d, block_names)

    # Append flow data if provided
    if flow_data:
        with h5py.File(filepath, "a") as f:
            base_name = list(f.keys())[0]
            if base_name.startswith(" "):
                base_name = list(f.keys())[1]
            base = f[base_name]

            for zone_idx, zone_name in enumerate(block_names):
                if zone_name not in base:
                    continue
                zone = base[zone_name]

                sol = zone.create_group("FlowSolution")
                sol.attrs["label"] = np.bytes_("FlowSolution_t")

                for field_name, field_arrays in flow_data.items():
                    if zone_idx < len(field_arrays):
                        fd = sol.create_group(field_name)
                        fd.attrs["label"] = np.bytes_("DataArray_t")
                        fd.create_dataset(
                            " data",
                            data=field_arrays[zone_idx].astype(np.float64),
                        )


# ── CGNS Boundary Condition Writers ──────────────────────────


# Mapping from logical BC names to CGNS BC types
_CGNS_BC_TYPE_MAP = {
    "inlet": "BCInflowSubsonic",
    "outlet": "BCOutflowSubsonic",
    "blade": "BCWallViscous",
    "hub": "BCWallViscous",
    "shroud": "BCWallViscous",
    "periodic_upper": "BCSymmetryPlane",
    "periodic_lower": "BCSymmetryPlane",
    "periodic": "BCSymmetryPlane",
}


def write_cgns_boundary_conditions(
    zone_group: h5py.Group,
    patch_map: dict[str, str],
) -> None:
    """Write ZoneBC nodes to a CGNS zone.

    Args:
        zone_group: h5py group for the zone.
        patch_map: Dict mapping face names to BC types.
            Keys: face names (e.g. "bottom", "top", "left", "right")
            Values: logical BC types (e.g. "inlet", "outlet", "blade")
    """
    zonebc = zone_group.create_group("ZoneBC")
    zonebc.attrs["label"] = np.bytes_("ZoneBC_t")

    for face_name, bc_logical in patch_map.items():
        cgns_type = _CGNS_BC_TYPE_MAP.get(bc_logical, "BCGeneral")

        bc_node = zonebc.create_group(face_name)
        bc_node.attrs["label"] = np.bytes_("BC_t")
        bc_node.create_dataset(" data", data=np.bytes_(cgns_type))

        # GridLocation node
        gl = bc_node.create_group("GridLocation")
        gl.attrs["label"] = np.bytes_("GridLocation_t")
        gl.create_dataset(" data", data=np.bytes_("FaceCenter"))


def write_cgns_connectivity(
    zone_group: h5py.Group,
    donor_zone_name: str,
    transform: list[int] | None = None,
) -> None:
    """Write inter-block 1-to-1 connectivity to a CGNS zone.

    Args:
        zone_group: h5py group for the zone.
        donor_zone_name: Name of the donor (connected) zone.
        transform: Short-form transformation indices (default: [1, 2, 3] = identity).
    """
    if transform is None:
        transform = [1, 2, 3]

    conn_name = f"conn_{donor_zone_name}"
    conn = zone_group.create_group(conn_name)
    conn.attrs["label"] = np.bytes_("GridConnectivity1to1_t")
    conn.create_dataset(" data", data=np.bytes_(donor_zone_name))

    # Transform
    tr = conn.create_group("Transform")
    tr.attrs["label"] = np.bytes_("\"int[IndexDimension]\"")
    tr.create_dataset(" data", data=np.array(transform, dtype=np.int32))
