"""CGNS mesh reader using h5py.

Reads structured and unstructured CGNS-HDF5 files back into numpy arrays.
This complements cgns_writer.py to provide round-trip CGNS support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import h5py


class CGNSReadError(Exception):
    """Raised when a CGNS file cannot be read."""


def read_cgns(filepath: str | Path) -> dict:
    """Read a CGNS-HDF5 file.

    Args:
        filepath: Path to .cgns file.

    Returns:
        Dict with:
          'zones': list of dicts per zone, each containing:
            'name': zone name
            'zone_type': 'Structured' or 'Unstructured'
            'dimensions': zone size array
            'coordinates': dict of 'CoordinateX/Y/Z' -> NDArray
            'points': (N, 3) combined coordinate array
            'flow_solution': dict of field_name -> NDArray (if present)
          'base_name': name of the CGNS base

    Raises:
        CGNSReadError: If the file is not valid CGNS.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise CGNSReadError(f"File not found: {filepath}")

    try:
        f = h5py.File(filepath, "r")
    except OSError as e:
        raise CGNSReadError(f"Cannot open as HDF5: {filepath}\n{e}")

    result = {"zones": [], "base_name": ""}

    try:
        # Find the base (skip library version node)
        base_group = None
        for key in f.keys():
            grp = f[key]
            if isinstance(grp, h5py.Group):
                label = grp.attrs.get("label", b"")
                if isinstance(label, bytes):
                    label = label.decode()
                if label == "CGNSBase_t" or key != " CGNSLibraryVersion":
                    if isinstance(grp, h5py.Group) and any(
                        isinstance(f[key][sk], h5py.Group) for sk in f[key].keys()
                        if not sk.startswith(" ")
                    ):
                        base_group = grp
                        result["base_name"] = key
                        break

        if base_group is None:
            raise CGNSReadError(f"No CGNS base found in {filepath}")

        # Read each zone
        for zone_name in base_group.keys():
            if zone_name.startswith(" "):
                continue

            zone_grp = base_group[zone_name]
            if not isinstance(zone_grp, h5py.Group):
                continue

            zone_data = {"name": zone_name, "coordinates": {}}

            # Zone type
            if "ZoneType" in zone_grp:
                zt = zone_grp["ZoneType"]
                if " data" in zt:
                    zt_val = zt[" data"][()]
                    if isinstance(zt_val, bytes):
                        zt_val = zt_val.decode()
                    zone_data["zone_type"] = str(zt_val).strip()
                else:
                    zone_data["zone_type"] = "Unknown"
            else:
                zone_data["zone_type"] = "Unknown"

            # Dimensions
            if " data" in zone_grp:
                zone_data["dimensions"] = zone_grp[" data"][()]
            else:
                zone_data["dimensions"] = None

            # Grid coordinates
            if "GridCoordinates" in zone_grp:
                gc = zone_grp["GridCoordinates"]
                for coord_name in ["CoordinateX", "CoordinateY", "CoordinateZ"]:
                    if coord_name in gc:
                        coord_grp = gc[coord_name]
                        if " data" in coord_grp:
                            zone_data["coordinates"][coord_name] = np.array(
                                coord_grp[" data"], dtype=np.float64
                            )

            # Combine into points array
            cx = zone_data["coordinates"].get("CoordinateX")
            cy = zone_data["coordinates"].get("CoordinateY")
            cz = zone_data["coordinates"].get("CoordinateZ")

            if cx is not None and cy is not None and cz is not None:
                zone_data["points"] = np.column_stack((
                    cx.ravel(), cy.ravel(), cz.ravel()
                ))
            elif cx is not None and cy is not None:
                zone_data["points"] = np.column_stack((
                    cx.ravel(), cy.ravel(), np.zeros_like(cx.ravel())
                ))
            else:
                zone_data["points"] = np.empty((0, 3))

            # Flow solution (if present)
            zone_data["flow_solution"] = {}
            if "FlowSolution" in zone_grp:
                fs = zone_grp["FlowSolution"]
                for field_name in fs.keys():
                    if field_name.startswith(" "):
                        continue
                    field_grp = fs[field_name]
                    if isinstance(field_grp, h5py.Group) and " data" in field_grp:
                        zone_data["flow_solution"][field_name] = np.array(
                            field_grp[" data"], dtype=np.float64
                        )

            result["zones"].append(zone_data)

    finally:
        f.close()

    if not result["zones"]:
        raise CGNSReadError(f"No zones found in CGNS file: {filepath}")

    return result


def cgns_to_points(filepath: str | Path) -> NDArray[np.float64]:
    """Read a CGNS file and return all points as a single (N, 3) array.

    Convenience function for simple use cases.
    """
    data = read_cgns(filepath)
    all_points = []
    for zone in data["zones"]:
        if len(zone["points"]) > 0:
            all_points.append(zone["points"])

    if not all_points:
        raise CGNSReadError(f"No coordinate data found in {filepath}")

    return np.concatenate(all_points, axis=0)


def cgns_info(filepath: str | Path) -> dict:
    """Get summary information about a CGNS file.

    Args:
        filepath: Path to .cgns file.

    Returns:
        Dict with summary statistics.
    """
    data = read_cgns(filepath)

    total_points = sum(len(z["points"]) for z in data["zones"])
    zone_summaries = []
    for z in data["zones"]:
        summary = {
            "name": z["name"],
            "type": z.get("zone_type", "Unknown"),
            "n_points": len(z["points"]),
        }
        if len(z["points"]) > 0:
            pts = z["points"]
            summary["x_range"] = (float(pts[:, 0].min()), float(pts[:, 0].max()))
            summary["y_range"] = (float(pts[:, 1].min()), float(pts[:, 1].max()))
            summary["z_range"] = (float(pts[:, 2].min()), float(pts[:, 2].max()))
        if z["flow_solution"]:
            summary["fields"] = list(z["flow_solution"].keys())
        zone_summaries.append(summary)

    return {
        "base_name": data["base_name"],
        "n_zones": len(data["zones"]),
        "total_points": total_points,
        "zones": zone_summaries,
    }
