"""FEA mesh export and CFD-to-FEA load mapping.

Converts blade geometry to solid mesh for structural analysis and
maps CFD surface pressures onto the FEA mesh for coupled simulations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def blade_surface_to_solid_mesh(
    surface_points: NDArray[np.float64],
    ni: int,
    nj: int,
    thickness: float = 0.002,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Extrude a blade surface mesh into a solid mesh for FEA.

    Takes the blade surface (from the CFD mesh) and creates a volumetric
    mesh by extruding inward by the blade thickness.

    Args:
        surface_points: (ni*nj, 3) blade surface points.
        ni: Points in streamwise direction.
        nj: Points in spanwise direction.
        thickness: Blade wall thickness (m) for extrusion.

    Returns:
        (nodes, elements) tuple where nodes is (N, 3) and elements is
        (M, 8) for hex elements (or (M, 4) for tet).
    """
    n_surface = ni * nj
    surface = surface_points.reshape(ni, nj, 3)

    # Compute outward normals at each point
    normals = np.zeros_like(surface)
    for i in range(ni):
        for j in range(nj):
            ip = min(i + 1, ni - 1)
            im = max(i - 1, 0)
            jp = min(j + 1, nj - 1)
            jm = max(j - 1, 0)

            du = surface[ip, j] - surface[im, j]
            dv = surface[i, jp] - surface[i, jm]
            n = np.cross(du, dv)
            n_len = np.linalg.norm(n)
            if n_len > 1e-15:
                normals[i, j] = n / n_len

    # Create inner surface (offset inward)
    inner_surface = surface - normals * thickness

    # Combine into node array: outer then inner
    outer_flat = surface.reshape(-1, 3)
    inner_flat = inner_surface.reshape(-1, 3)
    nodes = np.vstack([outer_flat, inner_flat])

    # Build hex elements (8-node bricks) connecting outer to inner
    elements = []
    for i in range(ni - 1):
        for j in range(nj - 1):
            # Outer face nodes
            n0 = i * nj + j
            n1 = (i + 1) * nj + j
            n2 = (i + 1) * nj + (j + 1)
            n3 = i * nj + (j + 1)
            # Inner face nodes (offset by n_surface)
            n4 = n0 + n_surface
            n5 = n1 + n_surface
            n6 = n2 + n_surface
            n7 = n3 + n_surface
            elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    return nodes, np.array(elements, dtype=np.int64)


def map_cfd_pressure_to_fea(
    cfd_points: NDArray[np.float64],
    cfd_pressure: NDArray[np.float64],
    fea_surface_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map CFD surface pressure data onto FEA surface nodes.

    Uses nearest-neighbor interpolation to transfer pressure from
    CFD mesh nodes to FEA mesh nodes.

    Args:
        cfd_points: (N_cfd, 3) CFD surface node coordinates.
        cfd_pressure: (N_cfd,) pressure values at CFD nodes.
        fea_surface_points: (N_fea, 3) FEA surface node coordinates.

    Returns:
        (N_fea,) pressure values interpolated to FEA nodes.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(cfd_points)
    _, indices = tree.query(fea_surface_points)
    return cfd_pressure[indices]


def identify_root_nodes(
    nodes: NDArray[np.float64],
    axis: int = 2,
    tolerance: float = 1e-4,
) -> list[int]:
    """Identify blade root nodes for fixed boundary condition.

    Finds nodes at the minimum position along the specified axis
    (typically the hub/root of the blade).

    Args:
        nodes: (N, 3) node coordinates.
        axis: 0=x, 1=y, 2=z. Axis along which root is at minimum.
        tolerance: Distance tolerance from minimum to include.

    Returns:
        List of 0-indexed node IDs at the root.
    """
    min_val = nodes[:, axis].min()
    root_mask = np.abs(nodes[:, axis] - min_val) < tolerance
    return list(np.where(root_mask)[0])


def export_fea_mesh_abaqus(
    filepath: str | Path,
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    element_type: str = "C3D8",
) -> None:
    """Export a mesh in Abaqus/CalculiX .inp format (mesh only, no analysis).

    Can be imported into CalculiX, Abaqus, or other Abaqus-compatible tools.
    """
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        f.write("*HEADING\nAstraTurbo FEA Mesh\n")
        f.write("*NODE\n")
        for i, n in enumerate(nodes):
            f.write(f"{i+1}, {n[0]:.10e}, {n[1]:.10e}, {n[2]:.10e}\n")
        f.write(f"*ELEMENT, TYPE={element_type}, ELSET=ALL\n")
        for i, e in enumerate(elements):
            ids = ", ".join(str(x + 1) for x in e)
            f.write(f"{i+1}, {ids}\n")
