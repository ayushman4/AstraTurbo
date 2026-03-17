"""CalculiX input file generation for turbomachinery blades.

Generates .inp files for CalculiX (open-source FEA solver, Abaqus-compatible)
for structural analysis of blades under centrifugal and aerodynamic loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .material import Material, INCONEL_718


def write_calculix_input(
    filepath: str | Path,
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    material: Material | None = None,
    omega: float = 0.0,
    rotation_axis: tuple[float, float, float] = (0, 0, 1),
    pressure_loads: dict[str, NDArray[np.float64]] | None = None,
    fixed_nodes: list[int] | None = None,
    element_type: str = "C3D10",
    analysis_type: str = "static",
) -> None:
    """Write a CalculiX/Abaqus input file for blade structural analysis.

    Args:
        filepath: Output .inp file path.
        nodes: (N, 3) node coordinates.
        elements: (M, nodes_per_element) element connectivity (0-indexed).
        material: Material properties (default: Inconel 718).
        omega: Angular velocity (rad/s) for centrifugal load.
        rotation_axis: Rotation axis direction.
        pressure_loads: Dict mapping surface set name to (N_faces, 1) pressure array.
        fixed_nodes: List of node IDs to fix (root of blade).
        element_type: CalculiX element type (C3D10 = 10-node tet, C3D8 = 8-node hex).
        analysis_type: 'static', 'frequency', or 'buckle'.
    """
    filepath = Path(filepath)
    if material is None:
        material = INCONEL_718

    with open(filepath, "w") as f:
        f.write("** AstraTurbo CalculiX Input File\n")
        f.write(f"** Blade structural analysis ({analysis_type})\n")
        f.write(f"** Material: {material.name}\n")
        f.write(f"** Omega: {omega:.2f} rad/s\n\n")

        # Heading
        f.write("*HEADING\n")
        f.write("AstraTurbo Blade Analysis\n\n")

        # Nodes
        f.write("*NODE\n")
        for i, node in enumerate(nodes):
            f.write(f"{i + 1}, {node[0]:.10e}, {node[1]:.10e}, {node[2]:.10e}\n")
        f.write("\n")

        # Elements
        f.write(f"*ELEMENT, TYPE={element_type}, ELSET=BLADE\n")
        for i, elem in enumerate(elements):
            node_ids = ", ".join(str(n + 1) for n in elem)  # 1-indexed
            f.write(f"{i + 1}, {node_ids}\n")
        f.write("\n")

        # Node sets
        f.write("*NSET, NSET=ALL_NODES, GENERATE\n")
        f.write(f"1, {len(nodes)}, 1\n\n")

        if fixed_nodes:
            f.write("*NSET, NSET=FIXED_ROOT\n")
            for i, nid in enumerate(fixed_nodes):
                f.write(f"{nid + 1}")
                if (i + 1) % 10 == 0:
                    f.write("\n")
                else:
                    f.write(", ")
            f.write("\n\n")

        # Material
        f.write(material.to_calculix_format())
        f.write("\n\n")

        # Section assignment
        f.write("*SOLID SECTION, ELSET=BLADE, MATERIAL=" + material.name + "\n\n")

        # Analysis step
        if analysis_type == "static":
            f.write("*STEP\n")
            f.write("*STATIC\n\n")
        elif analysis_type == "frequency":
            f.write("*STEP\n")
            f.write("*FREQUENCY\n")
            f.write("20\n\n")  # First 20 modes
        elif analysis_type == "buckle":
            f.write("*STEP\n")
            f.write("*BUCKLE\n")
            f.write("5\n\n")  # First 5 buckling modes

        # Boundary conditions (fixed root)
        if fixed_nodes:
            f.write("*BOUNDARY\n")
            f.write("FIXED_ROOT, 1, 6, 0.0\n\n")

        # Centrifugal load
        if abs(omega) > 1e-10:
            f.write("*DLOAD\n")
            ax = rotation_axis
            f.write(f"BLADE, CENTRIF, {omega**2:.6e}, "
                    f"0.0, 0.0, 0.0, {ax[0]}, {ax[1]}, {ax[2]}\n\n")

        # Pressure loads from CFD
        if pressure_loads:
            for surface_name, pressures in pressure_loads.items():
                f.write(f"** Pressure load from CFD on {surface_name}\n")
                f.write("*DLOAD\n")
                for i, p in enumerate(pressures):
                    f.write(f"{i + 1}, P, {float(p):.6e}\n")
                f.write("\n")

        # Output requests
        f.write("*NODE FILE\n")
        f.write("U\n")  # Displacements
        f.write("*EL FILE\n")
        f.write("S, E\n")  # Stresses, strains
        f.write("*NODE PRINT, NSET=ALL_NODES\n")
        f.write("U\n")
        f.write("*EL PRINT, ELSET=BLADE\n")
        f.write("S\n\n")

        f.write("*END STEP\n")
