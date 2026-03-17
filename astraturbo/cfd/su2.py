"""SU2 configuration file generator for turbomachinery simulations."""

from __future__ import annotations

from pathlib import Path


def write_su2_config(
    filepath: str | Path,
    mesh_file: str = "mesh.su2",
    mach_number: float = 0.3,
    aoa: float = 0.0,
    reynolds: float = 1e6,
    n_iterations: int = 5000,
    turbulence_model: str = "SST",
) -> None:
    """Write an SU2 configuration file for a turbomachinery case.

    Args:
        filepath: Output .cfg file path.
        mesh_file: Path to the SU2 mesh file.
        mach_number: Freestream Mach number.
        aoa: Angle of attack (degrees).
        reynolds: Reynolds number.
        n_iterations: Maximum iterations.
        turbulence_model: SA or SST.
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        f.write("% AstraTurbo SU2 Configuration\n")
        f.write("%\n\n")

        f.write("% --- Problem definition ---\n")
        f.write("SOLVER= RANS\n")
        f.write("MATH_PROBLEM= DIRECT\n")
        f.write("RESTART_SOL= NO\n\n")

        f.write("% --- Freestream conditions ---\n")
        f.write(f"MACH_NUMBER= {mach_number}\n")
        f.write(f"AOA= {aoa}\n")
        f.write(f"REYNOLDS_NUMBER= {reynolds}\n")
        f.write("REYNOLDS_LENGTH= 1.0\n\n")

        f.write("% --- Turbulence model ---\n")
        f.write(f"KIND_TURB_MODEL= {turbulence_model}\n\n")

        f.write("% --- Numerical methods ---\n")
        f.write("NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES\n")
        f.write("CFL_NUMBER= 10.0\n")
        f.write("CFL_ADAPT= NO\n\n")

        f.write("% --- Linear solver ---\n")
        f.write("LINEAR_SOLVER= FGMRES\n")
        f.write("LINEAR_SOLVER_PREC= ILU\n")
        f.write("LINEAR_SOLVER_ERROR= 1e-6\n")
        f.write("LINEAR_SOLVER_ITER= 10\n\n")

        f.write("% --- Convergence ---\n")
        f.write(f"ITER= {n_iterations}\n")
        f.write("CONV_RESIDUAL_MINVAL= -8\n")
        f.write("CONV_STARTITER= 10\n\n")

        f.write("% --- Input/output ---\n")
        f.write(f"MESH_FILENAME= {mesh_file}\n")
        f.write("MESH_FORMAT= SU2\n")
        f.write("SOLUTION_FILENAME= solution_flow.dat\n")
        f.write("RESTART_FILENAME= restart_flow.dat\n")
        f.write("OUTPUT_WRT_FREQ= 100\n")
        f.write("SCREEN_WRT_FREQ_INNER= 1\n\n")

        f.write("% --- Boundary markers ---\n")
        f.write("MARKER_HEATFLUX= ( blade, 0.0, hub, 0.0, shroud, 0.0 )\n")
        f.write("MARKER_FAR= ( inlet, outlet )\n")
