"""OpenFOAM case setup for turbomachinery simulations.

Generates the directory structure and configuration files needed to
run a turbomachinery CFD simulation in OpenFOAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_openfoam_case(
    case_dir: str | Path,
    mesh_format: str = "blockMesh",
    turbulence_model: str = "kOmegaSST",
    solver: str = "simpleFoam",
    inlet_velocity: float = 100.0,
    outlet_pressure: float = 101325.0,
    density: float = 1.225,
    viscosity: float = 1.8e-5,
) -> Path:
    """Create an OpenFOAM case directory with turbomachinery setup.

    Args:
        case_dir: Path for the case directory.
        mesh_format: 'blockMesh' or 'cgns' (for cgnsToFoam).
        turbulence_model: Turbulence model name.
        solver: OpenFOAM solver name.
        inlet_velocity: Inlet velocity magnitude (m/s).
        outlet_pressure: Outlet static pressure (Pa).
        density: Fluid density (kg/m^3).
        viscosity: Kinematic viscosity (m^2/s).

    Returns:
        Path to the created case directory.
    """
    case_dir = Path(case_dir)

    # Create directory structure
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)

    # Write controlDict
    _write_control_dict(case_dir / "system" / "controlDict", solver)

    # Write fvSchemes
    _write_fv_schemes(case_dir / "system" / "fvSchemes")

    # Write fvSolution
    _write_fv_solution(case_dir / "system" / "fvSolution", solver)

    # Write transportProperties
    _write_transport_properties(
        case_dir / "constant" / "transportProperties", viscosity
    )

    # Write turbulenceProperties
    _write_turbulence_properties(
        case_dir / "constant" / "turbulenceProperties", turbulence_model
    )

    # Write boundary conditions
    _write_velocity_bc(case_dir / "0" / "U", inlet_velocity)
    _write_pressure_bc(case_dir / "0" / "p", outlet_pressure)

    return case_dir


def _write_foam_header(f, cls: str, obj: str) -> None:
    f.write("FoamFile\n{\n")
    f.write("    version     2.0;\n")
    f.write("    format      ascii;\n")
    f.write(f"    class       {cls};\n")
    f.write(f'    object      {obj};\n')
    f.write("}\n\n")


def _write_control_dict(filepath: Path, solver: str) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "dictionary", "controlDict")
        f.write(f'application     {solver};\n\n')
        f.write("startFrom       startTime;\n")
        f.write("startTime       0;\n")
        f.write("stopAt          endTime;\n")
        f.write("endTime         1000;\n")
        f.write("deltaT          1;\n")
        f.write("writeControl    timeStep;\n")
        f.write("writeInterval   100;\n")
        f.write("purgeWrite      3;\n")
        f.write("writeFormat     ascii;\n")
        f.write("writePrecision  8;\n")
        f.write("writeCompression off;\n")
        f.write("timeFormat      general;\n")
        f.write("timePrecision   6;\n")
        f.write("runTimeModifiable true;\n")


def _write_fv_schemes(filepath: Path) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "dictionary", "fvSchemes")
        f.write("ddtSchemes { default steadyState; }\n")
        f.write("gradSchemes { default Gauss linear; }\n")
        f.write("divSchemes\n{\n")
        f.write("    default none;\n")
        f.write("    div(phi,U) bounded Gauss linearUpwind grad(U);\n")
        f.write("    div(phi,k) bounded Gauss upwind;\n")
        f.write("    div(phi,omega) bounded Gauss upwind;\n")
        f.write("    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n")
        f.write("}\n")
        f.write("laplacianSchemes { default Gauss linear corrected; }\n")
        f.write("interpolationSchemes { default linear; }\n")
        f.write("snGradSchemes { default corrected; }\n")


def _write_fv_solution(filepath: Path, solver: str) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "dictionary", "fvSolution")
        f.write("solvers\n{\n")
        f.write("    p { solver GAMG; tolerance 1e-06; relTol 0.1;\n")
        f.write("        smoother GaussSeidel; }\n")
        f.write("    U { solver smoothSolver; smoother GaussSeidel;\n")
        f.write("        tolerance 1e-06; relTol 0.1; }\n")
        f.write("    k { solver smoothSolver; smoother GaussSeidel;\n")
        f.write("        tolerance 1e-06; relTol 0.1; }\n")
        f.write("    omega { solver smoothSolver; smoother GaussSeidel;\n")
        f.write("        tolerance 1e-06; relTol 0.1; }\n")
        f.write("}\n\n")
        f.write("SIMPLE\n{\n")
        f.write("    nNonOrthogonalCorrectors 0;\n")
        f.write("    residualControl { p 1e-4; U 1e-4; k 1e-4; omega 1e-4; }\n")
        f.write("}\n\n")
        f.write("relaxationFactors\n{\n")
        f.write("    fields { p 0.3; }\n")
        f.write("    equations { U 0.7; k 0.7; omega 0.7; }\n")
        f.write("}\n")


def _write_transport_properties(filepath: Path, viscosity: float) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "dictionary", "transportProperties")
        f.write(f"nu              [0 2 -1 0 0 0 0] {viscosity:.6e};\n")


def _write_turbulence_properties(filepath: Path, model: str) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "dictionary", "turbulenceProperties")
        f.write("simulationType RAS;\n\n")
        f.write("RAS\n{\n")
        f.write(f"    RASModel    {model};\n")
        f.write("    turbulence  on;\n")
        f.write("    printCoeffs on;\n")
        f.write("}\n")


def _write_velocity_bc(filepath: Path, velocity: float) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "volVectorField", "U")
        f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
        f.write(f"internalField   uniform ({velocity} 0 0);\n\n")
        f.write("boundaryField\n{\n")
        f.write(f"    inlet {{ type fixedValue; value uniform ({velocity} 0 0); }}\n")
        f.write("    outlet { type zeroGradient; }\n")
        f.write("    blade { type noSlip; }\n")
        f.write("    hub { type noSlip; }\n")
        f.write("    shroud { type noSlip; }\n")
        f.write("}\n")


def _write_pressure_bc(filepath: Path, pressure: float) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "volScalarField", "p")
        f.write("dimensions      [0 2 -2 0 0 0 0];\n\n")
        f.write("internalField   uniform 0;\n\n")
        f.write("boundaryField\n{\n")
        f.write("    inlet { type zeroGradient; }\n")
        f.write("    outlet { type fixedValue; value uniform 0; }\n")
        f.write("    blade { type zeroGradient; }\n")
        f.write("    hub { type zeroGradient; }\n")
        f.write("    shroud { type zeroGradient; }\n")
        f.write("}\n")
