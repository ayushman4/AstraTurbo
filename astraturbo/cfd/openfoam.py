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
    patch_names: dict[str, str] | None = None,
    compressible: bool = False,
    total_pressure: float = 101325.0,
    total_temperature: float = 288.15,
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
        patch_names: Mapping of logical names to actual mesh patch names.
        compressible: If True, write compressible BCs and thermophysical properties.
        total_pressure: Total pressure at inlet (Pa), used when compressible=True.
        total_temperature: Total temperature at inlet (K), used when compressible=True.

    Returns:
        Path to the created case directory.
    """
    case_dir = Path(case_dir)

    # Security: validate solver and turbulence model names
    _ALLOWED_SOLVERS = {
        "simpleFoam", "pimpleFoam", "rhoSimpleFoam", "rhoPimpleFoam",
        "sonicFoam", "compressibleInterFoam",
    }
    _ALLOWED_TURB = {
        "kOmegaSST", "kEpsilon", "SpalartAllmaras", "kOmega",
        "realizableKE", "LamBremhorstKE", "laminar",
    }
    if solver not in _ALLOWED_SOLVERS:
        raise ValueError(f"Invalid solver '{solver}'. Allowed: {_ALLOWED_SOLVERS}")
    if turbulence_model not in _ALLOWED_TURB:
        raise ValueError(f"Invalid turbulence model '{turbulence_model}'. Allowed: {_ALLOWED_TURB}")

    # Create directory structure
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)

    # Default patch name mapping
    if patch_names is None:
        patch_names = {
            "inlet": "inlet",
            "outlet": "outlet",
            "blade": "blade",
            "hub": "hub",
            "shroud": "shroud",
            "periodic_upper": "periodic_upper",
            "periodic_lower": "periodic_lower",
        }

    # Write controlDict
    _write_control_dict(case_dir / "system" / "controlDict", solver)

    # Write fvSchemes
    _write_fv_schemes(case_dir / "system" / "fvSchemes", compressible=compressible)

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

    # Write boundary conditions (compressible vs incompressible)
    if compressible:
        _write_thermophysical_properties(case_dir / "constant" / "thermophysicalProperties")
        _write_compressible_velocity_bc(
            case_dir / "0" / "U", inlet_velocity, patch_names
        )
        _write_compressible_pressure_bc(
            case_dir / "0" / "p", total_pressure, outlet_pressure, patch_names
        )
        _write_temperature_bc(
            case_dir / "0" / "T", total_temperature, patch_names
        )
    else:
        _write_velocity_bc(case_dir / "0" / "U", inlet_velocity, patch_names)
        _write_pressure_bc(case_dir / "0" / "p", outlet_pressure, patch_names)

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


def _write_fv_schemes(filepath: Path, compressible: bool = False) -> None:
    with open(filepath, "w") as f:
        _write_foam_header(f, "dictionary", "fvSchemes")
        f.write("ddtSchemes { default steadyState; }\n")
        f.write("gradSchemes { default Gauss linear; }\n")
        f.write("divSchemes\n{\n")
        f.write("    default none;\n")
        f.write("    div(phi,U) bounded Gauss linearUpwind grad(U);\n")
        f.write("    div(phi,k) bounded Gauss upwind;\n")
        f.write("    div(phi,omega) bounded Gauss upwind;\n")
        if compressible:
            f.write("    div(phi,e) bounded Gauss upwind;\n")
            f.write("    div(phi,K) bounded Gauss upwind;\n")
            f.write("    div(phid,p) Gauss upwind;\n")
            f.write("    div(phi,Ekp) bounded Gauss upwind;\n")
            f.write("    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n")
        else:
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


def _write_velocity_bc(
    filepath: Path, velocity: float, patch_names: dict[str, str] | None = None,
) -> None:
    if patch_names is None:
        patch_names = {"inlet": "inlet", "outlet": "outlet", "blade": "blade",
                       "hub": "hub", "shroud": "shroud"}
    with open(filepath, "w") as f:
        _write_foam_header(f, "volVectorField", "U")
        f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
        f.write(f"internalField   uniform ({velocity} 0 0);\n\n")
        f.write("boundaryField\n{\n")
        f.write(f"    {patch_names.get('inlet', 'inlet')} {{ type fixedValue; value uniform ({velocity} 0 0); }}\n")
        f.write(f"    {patch_names.get('outlet', 'outlet')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('blade', 'blade')} {{ type noSlip; }}\n")
        f.write(f"    {patch_names.get('hub', 'hub')} {{ type noSlip; }}\n")
        f.write(f"    {patch_names.get('shroud', 'shroud')} {{ type noSlip; }}\n")
        f.write("}\n")


def _write_pressure_bc(
    filepath: Path, pressure: float, patch_names: dict[str, str] | None = None,
) -> None:
    if patch_names is None:
        patch_names = {"inlet": "inlet", "outlet": "outlet", "blade": "blade",
                       "hub": "hub", "shroud": "shroud"}
    with open(filepath, "w") as f:
        _write_foam_header(f, "volScalarField", "p")
        f.write("dimensions      [0 2 -2 0 0 0 0];\n\n")
        f.write("internalField   uniform 0;\n\n")
        f.write("boundaryField\n{\n")
        f.write(f"    {patch_names.get('inlet', 'inlet')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('outlet', 'outlet')} {{ type fixedValue; value uniform 0; }}\n")
        f.write(f"    {patch_names.get('blade', 'blade')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('hub', 'hub')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('shroud', 'shroud')} {{ type zeroGradient; }}\n")
        f.write("}\n")


# ── Compressible BC writers ──────────────────────────────────

def _write_thermophysical_properties(filepath: Path) -> None:
    """Write thermophysicalProperties for ideal gas (perfect gas, Sutherland)."""
    with open(filepath, "w") as f:
        _write_foam_header(f, "dictionary", "thermophysicalProperties")
        f.write("thermoType\n{\n")
        f.write("    type            hePsiThermo;\n")
        f.write("    mixture         pureMixture;\n")
        f.write("    transport       sutherland;\n")
        f.write("    thermo          hConst;\n")
        f.write("    equationOfState perfectGas;\n")
        f.write("    specie          specie;\n")
        f.write("    energy          sensibleInternalEnergy;\n")
        f.write("}\n\n")
        f.write("mixture\n{\n")
        f.write("    specie { molWeight 28.96; }\n")
        f.write("    thermodynamics { Cp 1005; Hf 0; }\n")
        f.write("    transport { As 1.458e-06; Ts 110.4; }\n")
        f.write("}\n")


def _write_compressible_velocity_bc(
    filepath: Path, velocity: float, patch_names: dict[str, str] | None = None,
) -> None:
    """Write velocity BCs for compressible solver (pressureInletOutletVelocity at inlet)."""
    if patch_names is None:
        patch_names = {"inlet": "inlet", "outlet": "outlet", "blade": "blade",
                       "hub": "hub", "shroud": "shroud"}
    with open(filepath, "w") as f:
        _write_foam_header(f, "volVectorField", "U")
        f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
        f.write(f"internalField   uniform ({velocity} 0 0);\n\n")
        f.write("boundaryField\n{\n")
        f.write(f"    {patch_names.get('inlet', 'inlet')}\n    {{\n")
        f.write(f"        type    pressureInletOutletVelocity;\n")
        f.write(f"        value   uniform ({velocity} 0 0);\n    }}\n")
        f.write(f"    {patch_names.get('outlet', 'outlet')}\n    {{\n")
        f.write(f"        type    inletOutlet;\n")
        f.write(f"        inletValue uniform (0 0 0);\n")
        f.write(f"        value   uniform ({velocity} 0 0);\n    }}\n")
        f.write(f"    {patch_names.get('blade', 'blade')} {{ type noSlip; }}\n")
        f.write(f"    {patch_names.get('hub', 'hub')} {{ type noSlip; }}\n")
        f.write(f"    {patch_names.get('shroud', 'shroud')} {{ type noSlip; }}\n")
        f.write("}\n")


def _write_compressible_pressure_bc(
    filepath: Path,
    total_pressure: float,
    outlet_pressure: float,
    patch_names: dict[str, str] | None = None,
) -> None:
    """Write pressure BCs for compressible solver (totalPressure at inlet, fixedValue at outlet)."""
    if patch_names is None:
        patch_names = {"inlet": "inlet", "outlet": "outlet", "blade": "blade",
                       "hub": "hub", "shroud": "shroud"}
    with open(filepath, "w") as f:
        _write_foam_header(f, "volScalarField", "p")
        f.write("dimensions      [1 -1 -2 0 0 0 0];\n\n")
        f.write(f"internalField   uniform {total_pressure};\n\n")
        f.write("boundaryField\n{\n")
        f.write(f"    {patch_names.get('inlet', 'inlet')}\n    {{\n")
        f.write(f"        type        totalPressure;\n")
        f.write(f"        p0          uniform {total_pressure};\n")
        f.write(f"        gamma       1.4;\n")
        f.write(f"        value       uniform {total_pressure};\n    }}\n")
        f.write(f"    {patch_names.get('outlet', 'outlet')}\n    {{\n")
        f.write(f"        type        fixedValue;\n")
        f.write(f"        value       uniform {outlet_pressure};\n    }}\n")
        f.write(f"    {patch_names.get('blade', 'blade')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('hub', 'hub')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('shroud', 'shroud')} {{ type zeroGradient; }}\n")
        f.write("}\n")


def _write_temperature_bc(
    filepath: Path,
    total_temperature: float,
    patch_names: dict[str, str] | None = None,
) -> None:
    """Write temperature BCs (totalTemperature at inlet)."""
    if patch_names is None:
        patch_names = {"inlet": "inlet", "outlet": "outlet", "blade": "blade",
                       "hub": "hub", "shroud": "shroud"}
    with open(filepath, "w") as f:
        _write_foam_header(f, "volScalarField", "T")
        f.write("dimensions      [0 0 0 1 0 0 0];\n\n")
        f.write(f"internalField   uniform {total_temperature};\n\n")
        f.write("boundaryField\n{\n")
        f.write(f"    {patch_names.get('inlet', 'inlet')}\n    {{\n")
        f.write(f"        type        totalTemperature;\n")
        f.write(f"        T0          uniform {total_temperature};\n")
        f.write(f"        gamma       1.4;\n")
        f.write(f"        value       uniform {total_temperature};\n    }}\n")
        f.write(f"    {patch_names.get('outlet', 'outlet')}\n    {{\n")
        f.write(f"        type        inletOutlet;\n")
        f.write(f"        inletValue  uniform {total_temperature};\n")
        f.write(f"        value       uniform {total_temperature};\n    }}\n")
        f.write(f"    {patch_names.get('blade', 'blade')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('hub', 'hub')} {{ type zeroGradient; }}\n")
        f.write(f"    {patch_names.get('shroud', 'shroud')} {{ type zeroGradient; }}\n")
        f.write("}\n")
