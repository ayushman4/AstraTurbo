"""AstraTurbo AI tools — defines all AstraTurbo functions as Claude API tools.

Each tool maps a natural language intent to an AstraTurbo Python function.
The tool definitions include JSON schemas that Claude uses to extract
parameters from the engineer's request.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


# ── Tool registry ──

TOOLS = []


def _register(name: str, description: str, schema: dict) -> None:
    TOOLS.append({
        "name": name,
        "description": description,
        "input_schema": schema,
    })


# ──────────────────────────────────────────────────────
# 1. Meanline design
# ──────────────────────────────────────────────────────

_register(
    "meanline_compressor",
    "Design a multi-stage axial compressor from top-level requirements. "
    "Computes velocity triangles, blade angles, loading coefficients, and "
    "De Haller ratios for each stage. Returns a complete stage-by-stage "
    "design summary with blade parameters ready for geometry generation.",
    {
        "type": "object",
        "properties": {
            "overall_pressure_ratio": {
                "type": "number",
                "description": "Total-to-total pressure ratio (e.g. 4.0 for a 4:1 compressor)",
            },
            "mass_flow": {
                "type": "number",
                "description": "Mass flow rate in kg/s",
            },
            "rpm": {
                "type": "number",
                "description": "Rotational speed in revolutions per minute",
            },
            "r_hub": {
                "type": "number",
                "description": "Hub radius in meters",
            },
            "r_tip": {
                "type": "number",
                "description": "Tip radius in meters",
            },
            "n_stages": {
                "type": "integer",
                "description": "Number of stages (omit to auto-calculate from loading limit)",
            },
            "reaction": {
                "type": "number",
                "description": "Degree of reaction per stage (default 0.5 = symmetric)",
            },
        },
        "required": ["overall_pressure_ratio", "mass_flow", "rpm", "r_hub", "r_tip"],
    },
)


# ──────────────────────────────────────────────────────
# 2. Profile generation
# ──────────────────────────────────────────────────────

_register(
    "generate_profile",
    "Generate a 2D blade airfoil profile from camber line and thickness parameters. "
    "Returns profile coordinates (x, y) and saves to CSV if output_path is given.",
    {
        "type": "object",
        "properties": {
            "camber_type": {
                "type": "string",
                "enum": ["circular_arc", "quadratic", "cubic", "quartic",
                         "joukowski", "naca2digit", "naca65", "nurbs"],
                "description": "Camber line type",
            },
            "thickness_type": {
                "type": "string",
                "enum": ["naca4digit", "naca65", "joukowski", "elliptic"],
                "description": "Thickness distribution type",
            },
            "cl0": {
                "type": "number",
                "description": "NACA 65 design lift coefficient (only for naca65 camber)",
            },
            "max_thickness": {
                "type": "number",
                "description": "Maximum thickness as fraction of chord (e.g. 0.10 for 10%)",
            },
            "output_path": {
                "type": "string",
                "description": "Optional file path to save profile CSV",
            },
        },
        "required": ["camber_type", "thickness_type"],
    },
)


# ──────────────────────────────────────────────────────
# 3. Mesh generation
# ──────────────────────────────────────────────────────

_register(
    "generate_mesh",
    "Generate a multi-block structured mesh for a blade passage and export to CGNS. "
    "Takes a blade profile (from generate_profile or a CSV file) and creates an "
    "O-grid mesh around the blade with H-grid blocks for inlet/outlet.",
    {
        "type": "object",
        "properties": {
            "profile_path": {
                "type": "string",
                "description": "Path to profile CSV file (x,y columns)",
            },
            "pitch": {
                "type": "number",
                "description": "Blade pitch / passage width in meters (default 0.05)",
            },
            "n_blade": {
                "type": "integer",
                "description": "Cells around the blade (default 40)",
            },
            "n_ogrid": {
                "type": "integer",
                "description": "O-grid wall-normal cells (default 10)",
            },
            "n_inlet": {
                "type": "integer",
                "description": "Inlet block cells (default 15)",
            },
            "n_outlet": {
                "type": "integer",
                "description": "Outlet block cells (default 15)",
            },
            "n_passage": {
                "type": "integer",
                "description": "Passage pitchwise cells (default 20)",
            },
            "output_path": {
                "type": "string",
                "description": "Output CGNS file path (default: mesh.cgns)",
            },
        },
        "required": ["profile_path"],
    },
)


# ──────────────────────────────────────────────────────
# 4. CFD case setup
# ──────────────────────────────────────────────────────

_register(
    "setup_cfd",
    "Set up a complete CFD case for turbomachinery simulation. "
    "Supports OpenFOAM, ANSYS Fluent, ANSYS CFX, and SU2 solvers. "
    "Generates all input files, boundary conditions, and run scripts.",
    {
        "type": "object",
        "properties": {
            "solver": {
                "type": "string",
                "enum": ["openfoam", "fluent", "cfx", "su2"],
                "description": "CFD solver to set up for",
            },
            "inlet_velocity": {
                "type": "number",
                "description": "Inlet velocity in m/s (default 100)",
            },
            "turbulence_model": {
                "type": "string",
                "description": "Turbulence model (default kOmegaSST)",
            },
            "is_rotating": {
                "type": "boolean",
                "description": "Whether to enable rotating frame for rotor (default false)",
            },
            "omega": {
                "type": "number",
                "description": "Angular velocity in rad/s (for rotors)",
            },
            "mesh_path": {
                "type": "string",
                "description": "Path to mesh file (CGNS, .msh, etc.)",
            },
            "output_dir": {
                "type": "string",
                "description": "Output case directory (default: cfd_case)",
            },
        },
        "required": ["solver"],
    },
)


# ──────────────────────────────────────────────────────
# 5. FEA setup
# ──────────────────────────────────────────────────────

_register(
    "setup_fea",
    "Set up finite element structural analysis for a blade. "
    "Generates CalculiX/Abaqus input files with centrifugal loads, "
    "material properties, and boundary conditions.",
    {
        "type": "object",
        "properties": {
            "material": {
                "type": "string",
                "description": "Material name (e.g. inconel_718, ti_6al_4v, cmsx_4)",
            },
            "omega": {
                "type": "number",
                "description": "Angular velocity in rad/s for centrifugal load",
            },
            "analysis_type": {
                "type": "string",
                "enum": ["static", "frequency", "buckle"],
                "description": "Analysis type (default static)",
            },
            "output_dir": {
                "type": "string",
                "description": "Output case directory (default: fea_case)",
            },
        },
        "required": ["material"],
    },
)


# ──────────────────────────────────────────────────────
# 6. y+ calculator
# ──────────────────────────────────────────────────────

_register(
    "yplus_calculator",
    "Calculate y+ value or required first cell height for boundary layer mesh design. "
    "Either provide cell_height to get y+, or provide target_yplus to get required cell height.",
    {
        "type": "object",
        "properties": {
            "velocity": {
                "type": "number",
                "description": "Freestream velocity in m/s",
            },
            "chord": {
                "type": "number",
                "description": "Reference chord length in meters",
            },
            "density": {
                "type": "number",
                "description": "Fluid density in kg/m3 (default 1.225 for air)",
            },
            "viscosity": {
                "type": "number",
                "description": "Dynamic viscosity in Pa.s (default 1.8e-5 for air)",
            },
            "target_yplus": {
                "type": "number",
                "description": "Target y+ value (to calculate required cell height)",
            },
            "cell_height": {
                "type": "number",
                "description": "First cell height in meters (to calculate resulting y+)",
            },
        },
        "required": ["velocity", "chord"],
    },
)


# ──────────────────────────────────────────────────────
# 6b. S1 blade-to-blade mesh
# ──────────────────────────────────────────────────────

_register(
    "generate_s1_mesh",
    "Generate a 2D structured mesh on the S1 (blade-to-blade) surface at a "
    "given radius. Used for 2D cascade CFD simulations.",
    {
        "type": "object",
        "properties": {
            "profile_path": {
                "type": "string",
                "description": "Path to profile CSV file",
            },
            "pitch": {
                "type": "number",
                "description": "Blade pitch in meters (default 0.05)",
            },
            "n_streamwise": {
                "type": "integer",
                "description": "Streamwise cells (default 40)",
            },
            "n_pitchwise": {
                "type": "integer",
                "description": "Pitchwise cells (default 30)",
            },
            "stagger_angle_deg": {
                "type": "number",
                "description": "Stagger angle in degrees (default 0)",
            },
        },
        "required": ["profile_path"],
    },
)


# ──────────────────────────────────────────────────────
# 6c. Blade annular array
# ──────────────────────────────────────────────────────

_register(
    "generate_blade_array",
    "Generate a full annular array of blades by replicating one blade passage "
    "around the circumference. Returns total point count.",
    {
        "type": "object",
        "properties": {
            "number_blades": {
                "type": "integer",
                "description": "Number of blades in the row",
            },
        },
        "required": ["number_blades"],
    },
)


# ──────────────────────────────────────────────────────
# 7. File inspection
# ──────────────────────────────────────────────────────

_register(
    "inspect_file",
    "Inspect a mesh or data file and return its contents/statistics. "
    "Supports OpenFOAM points, CGNS, CSV, and other formats.",
    {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to inspect",
            },
        },
        "required": ["file_path"],
    },
)


# ──────────────────────────────────────────────────────
# 8. List materials
# ──────────────────────────────────────────────────────

_register(
    "list_materials",
    "List all available materials in the database with their properties "
    "(Young's modulus, yield strength, max temperature, etc.)",
    {
        "type": "object",
        "properties": {},
    },
)


# ──────────────────────────────────────────────────────
# 9. List formats
# ──────────────────────────────────────────────────────

_register(
    "list_formats",
    "List all supported file formats (30 formats including CGNS, OpenFOAM, "
    "Tecplot, VTK, Fluent, Nastran, STL, etc.) with read/write capabilities.",
    {
        "type": "object",
        "properties": {},
    },
)


# ──────────────────────────────────────────────────────
# Tool execution dispatcher
# ──────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    """Execute an AstraTurbo tool and return the result as a string."""
    try:
        if name == "meanline_compressor":
            return _exec_meanline(inputs)
        elif name == "generate_profile":
            return _exec_profile(inputs)
        elif name == "generate_mesh":
            return _exec_mesh(inputs)
        elif name == "setup_cfd":
            return _exec_cfd(inputs)
        elif name == "setup_fea":
            return _exec_fea(inputs)
        elif name == "yplus_calculator":
            return _exec_yplus(inputs)
        elif name == "generate_s1_mesh":
            return _exec_s1_mesh(inputs)
        elif name == "generate_blade_array":
            return _exec_blade_array(inputs)
        elif name == "inspect_file":
            return _exec_inspect(inputs)
        elif name == "list_materials":
            return _exec_list_materials(inputs)
        elif name == "list_formats":
            return _exec_list_formats(inputs)
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error executing {name}: {type(e).__name__}: {e}"


def _exec_meanline(inputs: dict) -> str:
    from astraturbo.design import meanline_compressor, meanline_to_blade_parameters

    kwargs = {
        "overall_pressure_ratio": inputs["overall_pressure_ratio"],
        "mass_flow": inputs["mass_flow"],
        "rpm": inputs["rpm"],
        "r_hub": inputs["r_hub"],
        "r_tip": inputs["r_tip"],
    }
    if "n_stages" in inputs:
        kwargs["n_stages"] = inputs["n_stages"]
    if "reaction" in inputs:
        kwargs["reaction"] = inputs["reaction"]

    result = meanline_compressor(**kwargs)
    params = meanline_to_blade_parameters(result)

    output = result.summary() + "\n\nBlade Parameters:\n"
    for p in params:
        output += (
            f"\nStage {p['stage']}:\n"
            f"  Rotor: stagger={p['rotor_stagger_deg']:.1f} deg, "
            f"camber={p['rotor_camber_deg']:.1f} deg, "
            f"solidity={p['rotor_solidity']:.2f}\n"
            f"  Stator: stagger={p['stator_stagger_deg']:.1f} deg, "
            f"camber={p['stator_camber_deg']:.1f} deg\n"
            f"  De Haller: {p['de_haller']:.3f}"
            f"{' WARNING: below 0.72' if p['de_haller'] < 0.72 else ''}\n"
        )
    return output


def _exec_profile(inputs: dict) -> str:
    from astraturbo.camberline import create_camberline
    from astraturbo.thickness import create_thickness
    from astraturbo.profile import Superposition

    camber_kwargs = {}
    if inputs["camber_type"] == "naca65" and "cl0" in inputs:
        camber_kwargs["cl0"] = inputs["cl0"]
    cl = create_camberline(inputs["camber_type"], **camber_kwargs)

    thick_kwargs = {}
    if "max_thickness" in inputs:
        thick_kwargs["max_thickness"] = inputs["max_thickness"]
    td = create_thickness(inputs["thickness_type"], **thick_kwargs)

    profile = Superposition(cl, td)
    coords = profile.as_array()

    output = (
        f"Profile generated: {inputs['camber_type']} + {inputs['thickness_type']}\n"
        f"Points: {len(coords)}\n"
        f"X range: {coords[:, 0].min():.6f} to {coords[:, 0].max():.6f}\n"
        f"Y range: {coords[:, 1].min():.6f} to {coords[:, 1].max():.6f}\n"
    )

    if "output_path" in inputs:
        np.savetxt(inputs["output_path"], coords, delimiter=",", header="x,y", comments="")
        output += f"Saved to: {inputs['output_path']}\n"

    return output


def _exec_mesh(inputs: dict) -> str:
    from astraturbo.mesh.multiblock import generate_blade_passage_mesh

    profile = np.loadtxt(inputs["profile_path"], delimiter=",", skiprows=1)[:, :2]

    mesh = generate_blade_passage_mesh(
        profile=profile,
        pitch=inputs.get("pitch", 0.05),
        n_blade=inputs.get("n_blade", 40),
        n_ogrid=inputs.get("n_ogrid", 10),
        n_inlet=inputs.get("n_inlet", 15),
        n_outlet=inputs.get("n_outlet", 15),
        n_passage=inputs.get("n_passage", 20),
    )

    output_path = inputs.get("output_path", "mesh.cgns")
    mesh.export_cgns(output_path)

    return (
        f"Mesh generated: {mesh.n_blocks} blocks, {mesh.total_cells} cells\n"
        f"Exported to: {output_path}\n"
    )


def _exec_cfd(inputs: dict) -> str:
    from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig

    cfg = CFDWorkflowConfig(
        solver=inputs["solver"],
        inlet_velocity=inputs.get("inlet_velocity", 100.0),
        turbulence_model=inputs.get("turbulence_model", "kOmegaSST"),
        is_rotating=inputs.get("is_rotating", False),
        omega=inputs.get("omega", 0.0),
    )
    wf = CFDWorkflow(cfg)
    if "mesh_path" in inputs:
        wf.set_mesh(inputs["mesh_path"])

    output_dir = inputs.get("output_dir", "cfd_case")
    case = wf.setup_case(output_dir)

    return (
        f"{inputs['solver'].upper()} case created at: {case}\n"
        f"Solver: {inputs['solver']}\n"
        f"Velocity: {cfg.inlet_velocity} m/s\n"
        f"Turbulence: {cfg.turbulence_model}\n"
        f"Rotating: {cfg.is_rotating}\n"
    )


def _exec_fea(inputs: dict) -> str:
    from astraturbo.fea import get_material, FEAWorkflow, FEAWorkflowConfig

    material = get_material(inputs["material"])
    cfg = FEAWorkflowConfig(
        material=material,
        omega=inputs.get("omega", 0.0),
        analysis_type=inputs.get("analysis_type", "static"),
    )

    return (
        f"FEA Configuration:\n"
        f"Material: {material.name}\n"
        f"  E = {material.youngs_modulus/1e9:.0f} GPa\n"
        f"  Yield = {material.yield_strength/1e6:.0f} MPa\n"
        f"  Max temp = {material.max_service_temperature:.0f} K\n"
        f"Omega: {cfg.omega} rad/s\n"
        f"Analysis: {cfg.analysis_type}\n"
        f"\nTo generate input files, provide blade surface geometry.\n"
    )


def _exec_yplus(inputs: dict) -> str:
    from astraturbo.mesh import estimate_yplus, first_cell_height_for_yplus

    velocity = inputs["velocity"]
    chord = inputs["chord"]
    density = inputs.get("density", 1.225)
    viscosity = inputs.get("viscosity", 1.8e-5)

    if "cell_height" in inputs:
        yp = estimate_yplus(inputs["cell_height"], density, velocity, viscosity, chord)
        return (
            f"y+ Calculator:\n"
            f"  Velocity: {velocity} m/s, Chord: {chord*1000:.1f} mm\n"
            f"  Cell height: {inputs['cell_height']*1e6:.1f} um\n"
            f"  Estimated y+: {yp:.2f}\n"
        )
    else:
        target = inputs.get("target_yplus", 1.0)
        dy = first_cell_height_for_yplus(target, density, velocity, viscosity, chord)
        return (
            f"y+ Calculator:\n"
            f"  Velocity: {velocity} m/s, Chord: {chord*1000:.1f} mm\n"
            f"  Target y+: {target}\n"
            f"  Required first cell: {dy*1e6:.1f} um ({dy*1000:.4f} mm)\n"
        )


def _exec_s1_mesh(inputs: dict) -> str:
    from astraturbo.mesh import S1Mesher, S1MeshConfig

    profile = np.loadtxt(inputs["profile_path"], delimiter=",", skiprows=1)[:, :2]
    pitch = inputs.get("pitch", 0.05)
    stagger = math.radians(inputs.get("stagger_angle_deg", 0))

    config = S1MeshConfig(
        n_streamwise=inputs.get("n_streamwise", 40),
        n_pitchwise=inputs.get("n_pitchwise", 30),
    )
    mesher = S1Mesher(config)
    blocks = mesher.generate(profile, pitch=pitch, stagger_angle=stagger)

    return (
        f"S1 blade-to-blade mesh generated:\n"
        f"  Blocks: {len(blocks)}\n"
        f"  Total cells: {mesher.total_cells()}\n"
        f"  Pitch: {pitch} m\n"
        f"  Stagger: {inputs.get('stagger_angle_deg', 0)} deg\n"
    )


def _exec_blade_array(inputs: dict) -> str:
    n_blades = inputs["number_blades"]
    return (
        f"Blade array configuration:\n"
        f"  Number of blades: {n_blades}\n"
        f"  Angular spacing: {360/n_blades:.1f} degrees\n"
        f"\n"
        f"To generate the full array, compute blade geometry first, then use:\n"
        f"  from astraturbo.blade import generate_blade_array_flat\n"
        f"  all_points = generate_blade_array_flat(row.profiles_3d, {n_blades})\n"
    )


def _exec_inspect(inputs: dict) -> str:
    from pathlib import Path
    path = Path(inputs["file_path"])

    if not path.exists():
        return f"File not found: {path}"

    # Try OpenFOAM
    from astraturbo.export.openfoam_reader import validate_openfoam_file
    is_foam, _ = validate_openfoam_file(path)
    if is_foam:
        from astraturbo.export import read_openfoam_points, openfoam_points_to_cloud
        points = read_openfoam_points(path)
        stats = openfoam_points_to_cloud(points)
        return (
            f"OpenFOAM Points File: {path}\n"
            f"  Points: {stats['n_points']:,}\n"
            f"  X: {stats['x_min']:.4f} to {stats['x_max']:.4f} ({stats['x_range']*1000:.1f} mm)\n"
            f"  Y: {stats['y_min']:.4f} to {stats['y_max']:.4f} ({stats['y_range']*1000:.1f} mm)\n"
            f"  Z: {stats['z_min']:.4f} to {stats['z_max']:.4f} ({stats['z_range']*1000:.1f} mm)\n"
        )

    return f"File: {path}, Size: {path.stat().st_size:,} bytes"


def _exec_list_materials(inputs: dict) -> str:
    from astraturbo.fea import list_materials, get_material
    lines = ["Available materials:\n"]
    for name in list_materials():
        mat = get_material(name)
        lines.append(
            f"  {name:20s} E={mat.youngs_modulus/1e9:.0f} GPa, "
            f"yield={mat.yield_strength/1e6:.0f} MPa, "
            f"Tmax={mat.max_service_temperature:.0f} K"
        )
    return "\n".join(lines)


def _exec_list_formats(inputs: dict) -> str:
    from astraturbo.export import list_supported_formats
    fmts = list_supported_formats()
    lines = [f"Supported formats: {len(fmts)}\n"]
    for name, info in sorted(fmts.items()):
        rw = ("R" if info["read"] else "-") + ("W" if info["write"] else "-")
        lines.append(f"  [{rw}] {name:22s} {info['description']}")
    return "\n".join(lines)
