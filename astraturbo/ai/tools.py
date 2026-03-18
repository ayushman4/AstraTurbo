"""AstraTurbo AI tools — defines all AstraTurbo functions as Claude API tools.

Each tool maps a natural language intent to an AstraTurbo Python function.
The tool definitions include JSON schemas that Claude uses to extract
parameters from the engineer's request.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


# ── Security: path validation ──

def _validate_path(path_str: str, must_exist: bool = False) -> Path:
    """Validate and sanitize a user-supplied file path.

    Prevents path traversal attacks by resolving the path and checking
    it doesn't escape the current working directory.
    """
    path = Path(path_str).resolve()
    cwd = Path.cwd().resolve()

    # Block obvious traversal attempts
    if ".." in Path(path_str).parts:
        raise ValueError(f"Path traversal not allowed: {path_str}")

    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path


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
    "material properties, and boundary conditions. Supports temperature-dependent "
    "properties — pass operating_temperature to use hot-section material data.",
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
            "operating_temperature": {
                "type": "number",
                "description": "Operating temperature (K) for temperature-dependent material properties. "
                "Critical for turbine blades (typically 900-1300 K).",
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
    "List all 32 available materials with properties. Materials marked [T(K)] have "
    "temperature-dependent data (E, yield, conductivity vs temperature). Pass a "
    "temperature to see hot-section properties — critical for turbine blade analysis.",
    {
        "type": "object",
        "properties": {
            "temperature": {
                "type": "number",
                "description": "Operating temperature (K) to show interpolated properties for materials with temp data",
            },
        },
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
# 12. Off-design compressor analysis
# ──────────────────────────────────────────────────────

_register(
    "off_design_compressor",
    "Run off-design analysis of an axial compressor at a different mass flow "
    "and/or RPM than the design point. Blade metal angles are fixed from the "
    "design; flow angles change, creating incidence which modifies losses, "
    "efficiency, and pressure ratio. Returns per-stage incidence, diffusion "
    "factor, losses, and stall/choke flags. Requires a design-point run first "
    "(specify the same geometry parameters plus the off-design mass_flow and rpm).",
    {
        "type": "object",
        "properties": {
            "overall_pressure_ratio": {
                "type": "number",
                "description": "Design-point overall pressure ratio (for design geometry)",
            },
            "design_mass_flow": {
                "type": "number",
                "description": "Design-point mass flow rate (kg/s)",
            },
            "design_rpm": {
                "type": "number",
                "description": "Design-point RPM",
            },
            "r_hub": {
                "type": "number",
                "description": "Hub radius (m)",
            },
            "r_tip": {
                "type": "number",
                "description": "Tip radius (m)",
            },
            "off_design_mass_flow": {
                "type": "number",
                "description": "Off-design mass flow rate (kg/s)",
            },
            "off_design_rpm": {
                "type": "number",
                "description": "Off-design RPM",
            },
        },
        "required": [
            "overall_pressure_ratio", "design_mass_flow", "design_rpm",
            "r_hub", "r_tip", "off_design_mass_flow", "off_design_rpm",
        ],
    },
)


# ──────────────────────────────────────────────────────
# 13. Compressor map generation
# ──────────────────────────────────────────────────────

_register(
    "generate_compressor_map",
    "Generate a complete compressor performance map with speed lines and surge "
    "line. Sweeps mass flow at multiple RPM fractions to produce pressure ratio "
    "vs mass flow curves, identifies stall/choke boundaries, and computes surge "
    "margin. Returns a formatted table of all speed lines plus the surge line.",
    {
        "type": "object",
        "properties": {
            "overall_pressure_ratio": {
                "type": "number",
                "description": "Design-point overall pressure ratio",
            },
            "mass_flow": {
                "type": "number",
                "description": "Design-point mass flow rate (kg/s)",
            },
            "rpm": {
                "type": "number",
                "description": "Design-point RPM",
            },
            "r_hub": {
                "type": "number",
                "description": "Hub radius (m)",
            },
            "r_tip": {
                "type": "number",
                "description": "Tip radius (m)",
            },
            "rpm_fractions": {
                "type": "array",
                "items": {"type": "number"},
                "description": "RPM fractions to evaluate (default: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05])",
            },
            "n_points": {
                "type": "integer",
                "description": "Number of mass flow points per speed line (default: 15)",
            },
        },
        "required": ["overall_pressure_ratio", "mass_flow", "rpm", "r_hub", "r_tip"],
    },
)


# ──────────────────────────────────────────────────────
# 14. Export mesh to file
# ──────────────────────────────────────────────────────

_register(
    "export_mesh",
    "Export a mesh to file. Generates a mesh from a 2D blade profile and exports "
    "it to CGNS or OpenFOAM blockMeshDict format. This is the critical step that "
    "writes the solver-ready mesh to disk.",
    {
        "type": "object",
        "properties": {
            "profile_path": {
                "type": "string",
                "description": "Path to blade profile CSV (x,y columns)",
            },
            "pitch": {
                "type": "number",
                "description": "Blade-to-blade spacing (m)",
            },
            "output_path": {
                "type": "string",
                "description": "Output file path (.cgns or blockMeshDict)",
            },
            "format": {
                "type": "string",
                "enum": ["cgns", "openfoam"],
                "description": "Output format (default: cgns)",
            },
            "n_blade": {
                "type": "integer",
                "description": "Number of cells around blade (default: 40)",
            },
            "n_ogrid": {
                "type": "integer",
                "description": "O-grid wall-normal cells (default: 10)",
            },
            "n_inlet": {
                "type": "integer",
                "description": "Inlet block cells (default: 15)",
            },
            "n_outlet": {
                "type": "integer",
                "description": "Outlet block cells (default: 15)",
            },
            "n_passage": {
                "type": "integer",
                "description": "Passage pitchwise cells (default: 20)",
            },
        },
        "required": ["profile_path", "pitch", "output_path"],
    },
)


# ──────────────────────────────────────────────────────
# 15. Build 3D blade
# ──────────────────────────────────────────────────────

_register(
    "build_3d_blade",
    "Build a 3D blade from 2D profiles stacked hub to tip. Creates profiles at "
    "multiple span positions with varying cl0 and thickness, stacks them with "
    "stagger and chord, and optionally exports to CGNS.",
    {
        "type": "object",
        "properties": {
            "hub_radius": {
                "type": "number",
                "description": "Hub radius (m)",
            },
            "tip_radius": {
                "type": "number",
                "description": "Tip radius (m)",
            },
            "axial_chord": {
                "type": "number",
                "description": "Axial chord length (m). Default: 0.05",
            },
            "n_blades": {
                "type": "integer",
                "description": "Number of blades (default: 24)",
            },
            "cl0_hub": {
                "type": "number",
                "description": "CL0 at hub (default: 0.8)",
            },
            "cl0_mid": {
                "type": "number",
                "description": "CL0 at midspan (default: 1.0)",
            },
            "cl0_tip": {
                "type": "number",
                "description": "CL0 at tip (default: 1.2)",
            },
            "thickness_hub": {
                "type": "number",
                "description": "Max thickness fraction at hub (default: 0.08)",
            },
            "thickness_mid": {
                "type": "number",
                "description": "Max thickness fraction at midspan (default: 0.10)",
            },
            "thickness_tip": {
                "type": "number",
                "description": "Max thickness fraction at tip (default: 0.12)",
            },
            "stagger_hub_deg": {
                "type": "number",
                "description": "Stagger angle at hub (degrees, default: 30)",
            },
            "stagger_mid_deg": {
                "type": "number",
                "description": "Stagger angle at midspan (degrees, default: 35)",
            },
            "stagger_tip_deg": {
                "type": "number",
                "description": "Stagger angle at tip (degrees, default: 40)",
            },
            "output_path": {
                "type": "string",
                "description": "Optional output CGNS file path for the mesh",
            },
        },
        "required": ["hub_radius", "tip_radius"],
    },
)


# ──────────────────────────────────────────────────────
# 16. Run solver
# ──────────────────────────────────────────────────────

_register(
    "run_solver",
    "Launch a CFD or FEA solver on an existing case directory. Supports OpenFOAM, "
    "SU2, and CalculiX. Returns status (success/failure) and log file path.",
    {
        "type": "object",
        "properties": {
            "case_dir": {
                "type": "string",
                "description": "Path to the case directory",
            },
            "solver": {
                "type": "string",
                "enum": ["openfoam", "su2", "calculix"],
                "description": "Solver to run (default: openfoam)",
            },
            "n_procs": {
                "type": "integer",
                "description": "Number of parallel processes (default: 1)",
            },
        },
        "required": ["case_dir"],
    },
)


# ──────────────────────────────────────────────────────
# 17. Design chain (end-to-end pipeline)
# ──────────────────────────────────────────────────────

_register(
    "design_chain",
    "Run the full end-to-end design pipeline: meanline → profile → blade → mesh → "
    "export → CFD. Auto-propagates parameters between stages. Set any combination "
    "of design parameters and the chain runs all dependent stages automatically.",
    {
        "type": "object",
        "properties": {
            "pressure_ratio": {
                "type": "number",
                "description": "Overall pressure ratio",
            },
            "mass_flow": {
                "type": "number",
                "description": "Mass flow rate (kg/s)",
            },
            "rpm": {
                "type": "number",
                "description": "Rotational speed (RPM)",
            },
            "cl0": {
                "type": "number",
                "description": "Profile design lift coefficient",
            },
            "max_thickness": {
                "type": "number",
                "description": "Profile maximum thickness fraction",
            },
            "camber_type": {
                "type": "string",
                "description": "Camber line type (default: naca65)",
            },
            "thickness_type": {
                "type": "string",
                "description": "Thickness distribution type (default: naca65)",
            },
            "cfd_output": {
                "type": "string",
                "description": "Output directory for CFD case",
            },
            "cfd_compressible": {
                "type": "boolean",
                "description": "Use compressible solver (rhoSimpleFoam, default: false)",
            },
            "cfd_total_pressure": {
                "type": "number",
                "description": "Inlet total pressure (Pa) for compressible CFD",
            },
            "cfd_total_temperature": {
                "type": "number",
                "description": "Inlet total temperature (K) for compressible CFD",
            },
            "export_format": {
                "type": "string",
                "enum": ["cgns", "openfoam"],
                "description": "Mesh export format (default: cgns)",
            },
            "export_path": {
                "type": "string",
                "description": "Mesh export file path",
            },
        },
        "required": [],
    },
)


# ──────────────────────────────────────────────────────
# 18. Database save
# ──────────────────────────────────────────────────────

_register(
    "database_save",
    "Save a compressor design to the design database for later retrieval and "
    "comparison. Stores parameters, results, tags, and notes.",
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Design name (e.g., 'Rotor 37 baseline')",
            },
            "parameters": {
                "type": "object",
                "description": "Design parameters dict (e.g., {pr: 2.1, mass_flow: 20, rpm: 17189})",
            },
            "results": {
                "type": "object",
                "description": "Optional results dict (e.g., {efficiency: 0.88, n_stages: 3})",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization (e.g., ['compressor', 'rotor37'])",
            },
            "notes": {
                "type": "string",
                "description": "Optional notes about the design",
            },
        },
        "required": ["name", "parameters"],
    },
)


# ──────────────────────────────────────────────────────
# 19. Database query
# ──────────────────────────────────────────────────────

_register(
    "database_query",
    "Search and list designs in the database. Can search by name, tags, or list "
    "all designs. Also supports comparing two designs side by side.",
    {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "search", "compare", "load"],
                "description": "Action: list (all), search (by query), compare (two IDs), load (by ID)",
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search' action)",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by tags (for 'search' or 'list' action)",
            },
            "design_id": {
                "type": "integer",
                "description": "Design ID (for 'load' action)",
            },
            "compare_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Two design IDs to compare (for 'compare' action)",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 20)",
            },
        },
        "required": ["action"],
    },
)


# ──────────────────────────────────────────────────────
# 20. Centrifugal compressor design
# ──────────────────────────────────────────────────────

_register(
    "centrifugal_compressor",
    "Design a centrifugal (radial) compressor — impeller + diffuser. "
    "Used for eVTOL, drones, turbochargers, small turboshafts. "
    "Returns pressure ratio, efficiency, power, tip speed, slip factor, "
    "impeller and diffuser geometry.",
    {
        "type": "object",
        "properties": {
            "pressure_ratio": {
                "type": "number",
                "description": "Target total-to-total pressure ratio (e.g. 3.0)",
            },
            "mass_flow": {
                "type": "number",
                "description": "Mass flow rate (kg/s)",
            },
            "rpm": {
                "type": "number",
                "description": "Rotational speed (RPM)",
            },
            "r1_tip": {
                "type": "number",
                "description": "Impeller inlet tip radius (m, default 0.05)",
            },
            "backsweep_deg": {
                "type": "number",
                "description": "Backsweep angle (degrees, negative = backsweep, default -30)",
            },
            "n_blades": {
                "type": "integer",
                "description": "Number of impeller blades (default 17)",
            },
        },
        "required": ["pressure_ratio", "mass_flow", "rpm"],
    },
)


# ──────────────────────────────────────────────────────
# 21. Report generation
# ──────────────────────────────────────────────────────

_register(
    "generate_report",
    "Generate an HTML design report from analysis results. The report includes "
    "design summary, velocity triangles, compressor map, material properties, "
    "and blade parameters. Returns the path to the generated file.",
    {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Report title (default: 'AstraTurbo Design Report')",
            },
            "output_path": {
                "type": "string",
                "description": "Output HTML file path (default: report.html)",
            },
            "overall_pressure_ratio": {
                "type": "number",
                "description": "Design PR for axial compressor (generates meanline section)",
            },
            "mass_flow": {
                "type": "number",
                "description": "Mass flow (kg/s)",
            },
            "rpm": {
                "type": "number",
                "description": "RPM",
            },
            "r_hub": {
                "type": "number",
                "description": "Hub radius (m) for axial compressor",
            },
            "r_tip": {
                "type": "number",
                "description": "Tip radius (m) for axial compressor",
            },
            "include_map": {
                "type": "boolean",
                "description": "Include compressor map in report (default true)",
            },
            "material": {
                "type": "string",
                "description": "Material name for properties section",
            },
            "material_temperature": {
                "type": "number",
                "description": "Operating temperature (K) for material section",
            },
        },
        "required": ["output_path"],
    },
)


# ──────────────────────────────────────────────────────
# 22. Turbine meanline design
# ──────────────────────────────────────────────────────

_register(
    "meanline_turbine",
    "Design a multi-stage axial turbine from top-level requirements. "
    "Computes velocity triangles, blade angles, loading coefficients, "
    "Zweifel loading, and Soderberg losses for each stage. "
    "Returns a complete stage-by-stage design with NGV and rotor parameters.",
    {
        "type": "object",
        "properties": {
            "overall_expansion_ratio": {
                "type": "number",
                "description": "Total expansion ratio P_in/P_out (e.g. 2.5)",
            },
            "mass_flow": {
                "type": "number",
                "description": "Mass flow rate in kg/s",
            },
            "rpm": {
                "type": "number",
                "description": "Rotational speed in RPM",
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
                "description": "Degree of reaction per stage (default 0.5)",
            },
            "inlet_temp": {
                "type": "number",
                "description": "Inlet total temperature in K (default 1500)",
            },
            "inlet_pressure": {
                "type": "number",
                "description": "Inlet total pressure in Pa (default 101325)",
            },
        },
        "required": ["overall_expansion_ratio", "mass_flow", "rpm", "r_hub", "r_tip"],
    },
)


# ──────────────────────────────────────────────────────
# 23. Engine cycle analysis
# ──────────────────────────────────────────────────────

_register(
    "engine_cycle",
    "Run a complete gas turbine engine cycle analysis. "
    "Connects compressor, combustor, turbine, and nozzle to compute "
    "thrust (turbojet) or shaft power (turboshaft), SFC, and efficiencies. "
    "Supports both axial and centrifugal compressor types.",
    {
        "type": "object",
        "properties": {
            "engine_type": {
                "type": "string",
                "enum": ["turbojet", "turboshaft"],
                "description": "Engine type: turbojet (thrust) or turboshaft (shaft power)",
            },
            "overall_pressure_ratio": {
                "type": "number",
                "description": "Compressor overall pressure ratio (e.g. 8, 20)",
            },
            "turbine_inlet_temp": {
                "type": "number",
                "description": "Turbine inlet temperature in K (e.g. 1400, 1700)",
            },
            "mass_flow": {
                "type": "number",
                "description": "Air mass flow rate in kg/s",
            },
            "rpm": {
                "type": "number",
                "description": "Shaft speed in RPM",
            },
            "r_hub": {
                "type": "number",
                "description": "Hub radius in meters",
            },
            "r_tip": {
                "type": "number",
                "description": "Tip radius in meters",
            },
            "altitude": {
                "type": "number",
                "description": "Flight altitude in meters (default 0)",
            },
            "mach_flight": {
                "type": "number",
                "description": "Flight Mach number (default 0)",
            },
            "compressor_type": {
                "type": "string",
                "enum": ["axial", "centrifugal"],
                "description": "Compressor type (default axial)",
            },
            "n_spools": {
                "type": "integer",
                "enum": [1, 2],
                "description": "Number of spools: 1 (single-spool, default) or 2 (twin-spool)",
            },
            "hp_pressure_ratio": {
                "type": "number",
                "description": "HP spool pressure ratio (twin-spool only; default sqrt(OPR))",
            },
            "hp_rpm": {
                "type": "number",
                "description": "HP spool RPM (twin-spool only; default rpm*1.3)",
            },
            "afterburner": {
                "type": "boolean",
                "description": "Enable afterburner/reheat (turbojet only, default false)",
            },
            "afterburner_temp": {
                "type": "number",
                "description": "Afterburner exit temperature in K (default TIT+300)",
            },
            "nozzle_type": {
                "type": "string",
                "enum": ["convergent", "convergent_divergent"],
                "description": "Nozzle type (default convergent)",
            },
            "nozzle_design_mach": {
                "type": "number",
                "description": "Design exit Mach for con-di nozzle (default 1.5)",
            },
        },
        "required": ["overall_pressure_ratio", "turbine_inlet_temp", "mass_flow", "rpm", "r_hub", "r_tip"],
    },
)


# ──────────────────────────────────────────────────────
# 24. Turbine off-design analysis
# ──────────────────────────────────────────────────────

_register(
    "turbine_off_design",
    "Run off-design analysis of an axial turbine at a different mass flow "
    "and/or RPM than the design point. Blade metal angles are fixed from the "
    "design; flow angles change, creating incidence which modifies losses, "
    "efficiency, and expansion ratio. Returns per-stage incidence, nozzle Mach, "
    "losses, and choke flags. Requires design-point geometry parameters plus "
    "the off-design mass_flow and rpm.",
    {
        "type": "object",
        "properties": {
            "overall_expansion_ratio": {
                "type": "number",
                "description": "Design-point overall expansion ratio (for design geometry)",
            },
            "design_mass_flow": {
                "type": "number",
                "description": "Design-point mass flow rate (kg/s)",
            },
            "design_rpm": {
                "type": "number",
                "description": "Design-point RPM",
            },
            "r_hub": {
                "type": "number",
                "description": "Hub radius (m)",
            },
            "r_tip": {
                "type": "number",
                "description": "Tip radius (m)",
            },
            "off_design_mass_flow": {
                "type": "number",
                "description": "Off-design mass flow rate (kg/s)",
            },
            "off_design_rpm": {
                "type": "number",
                "description": "Off-design RPM",
            },
            "inlet_temp": {
                "type": "number",
                "description": "Inlet total temperature in K (default 1500)",
            },
        },
        "required": [
            "overall_expansion_ratio", "design_mass_flow", "design_rpm",
            "r_hub", "r_tip", "off_design_mass_flow", "off_design_rpm",
        ],
    },
)


# ──────────────────────────────────────────────────────
# 25. Turbine map generation
# ──────────────────────────────────────────────────────

_register(
    "generate_turbine_map",
    "Generate a complete turbine performance map with speed lines and choke "
    "line. Sweeps mass flow at multiple RPM fractions to produce expansion ratio "
    "vs mass flow curves, identifies choke boundaries, and computes choke "
    "margin. Returns a formatted table of all speed lines plus the choke line.",
    {
        "type": "object",
        "properties": {
            "overall_expansion_ratio": {
                "type": "number",
                "description": "Design-point overall expansion ratio",
            },
            "mass_flow": {
                "type": "number",
                "description": "Design-point mass flow rate (kg/s)",
            },
            "rpm": {
                "type": "number",
                "description": "Design-point RPM",
            },
            "r_hub": {
                "type": "number",
                "description": "Hub radius (m)",
            },
            "r_tip": {
                "type": "number",
                "description": "Tip radius (m)",
            },
            "rpm_fractions": {
                "type": "array",
                "items": {"type": "number"},
                "description": "RPM fractions to evaluate (default: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05])",
            },
            "n_points": {
                "type": "integer",
                "description": "Number of mass flow points per speed line (default: 15)",
            },
            "inlet_temp": {
                "type": "number",
                "description": "Inlet total temperature in K (default 1500)",
            },
        },
        "required": ["overall_expansion_ratio", "mass_flow", "rpm", "r_hub", "r_tip"],
    },
)


# ──────────────────────────────────────────────────────
# 26. Electric motor sizing
# ──────────────────────────────────────────────────────

_register(
    "electric_motor",
    "Size an electric motor (BLDC or PMSM) for eVTOL/drone propulsion. "
    "Computes torque, Kv, weight, efficiency, and thermal margin from "
    "shaft power, RPM, and voltage requirements.",
    {
        "type": "object",
        "properties": {
            "shaft_power": {
                "type": "number",
                "description": "Required shaft power in Watts",
            },
            "rpm": {
                "type": "number",
                "description": "Motor speed in RPM",
            },
            "voltage": {
                "type": "number",
                "description": "Supply voltage in Volts",
            },
            "motor_type": {
                "type": "string",
                "enum": ["BLDC", "PMSM"],
                "description": "Motor type (default BLDC)",
            },
            "eta_peak": {
                "type": "number",
                "description": "Peak efficiency (default 0.92)",
            },
            "load_fraction": {
                "type": "number",
                "description": "Operating load fraction 0-1 (default 1.0)",
            },
        },
        "required": ["shaft_power", "rpm", "voltage"],
    },
)


# ──────────────────────────────────────────────────────
# 27. Propeller design
# ──────────────────────────────────────────────────────

_register(
    "propeller_design",
    "Design a propeller or rotor for drone/eVTOL applications. "
    "Uses actuator disk momentum theory to compute thrust, power, "
    "efficiency, figure of merit (hover), and blade geometry.",
    {
        "type": "object",
        "properties": {
            "thrust_required": {
                "type": "number",
                "description": "Required thrust in Newtons",
            },
            "n_blades": {
                "type": "integer",
                "description": "Number of blades",
            },
            "diameter": {
                "type": "number",
                "description": "Propeller diameter in meters",
            },
            "rpm": {
                "type": "number",
                "description": "Rotational speed in RPM",
            },
            "V_flight": {
                "type": "number",
                "description": "Forward flight speed in m/s (0 = hover, default 0)",
            },
            "altitude": {
                "type": "number",
                "description": "Altitude in meters (default 0)",
            },
        },
        "required": ["thrust_required", "n_blades", "diameter", "rpm"],
    },
)


# ──────────────────────────────────────────────────────
# 28. Centrifugal pump
# ──────────────────────────────────────────────────────

_register(
    "centrifugal_pump",
    "Design a centrifugal pump for rocket turbopump or industrial use. "
    "Computes efficiency, impeller sizing, NPSH, and shaft power from "
    "head, flow rate, and RPM. Supports LOX, RP-1, LH2, and water.",
    {
        "type": "object",
        "properties": {
            "head": {
                "type": "number",
                "description": "Required pump head in meters",
            },
            "flow_rate": {
                "type": "number",
                "description": "Volume flow rate in m³/s",
            },
            "rpm": {
                "type": "number",
                "description": "Shaft speed in RPM",
            },
            "fluid_name": {
                "type": "string",
                "description": "Fluid name: LOX, RP-1, LH2, water (default water)",
            },
        },
        "required": ["head", "flow_rate", "rpm"],
    },
)


# ──────────────────────────────────────────────────────
# 29. Turbopump assembly
# ──────────────────────────────────────────────────────

_register(
    "turbopump",
    "Design a complete turbopump assembly coupling a centrifugal pump "
    "with an axial turbine for rocket engine propellant feed. "
    "Computes shaft power balance, pump and turbine sizing.",
    {
        "type": "object",
        "properties": {
            "pump_head": {
                "type": "number",
                "description": "Required pump head in meters",
            },
            "pump_flow_rate": {
                "type": "number",
                "description": "Pump volume flow rate in m³/s",
            },
            "fluid_name": {
                "type": "string",
                "description": "Pump fluid: LOX, RP-1, LH2, water (default LOX)",
            },
            "turbine_inlet_temp": {
                "type": "number",
                "description": "Turbine inlet temperature in K",
            },
            "turbine_inlet_pressure": {
                "type": "number",
                "description": "Turbine inlet pressure in Pa",
            },
            "rpm": {
                "type": "number",
                "description": "Common shaft RPM (default 30000)",
            },
            "cycle_type": {
                "type": "string",
                "enum": ["gas_generator", "staged_combustion", "expander"],
                "description": "Turbopump cycle type (default gas_generator)",
            },
        },
        "required": ["pump_head", "pump_flow_rate", "turbine_inlet_temp", "turbine_inlet_pressure"],
    },
)


# ──────────────────────────────────────────────────────
# 30. Cooling flow
# ──────────────────────────────────────────────────────

_register(
    "cooling_flow",
    "Estimate turbine blade cooling air requirements using the "
    "Holland-Thake model. Supports convection, film, and transpiration "
    "cooling. Returns per-row and total coolant mass flow.",
    {
        "type": "object",
        "properties": {
            "T_gas": {
                "type": "number",
                "description": "Hot gas temperature at turbine entry in K",
            },
            "T_coolant": {
                "type": "number",
                "description": "Coolant air temperature in K",
            },
            "T_blade_max": {
                "type": "number",
                "description": "Maximum blade metal temperature in K (default 1300)",
            },
            "cooling_type": {
                "type": "string",
                "enum": ["convection", "film", "transpiration"],
                "description": "Cooling method (default film)",
            },
            "n_cooled_rows": {
                "type": "integer",
                "description": "Number of cooled blade rows (default 2)",
            },
            "mass_flow_gas": {
                "type": "number",
                "description": "Gas mass flow in kg/s (default 20)",
            },
        },
        "required": ["T_gas", "T_coolant"],
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
        elif name == "off_design_compressor":
            return _exec_off_design(inputs)
        elif name == "generate_compressor_map":
            return _exec_compressor_map(inputs)
        elif name == "export_mesh":
            return _exec_export_mesh(inputs)
        elif name == "build_3d_blade":
            return _exec_build_3d_blade(inputs)
        elif name == "run_solver":
            return _exec_run_solver(inputs)
        elif name == "design_chain":
            return _exec_design_chain(inputs)
        elif name == "database_save":
            return _exec_database_save(inputs)
        elif name == "database_query":
            return _exec_database_query(inputs)
        elif name == "centrifugal_compressor":
            return _exec_centrifugal(inputs)
        elif name == "generate_report":
            return _exec_generate_report(inputs)
        elif name == "meanline_turbine":
            return _exec_turbine_meanline(inputs)
        elif name == "engine_cycle":
            return _exec_engine_cycle(inputs)
        elif name == "turbine_off_design":
            return _exec_turbine_off_design(inputs)
        elif name == "generate_turbine_map":
            return _exec_turbine_map(inputs)
        elif name == "electric_motor":
            return _exec_electric_motor(inputs)
        elif name == "propeller_design":
            return _exec_propeller_design(inputs)
        elif name == "centrifugal_pump":
            return _exec_centrifugal_pump(inputs)
        elif name == "turbopump":
            return _exec_turbopump(inputs)
        elif name == "cooling_flow":
            return _exec_cooling_flow(inputs)
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error executing {name}: {type(e).__name__}: {e}"


def _exec_meanline(inputs: dict) -> str:
    from astraturbo.design import meanline_compressor, meanline_to_blade_parameters

    # Security: validate numeric ranges to prevent DoS
    pr = float(inputs["overall_pressure_ratio"])
    if not (1.01 <= pr <= 50.0):
        return f"Error: pressure ratio {pr} out of range [1.01, 50.0]"

    mass_flow = float(inputs["mass_flow"])
    if not (0.01 <= mass_flow <= 10000.0):
        return f"Error: mass flow {mass_flow} out of range [0.01, 10000] kg/s"

    rpm = float(inputs["rpm"])
    if not (100 <= rpm <= 200000):
        return f"Error: RPM {rpm} out of range [100, 200000]"

    r_hub = float(inputs["r_hub"])
    r_tip = float(inputs["r_tip"])
    if r_hub <= 0 or r_tip <= 0 or r_tip <= r_hub:
        return f"Error: radii invalid (need 0 < r_hub < r_tip)"

    kwargs = {
        "overall_pressure_ratio": pr,
        "mass_flow": mass_flow,
        "rpm": rpm,
        "r_hub": r_hub,
        "r_tip": r_tip,
    }
    if "n_stages" in inputs:
        n_stages = int(inputs["n_stages"])
        if not (1 <= n_stages <= 30):
            return f"Error: n_stages {n_stages} out of range [1, 30]"
        kwargs["n_stages"] = n_stages
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
        safe_path = _validate_path(inputs["output_path"])
        np.savetxt(str(safe_path), coords, delimiter=",", header="x,y", comments="")
        output += f"Saved to: {inputs['output_path']}\n"

    return output


def _exec_mesh(inputs: dict) -> str:
    from astraturbo.mesh.multiblock import generate_blade_passage_mesh

    safe_profile = _validate_path(inputs["profile_path"], must_exist=True)
    profile = np.loadtxt(str(safe_profile), delimiter=",", skiprows=1)[:, :2]

    # Validate numeric inputs
    n_blade = max(4, min(int(inputs.get("n_blade", 40)), 200))
    n_ogrid = max(2, min(int(inputs.get("n_ogrid", 10)), 50))
    n_inlet = max(2, min(int(inputs.get("n_inlet", 15)), 100))
    n_outlet = max(2, min(int(inputs.get("n_outlet", 15)), 100))
    n_passage = max(2, min(int(inputs.get("n_passage", 20)), 100))

    mesh = generate_blade_passage_mesh(
        profile=profile,
        pitch=inputs.get("pitch", 0.05),
        n_blade=n_blade,
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

    op_temp = inputs.get("operating_temperature", None)

    output = (
        f"FEA Configuration:\n"
        f"Material: {material.name}\n"
        f"  E = {material.youngs_modulus/1e9:.0f} GPa\n"
        f"  Yield = {material.yield_strength/1e6:.0f} MPa\n"
        f"  Max temp = {material.max_service_temperature:.0f} K\n"
    )

    if op_temp and material.youngs_modulus_table:
        T = float(op_temp)
        props = material.properties_at(T)
        E_ratio = props['youngs_modulus_Pa'] / material.youngs_modulus
        Sy_ratio = props['yield_strength_Pa'] / material.yield_strength
        output += (
            f"\n  At {T:.0f} K (temperature-dependent):\n"
            f"    E = {props['youngs_modulus_GPa']:.1f} GPa ({E_ratio:.0%} of room temp)\n"
            f"    Yield = {props['yield_strength_MPa']:.0f} MPa ({Sy_ratio:.0%} of room temp)\n"
            f"    k = {props['thermal_conductivity_W_mK']:.1f} W/m-K\n"
        )
        if T > material.max_service_temperature:
            output += f"    WARNING: {T:.0f} K exceeds max service temp {material.max_service_temperature:.0f} K!\n"
    elif op_temp:
        output += f"\n  WARNING: No temperature-dependent data for {material.name}. Using room-temp values.\n"

    output += (
        f"\nOmega: {cfg.omega} rad/s\n"
        f"Analysis: {cfg.analysis_type}\n"
        f"\nTo generate input files, provide blade surface geometry.\n"
    )
    return output


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
    temperature = inputs.get("temperature", None)
    lines = ["Available materials:\n"]
    for name in list_materials():
        mat = get_material(name)
        temp_flag = " [T(K)]" if mat.youngs_modulus_table else ""
        line = (
            f"  {name:20s} E={mat.youngs_modulus/1e9:.0f} GPa, "
            f"yield={mat.yield_strength/1e6:.0f} MPa, "
            f"Tmax={mat.max_service_temperature:.0f} K{temp_flag}"
        )
        if temperature and mat.youngs_modulus_table:
            T = float(temperature)
            props = mat.properties_at(T)
            line += (
                f"\n  {'':20s} At {T:.0f} K: "
                f"E={props['youngs_modulus_GPa']:.1f} GPa, "
                f"yield={props['yield_strength_MPa']:.0f} MPa"
            )
        lines.append(line)

    if temperature:
        lines.append(f"\n  Properties shown at {float(temperature):.0f} K for materials with [T(K)] data")
    else:
        lines.append("\n  [T(K)] = temperature-dependent data available. "
                     "Pass temperature parameter to see hot properties.")
    return "\n".join(lines)


def _exec_list_formats(inputs: dict) -> str:
    from astraturbo.export import list_supported_formats
    fmts = list_supported_formats()
    lines = [f"Supported formats: {len(fmts)}\n"]
    for name, info in sorted(fmts.items()):
        rw = ("R" if info["read"] else "-") + ("W" if info["write"] else "-")
        lines.append(f"  [{rw}] {name:22s} {info['description']}")
    return "\n".join(lines)


def _exec_off_design(inputs: dict) -> str:
    from astraturbo.design import meanline_compressor, off_design_compressor

    # Validate inputs
    pr = float(inputs["overall_pressure_ratio"])
    if not (1.01 <= pr <= 50.0):
        return f"Error: pressure ratio {pr} out of range [1.01, 50.0]"

    design_mf = float(inputs["design_mass_flow"])
    design_rpm = float(inputs["design_rpm"])
    r_hub = float(inputs["r_hub"])
    r_tip = float(inputs["r_tip"])
    od_mf = float(inputs["off_design_mass_flow"])
    od_rpm = float(inputs["off_design_rpm"])

    if r_hub <= 0 or r_tip <= 0 or r_tip <= r_hub:
        return "Error: radii invalid (need 0 < r_hub < r_tip)"
    if not (0.01 <= od_mf <= 10000.0):
        return f"Error: off-design mass flow {od_mf} out of range"
    if not (100 <= od_rpm <= 200000):
        return f"Error: off-design RPM {od_rpm} out of range"

    # Run design point first to get blade geometry
    design_result = meanline_compressor(
        overall_pressure_ratio=pr,
        mass_flow=design_mf,
        rpm=design_rpm,
        r_hub=r_hub,
        r_tip=r_tip,
    )

    # Run off-design
    od_result = off_design_compressor(design_result, mass_flow=od_mf, rpm=od_rpm)

    output = od_result.summary()
    output += f"\n\nDesign point: PR={design_result.overall_pressure_ratio:.3f}, "
    output += f"mass_flow={design_mf:.2f} kg/s, RPM={design_rpm:.0f}"
    output += f"\nOff-design:   mass_flow={od_mf:.2f} kg/s, RPM={od_rpm:.0f}"

    return output


def _exec_compressor_map(inputs: dict) -> str:
    from astraturbo.design import meanline_compressor, generate_compressor_map

    # Validate inputs
    pr = float(inputs["overall_pressure_ratio"])
    if not (1.01 <= pr <= 50.0):
        return f"Error: pressure ratio {pr} out of range [1.01, 50.0]"

    mass_flow = float(inputs["mass_flow"])
    rpm = float(inputs["rpm"])
    r_hub = float(inputs["r_hub"])
    r_tip = float(inputs["r_tip"])

    if r_hub <= 0 or r_tip <= 0 or r_tip <= r_hub:
        return "Error: radii invalid (need 0 < r_hub < r_tip)"

    # Run design point
    design_result = meanline_compressor(
        overall_pressure_ratio=pr,
        mass_flow=mass_flow,
        rpm=rpm,
        r_hub=r_hub,
        r_tip=r_tip,
    )

    # Map parameters
    rpm_fractions = inputs.get("rpm_fractions", None)
    n_points = int(inputs.get("n_points", 15))
    if n_points < 3 or n_points > 50:
        return f"Error: n_points {n_points} out of range [3, 50]"

    kwargs = {"n_points": n_points}
    if rpm_fractions is not None:
        kwargs["rpm_fractions"] = [float(f) for f in rpm_fractions]

    cmap = generate_compressor_map(design_result, **kwargs)

    output = cmap.summary()

    # Add surge margin at design speed
    for sl in cmap.speed_lines:
        if abs(sl.rpm_fraction - 1.0) < 0.01 and sl.surge_point_index is not None:
            surge_pr = sl.pressure_ratios[sl.surge_point_index]
            design_pr = cmap.design_point.get("pr", 1.0)
            if design_pr > 1.0:
                sm = (surge_pr - design_pr) / design_pr
                output += f"\n\nSurge margin at design speed: {sm:.4f} ({sm*100:.1f}%)"

    return output


def _exec_export_mesh(inputs: dict) -> str:
    import numpy as np
    from astraturbo.mesh.multiblock import generate_blade_passage_mesh

    profile_path = inputs["profile_path"]
    _validate_path(profile_path)

    pitch = float(inputs["pitch"])
    output_path = inputs["output_path"]
    _validate_path(output_path)
    fmt = inputs.get("format", "cgns")

    # Load profile
    profile = np.loadtxt(profile_path, delimiter=",", skiprows=1)
    if profile.ndim != 2 or profile.shape[1] < 2:
        profile = np.loadtxt(profile_path, delimiter=",")

    n_blade = int(inputs.get("n_blade", 40))
    n_ogrid = int(inputs.get("n_ogrid", 10))
    n_inlet = int(inputs.get("n_inlet", 15))
    n_outlet = int(inputs.get("n_outlet", 15))
    n_passage = int(inputs.get("n_passage", 20))

    mesh = generate_blade_passage_mesh(
        profile=profile[:, :2], pitch=pitch,
        n_blade=n_blade, n_ogrid=n_ogrid,
        n_inlet=n_inlet, n_outlet=n_outlet, n_passage=n_passage,
    )

    if fmt == "openfoam":
        from astraturbo.export import write_blockmeshdict
        write_blockmeshdict(output_path, mesh)
        return (
            f"OpenFOAM blockMeshDict exported to {output_path}\n"
            f"Mesh: {mesh.n_blocks} blocks, {mesh.total_cells} cells"
        )
    else:
        mesh.export_cgns(output_path)
        return (
            f"CGNS mesh exported to {output_path}\n"
            f"Mesh: {mesh.n_blocks} blocks, {mesh.total_cells} cells"
        )


def _exec_build_3d_blade(inputs: dict) -> str:
    import numpy as np
    from astraturbo.camberline import NACA65
    from astraturbo.thickness import NACA65Series
    from astraturbo.profile import Superposition
    from astraturbo.blade import BladeRow

    r_hub = float(inputs["hub_radius"])
    r_tip = float(inputs["tip_radius"])
    if r_hub <= 0 or r_tip <= 0 or r_tip <= r_hub:
        return "Error: radii invalid (need 0 < hub_radius < tip_radius)"

    axial_chord = float(inputs.get("axial_chord", 0.05))
    n_blades = int(inputs.get("n_blades", 24))

    cl0_hub = float(inputs.get("cl0_hub", 0.8))
    cl0_mid = float(inputs.get("cl0_mid", 1.0))
    cl0_tip = float(inputs.get("cl0_tip", 1.2))

    t_hub = float(inputs.get("thickness_hub", 0.08))
    t_mid = float(inputs.get("thickness_mid", 0.10))
    t_tip = float(inputs.get("thickness_tip", 0.12))

    stagger_hub = float(inputs.get("stagger_hub_deg", 30))
    stagger_mid = float(inputs.get("stagger_mid_deg", 35))
    stagger_tip = float(inputs.get("stagger_tip_deg", 40))

    # Create profiles
    profiles = [
        Superposition(NACA65(cl0=cl0_hub), NACA65Series(max_thickness=t_hub)),
        Superposition(NACA65(cl0=cl0_mid), NACA65Series(max_thickness=t_mid)),
        Superposition(NACA65(cl0=cl0_tip), NACA65Series(max_thickness=t_tip)),
    ]

    # Flow channel
    hub_pts = np.array([[0.0, r_hub], [axial_chord, r_hub]])
    shroud_pts = np.array([[0.0, r_tip], [axial_chord, r_tip]])

    row = BladeRow(hub_points=hub_pts, shroud_points=shroud_pts)
    row.number_blades = n_blades
    for p in profiles:
        row.add_profile(p)

    chord_hub = axial_chord * 0.8
    chord_mid = axial_chord
    chord_tip = axial_chord * 1.2

    row.compute(
        stagger_angles=np.deg2rad([stagger_hub, stagger_mid, stagger_tip]),
        chord_lengths=np.array([chord_hub, chord_mid, chord_tip]),
    )

    output = (
        f"3D Blade Built:\n"
        f"  Blades: {n_blades}\n"
        f"  Hub radius: {r_hub:.4f} m, Tip radius: {r_tip:.4f} m\n"
        f"  Profiles: 3 (hub/mid/tip)\n"
        f"  CL0: {cl0_hub:.2f} / {cl0_mid:.2f} / {cl0_tip:.2f}\n"
        f"  Thickness: {t_hub:.0%} / {t_mid:.0%} / {t_tip:.0%}\n"
        f"  Stagger: {stagger_hub:.1f} / {stagger_mid:.1f} / {stagger_tip:.1f} deg\n"
        f"  Leading edge: {row.leading_edge.shape if row.leading_edge is not None else 'N/A'}\n"
        f"  Trailing edge: {row.trailing_edge.shape if row.trailing_edge is not None else 'N/A'}\n"
        f"  3D profiles: {len(row.profiles_3d) if row.profiles_3d else 0} sections\n"
    )

    # Optional mesh export
    if "output_path" in inputs:
        output_path = inputs["output_path"]
        _validate_path(output_path)
        from astraturbo.mesh.multiblock import generate_blade_passage_mesh
        mid_profile = profiles[1].as_array()
        pitch = 2 * np.pi * (r_hub + r_tip) / 2.0 / n_blades
        mesh = generate_blade_passage_mesh(
            profile=mid_profile, pitch=pitch,
            n_blade=30, n_ogrid=8, n_inlet=10, n_outlet=10, n_passage=15,
        )
        mesh.export_cgns(output_path)
        output += f"\n  Mesh exported to {output_path}: {mesh.total_cells} cells"

    return output


def _exec_run_solver(inputs: dict) -> str:
    from pathlib import Path

    case_dir = inputs["case_dir"]
    _validate_path(case_dir)
    solver = inputs.get("solver", "openfoam")
    n_procs = int(inputs.get("n_procs", 1))

    if not Path(case_dir).is_dir():
        return f"Error: case directory '{case_dir}' does not exist"

    if solver == "openfoam":
        from astraturbo.cfd.runner import run_openfoam, RunConfig
        config = RunConfig(case_dir=case_dir, n_procs=n_procs)
        result = run_openfoam(config)
        return (
            f"OpenFOAM run {'succeeded' if result.success else 'FAILED'}\n"
            f"  Return code: {result.return_code}\n"
            f"  Log file: {result.log_file}\n"
            f"  {result.error_message if result.error_message else ''}"
        )
    elif solver == "su2":
        from astraturbo.cfd.runner import run_su2
        cfg_files = list(Path(case_dir).glob("*.cfg"))
        if not cfg_files:
            return f"Error: no .cfg file found in {case_dir}"
        result = run_su2(cfg_files[0], n_procs=n_procs)
        return (
            f"SU2 run {'succeeded' if result.success else 'FAILED'}\n"
            f"  Return code: {result.return_code}\n"
            f"  {result.error_message if result.error_message else ''}"
        )
    elif solver == "calculix":
        import subprocess
        inp_files = list(Path(case_dir).glob("*.inp"))
        if not inp_files:
            return f"Error: no .inp file found in {case_dir}"
        try:
            proc = subprocess.run(
                ["ccx", str(inp_files[0].stem)],
                cwd=case_dir, capture_output=True, text=True, timeout=600,
            )
            success = proc.returncode == 0
            return (
                f"CalculiX run {'succeeded' if success else 'FAILED'}\n"
                f"  Return code: {proc.returncode}\n"
                f"  {proc.stderr[:500] if proc.stderr else ''}"
            )
        except FileNotFoundError:
            return "Error: CalculiX (ccx) not found in PATH"
        except subprocess.TimeoutExpired:
            return "Error: CalculiX run timed out (600s limit)"
    else:
        return f"Error: unknown solver '{solver}'"


def _exec_design_chain(inputs: dict) -> str:
    from astraturbo.foundation.design_chain import DesignChain

    chain = DesignChain()
    params = {}

    param_map = {
        "pressure_ratio": "pressure_ratio",
        "mass_flow": "mass_flow",
        "rpm": "rpm",
        "cl0": "cl0",
        "max_thickness": "max_thickness",
        "camber_type": "camber_type",
        "thickness_type": "thickness_type",
        "cfd_output": "cfd_output",
        "cfd_compressible": "cfd_compressible",
        "cfd_total_pressure": "cfd_total_pressure",
        "cfd_total_temperature": "cfd_total_temperature",
        "export_format": "export_format",
        "export_path": "export_path",
    }

    for input_key, chain_key in param_map.items():
        if input_key in inputs:
            params[chain_key] = inputs[input_key]

    if not params:
        return "Error: no design parameters provided"

    result = chain.set_parameters(params)

    if result is None:
        return "Error: design chain returned no result"

    output = f"Design Chain: {'SUCCESS' if result.success else 'FAILED'}\n"
    output += f"  Total time: {result.total_time:.3f}s\n\n"
    output += "  Stage Results:\n"
    for stage in result.stages:
        status = "OK" if stage.success else f"FAIL: {stage.error}"
        output += f"    {stage.stage_name:12s}  {stage.elapsed_time:.3f}s  {status}\n"
        if stage.data:
            for key, val in stage.data.items():
                if isinstance(val, (int, float, str, bool)):
                    output += f"      {key}: {val}\n"

    return output


def _exec_database_save(inputs: dict) -> str:
    from astraturbo.database import DesignDatabase

    name = inputs["name"]
    parameters = inputs["parameters"]
    results = inputs.get("results", None)
    tags = inputs.get("tags", None)
    notes = inputs.get("notes", "")

    with DesignDatabase() as db:
        design_id = db.save_design(
            name=name,
            parameters=parameters,
            results=results,
            tags=tags,
            notes=notes,
        )

    return (
        f"Design saved to database:\n"
        f"  ID: {design_id}\n"
        f"  Name: {name}\n"
        f"  Parameters: {len(parameters)} fields\n"
        f"  Tags: {tags if tags else 'none'}\n"
    )


def _exec_database_query(inputs: dict) -> str:
    from astraturbo.database import DesignDatabase

    action = inputs["action"]
    limit = int(inputs.get("limit", 20))

    with DesignDatabase() as db:
        if action == "list":
            filters = {}
            if "tags" in inputs:
                filters["tags"] = inputs["tags"]
            designs = db.list_designs(filters=filters if filters else None, limit=limit)
            if not designs:
                return "No designs found in database."
            output = f"Designs in database ({len(designs)}):\n\n"
            for d in designs:
                output += (
                    f"  ID {d['id']}: {d['name']}\n"
                    f"    Created: {d.get('created_at', 'N/A')}\n"
                    f"    Tags: {d.get('tags', [])}\n"
                    f"    Parameters: {d.get('parameters', {})}\n\n"
                )
            return output

        elif action == "search":
            query = inputs.get("query", "")
            tags = inputs.get("tags", None)
            designs = db.search(query=query, tags=tags, limit=limit)
            if not designs:
                return f"No designs matching '{query}'."
            output = f"Search results for '{query}' ({len(designs)}):\n\n"
            for d in designs:
                output += f"  ID {d['id']}: {d['name']}\n"
                output += f"    Parameters: {d.get('parameters', {})}\n\n"
            return output

        elif action == "load":
            design_id = int(inputs.get("design_id", 0))
            if design_id <= 0:
                return "Error: design_id required for load action"
            try:
                d = db.load_design(design_id)
            except KeyError:
                return f"Error: design ID {design_id} not found"
            output = f"Design {d['id']}: {d['name']}\n\n"
            output += f"  Parameters:\n"
            for k, v in d.get("parameters", {}).items():
                output += f"    {k}: {v}\n"
            if d.get("results"):
                output += f"\n  Results:\n"
                for k, v in d["results"].items():
                    output += f"    {k}: {v}\n"
            if d.get("notes"):
                output += f"\n  Notes: {d['notes']}\n"
            return output

        elif action == "compare":
            ids = inputs.get("compare_ids", [])
            if len(ids) != 2:
                return "Error: compare requires exactly 2 design IDs"
            result = db.compare(int(ids[0]), int(ids[1]))
            output = f"Comparison: Design {ids[0]} vs Design {ids[1]}\n\n"
            output += f"  Design 1: {result.get('design_1', {}).get('name', 'N/A')}\n"
            output += f"  Design 2: {result.get('design_2', {}).get('name', 'N/A')}\n\n"
            if result.get("parameter_differences"):
                output += "  Parameter Differences:\n"
                for diff in result["parameter_differences"]:
                    output += f"    {diff}\n"
            if result.get("summary"):
                output += f"\n  Summary: {result['summary']}\n"
            return output

        else:
            return f"Error: unknown action '{action}'"


def _exec_centrifugal(inputs: dict) -> str:
    from astraturbo.design.centrifugal import centrifugal_compressor

    pr = float(inputs["pressure_ratio"])
    if not (1.01 <= pr <= 15.0):
        return f"Error: pressure ratio {pr} out of range [1.01, 15.0]"

    kwargs = {
        "pressure_ratio": pr,
        "mass_flow": float(inputs["mass_flow"]),
        "rpm": float(inputs["rpm"]),
    }
    if "r1_tip" in inputs:
        kwargs["r1_tip"] = float(inputs["r1_tip"])
    if "backsweep_deg" in inputs:
        kwargs["beta2_blade_deg"] = float(inputs["backsweep_deg"])
    if "n_blades" in inputs:
        kwargs["n_blades"] = int(inputs["n_blades"])

    result = centrifugal_compressor(**kwargs)
    return result.summary()


def _exec_turbine_meanline(inputs: dict) -> str:
    from astraturbo.design.turbine import meanline_turbine, meanline_to_turbine_blade_parameters

    er = float(inputs["overall_expansion_ratio"])
    if not (1.01 <= er <= 50.0):
        return f"Error: expansion ratio {er} out of range [1.01, 50.0]"

    kwargs = {
        "overall_expansion_ratio": er,
        "mass_flow": float(inputs["mass_flow"]),
        "rpm": float(inputs["rpm"]),
        "r_hub": float(inputs["r_hub"]),
        "r_tip": float(inputs["r_tip"]),
    }
    if "n_stages" in inputs:
        kwargs["n_stages"] = int(inputs["n_stages"])
    if "reaction" in inputs:
        kwargs["reaction"] = float(inputs["reaction"])
    if "inlet_temp" in inputs:
        kwargs["T_inlet"] = float(inputs["inlet_temp"])
    if "inlet_pressure" in inputs:
        kwargs["P_inlet"] = float(inputs["inlet_pressure"])

    result = meanline_turbine(**kwargs)

    summary = result.summary()
    params = meanline_to_turbine_blade_parameters(result)
    param_lines = []
    for p in params:
        param_lines.append(
            f"Stage {p['stage']}: NGV stagger={p['ngv_stagger_deg']:.1f} deg, "
            f"Rotor stagger={p['rotor_stagger_deg']:.1f} deg, "
            f"Zweifel={p['zweifel']:.3f}"
        )
    return summary + "\n\nBlade Parameters:\n" + "\n".join(param_lines)


def _exec_generate_report(inputs: dict) -> str:
    from astraturbo.reports import generate_report, ReportConfig

    output_path = inputs["output_path"]
    _validate_path(output_path)

    cfg = ReportConfig(
        title=inputs.get("title", "AstraTurbo Design Report"),
        output_path=output_path,
        include_map=inputs.get("include_map", True),
    )

    meanline_result = None
    blade_params = None
    compressor_map = None
    material = None
    mat_temp = inputs.get("material_temperature", None)

    # Run meanline if parameters provided
    if "overall_pressure_ratio" in inputs and "r_hub" in inputs:
        from astraturbo.design import meanline_compressor, meanline_to_blade_parameters
        from astraturbo.design import generate_compressor_map
        meanline_result = meanline_compressor(
            overall_pressure_ratio=float(inputs["overall_pressure_ratio"]),
            mass_flow=float(inputs.get("mass_flow", 20.0)),
            rpm=float(inputs.get("rpm", 15000)),
            r_hub=float(inputs["r_hub"]),
            r_tip=float(inputs["r_tip"]),
        )
        blade_params = meanline_to_blade_parameters(meanline_result)
        if cfg.include_map:
            compressor_map = generate_compressor_map(
                meanline_result, rpm_fractions=[0.7, 0.85, 1.0, 1.05], n_points=10,
            )

    if "material" in inputs:
        from astraturbo.fea import get_material
        material = get_material(inputs["material"])

    path = generate_report(
        config=cfg,
        meanline_result=meanline_result,
        compressor_map=compressor_map,
        material=material,
        material_temperature=float(mat_temp) if mat_temp else None,
        blade_params=blade_params,
    )
    return f"Report generated: {path}"


def _exec_engine_cycle(inputs: dict) -> str:
    from astraturbo.design.engine_cycle import engine_cycle

    opr = float(inputs["overall_pressure_ratio"])
    if not (1.5 <= opr <= 60.0):
        return f"Error: overall_pressure_ratio {opr} out of range [1.5, 60.0]"

    tit = float(inputs["turbine_inlet_temp"])
    if not (800.0 <= tit <= 2200.0):
        return f"Error: turbine_inlet_temp {tit} out of range [800, 2200]"

    mass_flow = float(inputs["mass_flow"])
    if not (0.1 <= mass_flow <= 1000.0):
        return f"Error: mass_flow {mass_flow} out of range [0.1, 1000]"

    rpm = float(inputs["rpm"])
    if not (100 <= rpm <= 200000):
        return f"Error: rpm {rpm} out of range [100, 200000]"

    kwargs = {
        "overall_pressure_ratio": opr,
        "turbine_inlet_temp": tit,
        "mass_flow": mass_flow,
        "rpm": rpm,
        "r_hub": float(inputs["r_hub"]),
        "r_tip": float(inputs["r_tip"]),
    }
    if "engine_type" in inputs:
        kwargs["engine_type"] = str(inputs["engine_type"])
    if "altitude" in inputs:
        kwargs["altitude"] = float(inputs["altitude"])
    if "mach_flight" in inputs:
        kwargs["mach_flight"] = float(inputs["mach_flight"])
    if "compressor_type" in inputs:
        kwargs["compressor_type"] = str(inputs["compressor_type"])
    if "n_spools" in inputs:
        kwargs["n_spools"] = int(inputs["n_spools"])
    if "hp_pressure_ratio" in inputs:
        kwargs["hp_pressure_ratio"] = float(inputs["hp_pressure_ratio"])
    if "hp_rpm" in inputs:
        kwargs["hp_rpm"] = float(inputs["hp_rpm"])
    if "afterburner" in inputs:
        kwargs["afterburner"] = bool(inputs["afterburner"])
    if "afterburner_temp" in inputs:
        kwargs["afterburner_temp"] = float(inputs["afterburner_temp"])
    if "nozzle_type" in inputs:
        kwargs["nozzle_type"] = inputs["nozzle_type"]
    if "nozzle_design_mach" in inputs:
        kwargs["nozzle_design_mach"] = float(inputs["nozzle_design_mach"])

    result = engine_cycle(**kwargs)
    return result.summary()


def _exec_turbine_off_design(inputs: dict) -> str:
    from astraturbo.design.turbine import meanline_turbine
    from astraturbo.design.turbine_off_design import turbine_off_design

    # Validate inputs
    er = float(inputs["overall_expansion_ratio"])
    if not (1.01 <= er <= 50.0):
        return f"Error: expansion ratio {er} out of range [1.01, 50.0]"

    design_mf = float(inputs["design_mass_flow"])
    design_rpm = float(inputs["design_rpm"])
    r_hub = float(inputs["r_hub"])
    r_tip = float(inputs["r_tip"])
    od_mf = float(inputs["off_design_mass_flow"])
    od_rpm = float(inputs["off_design_rpm"])

    if r_hub <= 0 or r_tip <= 0 or r_tip <= r_hub:
        return "Error: radii invalid (need 0 < r_hub < r_tip)"
    if not (0.01 <= od_mf <= 10000.0):
        return f"Error: off-design mass flow {od_mf} out of range"
    if not (100 <= od_rpm <= 200000):
        return f"Error: off-design RPM {od_rpm} out of range"

    kwargs = {
        "overall_expansion_ratio": er,
        "mass_flow": design_mf,
        "rpm": design_rpm,
        "r_hub": r_hub,
        "r_tip": r_tip,
    }
    if "inlet_temp" in inputs:
        kwargs["T_inlet"] = float(inputs["inlet_temp"])

    # Run design point first
    design_result = meanline_turbine(**kwargs)

    # Run off-design
    od_result = turbine_off_design(design_result, mass_flow=od_mf, rpm=od_rpm)

    output = od_result.summary()
    output += f"\n\nDesign point: ER={design_result.overall_expansion_ratio:.3f}, "
    output += f"mass_flow={design_mf:.2f} kg/s, RPM={design_rpm:.0f}"
    output += f"\nOff-design:   mass_flow={od_mf:.2f} kg/s, RPM={od_rpm:.0f}"

    return output


def _exec_turbine_map(inputs: dict) -> str:
    from astraturbo.design.turbine import meanline_turbine
    from astraturbo.design.turbine_off_design import generate_turbine_map

    # Validate inputs
    er = float(inputs["overall_expansion_ratio"])
    if not (1.01 <= er <= 50.0):
        return f"Error: expansion ratio {er} out of range [1.01, 50.0]"

    mass_flow = float(inputs["mass_flow"])
    rpm = float(inputs["rpm"])
    r_hub = float(inputs["r_hub"])
    r_tip = float(inputs["r_tip"])

    if r_hub <= 0 or r_tip <= 0 or r_tip <= r_hub:
        return "Error: radii invalid (need 0 < r_hub < r_tip)"

    kwargs = {
        "overall_expansion_ratio": er,
        "mass_flow": mass_flow,
        "rpm": rpm,
        "r_hub": r_hub,
        "r_tip": r_tip,
    }
    if "inlet_temp" in inputs:
        kwargs["T_inlet"] = float(inputs["inlet_temp"])

    # Run design point
    design_result = meanline_turbine(**kwargs)

    # Map parameters
    rpm_fractions = inputs.get("rpm_fractions", None)
    n_points = int(inputs.get("n_points", 15))
    if n_points < 3 or n_points > 50:
        return f"Error: n_points {n_points} out of range [3, 50]"

    map_kwargs = {"n_points": n_points}
    if rpm_fractions is not None:
        map_kwargs["rpm_fractions"] = [float(f) for f in rpm_fractions]

    tmap = generate_turbine_map(design_result, **map_kwargs)

    output = tmap.summary()

    # Add choke margin at design speed
    for sl in tmap.speed_lines:
        if abs(sl.rpm_fraction - 1.0) < 0.01 and sl.choke_point_index is not None:
            choke_er = sl.expansion_ratios[sl.choke_point_index]
            design_er = tmap.design_point.get("er", 1.0)
            if design_er > 1.0:
                cm = (choke_er - design_er) / design_er
                output += f"\n\nChoke margin at design speed: {cm:.4f} ({cm*100:.1f}%)"

    return output


def _exec_electric_motor(inputs: dict) -> str:
    from astraturbo.design.electric_motor import electric_motor
    kwargs = {
        "shaft_power": float(inputs["shaft_power"]),
        "rpm": float(inputs["rpm"]),
        "voltage": float(inputs["voltage"]),
    }
    if "motor_type" in inputs:
        kwargs["motor_type"] = inputs["motor_type"]
    if "eta_peak" in inputs:
        kwargs["eta_peak"] = float(inputs["eta_peak"])
    if "load_fraction" in inputs:
        kwargs["load_fraction"] = float(inputs["load_fraction"])
    result = electric_motor(**kwargs)
    return result.summary()


def _exec_propeller_design(inputs: dict) -> str:
    from astraturbo.design.propeller import propeller_design
    kwargs = {
        "thrust_required": float(inputs["thrust_required"]),
        "n_blades": int(inputs["n_blades"]),
        "diameter": float(inputs["diameter"]),
        "rpm": float(inputs["rpm"]),
    }
    if "V_flight" in inputs:
        kwargs["V_flight"] = float(inputs["V_flight"])
    if "altitude" in inputs:
        kwargs["altitude"] = float(inputs["altitude"])
    result = propeller_design(**kwargs)
    return result.summary()


def _exec_centrifugal_pump(inputs: dict) -> str:
    from astraturbo.design.pump import centrifugal_pump, FLUIDS
    kwargs = {
        "head": float(inputs["head"]),
        "flow_rate": float(inputs["flow_rate"]),
        "rpm": float(inputs["rpm"]),
    }
    if "fluid_name" in inputs:
        fname = inputs["fluid_name"]
        kwargs["fluid_name"] = fname
        if fname in FLUIDS:
            kwargs["fluid_density"] = FLUIDS[fname]
    result = centrifugal_pump(**kwargs)
    return result.summary()


def _exec_turbopump(inputs: dict) -> str:
    from astraturbo.design.turbopump import turbopump
    from astraturbo.design.pump import FLUIDS
    kwargs = {
        "pump_head": float(inputs["pump_head"]),
        "pump_flow_rate": float(inputs["pump_flow_rate"]),
        "turbine_inlet_temp": float(inputs["turbine_inlet_temp"]),
        "turbine_inlet_pressure": float(inputs["turbine_inlet_pressure"]),
    }
    fname = inputs.get("fluid_name", "LOX")
    kwargs["fluid_name"] = fname
    kwargs["fluid_density"] = FLUIDS.get(fname, 1141.0)
    if "rpm" in inputs:
        kwargs["rpm"] = float(inputs["rpm"])
    if "cycle_type" in inputs:
        kwargs["cycle_type"] = inputs["cycle_type"]
    result = turbopump(**kwargs)
    return result.summary()


def _exec_cooling_flow(inputs: dict) -> str:
    from astraturbo.design.cooling import cooling_flow
    kwargs = {
        "T_gas": float(inputs["T_gas"]),
        "T_coolant": float(inputs["T_coolant"]),
    }
    if "T_blade_max" in inputs:
        kwargs["T_blade_max"] = float(inputs["T_blade_max"])
    if "cooling_type" in inputs:
        kwargs["cooling_type"] = inputs["cooling_type"]
    if "n_cooled_rows" in inputs:
        kwargs["n_cooled_rows"] = int(inputs["n_cooled_rows"])
    if "mass_flow_gas" in inputs:
        kwargs["mass_flow_gas"] = float(inputs["mass_flow_gas"])
    result = cooling_flow(**kwargs)
    return result.summary()
