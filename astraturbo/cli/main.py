"""CLI entry point for AstraTurbo.

Provides a fully functional command-line interface:
    astraturbo gui                          Launch the GUI
    astraturbo profile --camber naca65 ...  Generate a 2D profile
    astraturbo compute project.yaml         Compute 3D blade geometry
    astraturbo mesh project.yaml            Generate mesh and export
    astraturbo export project.yaml          Export geometry to CAD
    astraturbo info <file>                  Inspect a mesh/points file
    astraturbo throughflow --pr 1.5 ...     Run throughflow solver
    astraturbo smooth --input mesh.cgns ... Smooth a mesh
    astraturbo database {list|save|export}  Design database operations
    astraturbo hpc {submit|status|cancel}   HPC job management
    astraturbo sweep --parameter cl0 ...    Parametric sweep
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="astraturbo",
        description="AstraTurbo — Turbomachinery Design & Simulation Platform",
        epilog=(
            "Examples:\n"
            "  astraturbo gui\n"
            "  astraturbo profile --camber naca65 --thickness naca4digit -o blade.csv\n"
            "  astraturbo mesh --profile blade.csv --pitch 0.05 -o mesh.cgns\n"
            "  astraturbo info /path/to/points\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="astraturbo 0.1.0")

    subparsers = parser.add_subparsers(dest="command")

    # --- gui ---
    subparsers.add_parser("gui", help="Launch the graphical interface")

    # --- ai ---
    ai_parser = subparsers.add_parser("ai", help="AI design assistant (Claude API)")
    ai_parser.add_argument("prompt", nargs="?", default=None,
                           help="Design request (omit for interactive chat)")

    # --- profile ---
    prof_parser = subparsers.add_parser(
        "profile", help="Generate a 2D blade profile"
    )
    prof_parser.add_argument(
        "--camber", default="naca65",
        choices=["circular_arc", "quadratic", "cubic", "quartic",
                 "joukowski", "naca2digit", "naca65", "nurbs"],
        help="Camber line type (default: naca65)",
    )
    prof_parser.add_argument(
        "--thickness", default="naca4digit",
        choices=["naca4digit", "naca65", "joukowski", "elliptic"],
        help="Thickness distribution type (default: naca4digit)",
    )
    prof_parser.add_argument("--cl0", type=float, default=1.0, help="NACA65 lift coefficient")
    prof_parser.add_argument("--max-thickness", type=float, default=0.10, help="Max thickness ratio")
    prof_parser.add_argument("--samples", type=int, default=200, help="Number of sample points")
    prof_parser.add_argument("-o", "--output", default=None, help="Output CSV file (default: print to stdout)")
    prof_parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")

    # --- mesh ---
    mesh_parser = subparsers.add_parser("mesh", help="Generate a mesh from a profile")
    mesh_parser.add_argument("--profile", required=True, help="Path to profile CSV (x,y columns)")
    mesh_parser.add_argument("--pitch", type=float, default=0.05, help="Blade pitch")
    mesh_parser.add_argument("--n-blade", type=int, default=40, help="Cells around blade")
    mesh_parser.add_argument("--n-ogrid", type=int, default=10, help="O-grid wall-normal cells")
    mesh_parser.add_argument("--n-inlet", type=int, default=15, help="Inlet cells")
    mesh_parser.add_argument("--n-outlet", type=int, default=15, help="Outlet cells")
    mesh_parser.add_argument("--n-passage", type=int, default=20, help="Passage pitchwise cells")
    mesh_parser.add_argument(
        "--3d", dest="three_d", action="store_true",
        help="Generate 3D mesh by stacking at multiple span stations",
    )
    mesh_parser.add_argument("--n-span", type=int, default=3,
                             help="Number of span stations for 3D mesh (default: 3)")
    mesh_parser.add_argument("--span", type=float, default=0.05,
                             help="Total span height (m) for 3D mesh (default: 0.05)")
    mesh_parser.add_argument("--with-bcs", action="store_true",
                             help="Write CGNS boundary conditions (inlet/outlet/blade/periodic)")
    mesh_parser.add_argument(
        "--format", choices=["cgns", "openfoam", "vtk"], default="cgns",
        help="Output format (default: cgns)",
    )
    mesh_parser.add_argument("-o", "--output", default="mesh.cgns", help="Output file path")

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Inspect a mesh or points file")
    info_parser.add_argument("file", help="Path to file (OpenFOAM points, CGNS, etc.)")

    # --- cfd ---
    cfd_parser = subparsers.add_parser("cfd", help="Set up a CFD case")
    cfd_parser.add_argument(
        "--solver", choices=["openfoam", "fluent", "cfx", "su2"], default="openfoam",
        help="CFD solver (default: openfoam)",
    )
    cfd_parser.add_argument("--velocity", type=float, default=100.0, help="Inlet velocity (m/s)")
    cfd_parser.add_argument("--turbulence", default="kOmegaSST", help="Turbulence model")
    cfd_parser.add_argument("--rotating", action="store_true", help="Enable rotating frame (rotor)")
    cfd_parser.add_argument("--omega", type=float, default=0.0, help="Angular velocity (rad/s)")
    cfd_parser.add_argument("--mesh", default=None, help="Path to mesh file (CGNS, .msh, etc.)")
    cfd_parser.add_argument("--nprocs", type=int, default=1, help="Number of parallel processes")
    cfd_parser.add_argument("--compressible", action="store_true", help="Use compressible solver (rhoSimpleFoam)")
    cfd_parser.add_argument("--total-pressure", type=float, default=101325.0, help="Inlet total pressure (Pa)")
    cfd_parser.add_argument("--total-temperature", type=float, default=288.15, help="Inlet total temperature (K)")
    cfd_parser.add_argument("-o", "--output", default="cfd_case", help="Output case directory")

    # --- meanline ---
    ml_parser = subparsers.add_parser("meanline", help="Meanline compressor design")
    ml_parser.add_argument("--pr", type=float, required=True, help="Overall pressure ratio")
    ml_parser.add_argument("--mass-flow", type=float, required=True, help="Mass flow rate (kg/s)")
    ml_parser.add_argument("--rpm", type=float, required=True, help="Rotational speed (RPM)")
    ml_parser.add_argument("--r-hub", type=float, required=True, help="Hub radius (m)")
    ml_parser.add_argument("--r-tip", type=float, required=True, help="Tip radius (m)")
    ml_parser.add_argument("--n-stages", type=int, default=None, help="Number of stages (auto if omitted)")
    ml_parser.add_argument("--reaction", type=float, default=0.5, help="Degree of reaction (default: 0.5)")
    ml_parser.add_argument("--eta", type=float, default=0.90, help="Polytropic efficiency (default: 0.90)")
    ml_parser.add_argument("--radial-stations", type=int, default=3,
                           help="Number of radial stations for blade angles (default: 3 = hub/mid/tip)")
    ml_parser.add_argument("--off-design", action="store_true",
                           help="Run single off-design point at given conditions")
    ml_parser.add_argument("--map", action="store_true",
                           help="Generate full compressor map and print speed line table")
    ml_parser.add_argument("--rpm-fractions", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95,1.0,1.05",
                           help="Comma-separated RPM fractions for map (default: 0.5,0.6,...,1.05)")
    ml_parser.add_argument("--report", type=str, default=None,
                           help="Generate HTML design report to this file")

    # --- blade ---
    bl_parser = subparsers.add_parser("blade", help="Build 3D blade from hub-to-tip profiles")
    bl_parser.add_argument("--r-hub", type=float, required=True, help="Hub radius (m)")
    bl_parser.add_argument("--r-tip", type=float, required=True, help="Tip radius (m)")
    bl_parser.add_argument("--axial-chord", type=float, default=0.05, help="Axial chord length (m)")
    bl_parser.add_argument("--n-blades", type=int, default=24, help="Number of blades")
    bl_parser.add_argument("--cl0-hub", type=float, default=0.8, help="CL0 at hub")
    bl_parser.add_argument("--cl0-mid", type=float, default=1.0, help="CL0 at midspan")
    bl_parser.add_argument("--cl0-tip", type=float, default=1.2, help="CL0 at tip")
    bl_parser.add_argument("--thickness-hub", type=float, default=0.08, help="Max thickness at hub")
    bl_parser.add_argument("--thickness-mid", type=float, default=0.10, help="Max thickness at midspan")
    bl_parser.add_argument("--thickness-tip", type=float, default=0.12, help="Max thickness at tip")
    bl_parser.add_argument("--stagger-hub", type=float, default=30.0, help="Stagger at hub (deg)")
    bl_parser.add_argument("--stagger-mid", type=float, default=35.0, help="Stagger at midspan (deg)")
    bl_parser.add_argument("--stagger-tip", type=float, default=40.0, help="Stagger at tip (deg)")
    bl_parser.add_argument("-o", "--output", default=None, help="Output CGNS mesh file")

    # --- pipeline ---
    pipe_parser = subparsers.add_parser("pipeline", help="Run full design pipeline (meanline→profile→blade→mesh→CFD)")
    pipe_parser.add_argument("--pr", type=float, default=1.5, help="Pressure ratio")
    pipe_parser.add_argument("--mass-flow", type=float, default=20.0, help="Mass flow (kg/s)")
    pipe_parser.add_argument("--rpm", type=float, default=15000.0, help="RPM")
    pipe_parser.add_argument("--cl0", type=float, default=None, help="Override CL0 (auto from meanline if omitted)")
    pipe_parser.add_argument("--cfd-output", default=None, help="CFD case output directory")
    pipe_parser.add_argument("--compressible", action="store_true", help="Use compressible CFD")

    # --- centrifugal ---
    cent_parser = subparsers.add_parser("centrifugal", help="Centrifugal compressor meanline design")
    cent_parser.add_argument("--pr", type=float, required=True, help="Pressure ratio")
    cent_parser.add_argument("--mass-flow", type=float, required=True, help="Mass flow (kg/s)")
    cent_parser.add_argument("--rpm", type=float, required=True, help="RPM")
    cent_parser.add_argument("--r1-hub", type=float, default=0.02, help="Impeller inlet hub radius (m)")
    cent_parser.add_argument("--r1-tip", type=float, default=0.05, help="Impeller inlet tip radius (m)")
    cent_parser.add_argument("--r2", type=float, default=None, help="Impeller exit radius (m, auto if omitted)")
    cent_parser.add_argument("--backsweep", type=float, default=-30.0, help="Backsweep angle (deg, default -30)")
    cent_parser.add_argument("--n-blades", type=int, default=17, help="Number of impeller blades")
    cent_parser.add_argument("--report", type=str, default=None, help="Generate HTML report to this file")

    # --- turbine ---
    turb_parser = subparsers.add_parser("turbine", help="Axial turbine meanline design")
    turb_parser.add_argument("--expansion-ratio", type=float, required=True, help="Overall expansion ratio P_in/P_out")
    turb_parser.add_argument("--mass-flow", type=float, default=20.0, help="Mass flow (kg/s)")
    turb_parser.add_argument("--rpm", type=float, default=17189.0, help="RPM")
    turb_parser.add_argument("--r-hub", type=float, default=0.25, help="Hub radius (m)")
    turb_parser.add_argument("--r-tip", type=float, default=0.35, help="Tip radius (m)")
    turb_parser.add_argument("--n-stages", type=int, default=None, help="Number of stages (auto if omitted)")
    turb_parser.add_argument("--reaction", type=float, default=0.5, help="Degree of reaction (default 0.5)")
    turb_parser.add_argument("--eta", type=float, default=0.90, help="Polytropic efficiency (default 0.90)")
    turb_parser.add_argument("--inlet-temp", type=float, default=1500.0, help="Inlet total temperature (K)")
    turb_parser.add_argument("--inlet-pressure", type=float, default=101325.0, help="Inlet total pressure (Pa)")
    turb_parser.add_argument("--radial-stations", type=int, default=3, help="Radial stations for free-vortex")
    turb_parser.add_argument("--off-design", action="store_true",
                             help="Run single off-design point at given conditions")
    turb_parser.add_argument("--map", action="store_true",
                             help="Generate full turbine map and print speed line table")
    turb_parser.add_argument("--rpm-fractions", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95,1.0,1.05",
                             help="Comma-separated RPM fractions for map (default: 0.5,0.6,...,1.05)")
    turb_parser.add_argument("--report", type=str, default=None, help="Generate HTML report to this file")

    # --- fea ---
    fea_parser = subparsers.add_parser("fea", help="Set up FEA structural analysis")
    fea_parser.add_argument("--material", default="inconel_718", help="Material name")
    fea_parser.add_argument("--omega", type=float, default=0.0, help="Angular velocity (rad/s)")
    fea_parser.add_argument("--thickness", type=float, default=0.002, help="Blade wall thickness (m)")
    fea_parser.add_argument("--analysis", choices=["static", "frequency", "buckle"], default="static")
    fea_parser.add_argument("--surface", default=None, help="Blade surface points file (CSV: x,y,z)")
    fea_parser.add_argument("--ni", type=int, default=20, help="Surface grid points in streamwise dir")
    fea_parser.add_argument("--nj", type=int, default=10, help="Surface grid points in spanwise dir")
    fea_parser.add_argument("--list-materials", action="store_true", help="List available materials")
    fea_parser.add_argument("--temperature", type=float, default=None,
                           help="Operating temperature (K) for temperature-dependent properties")
    fea_parser.add_argument("-o", "--output", default="fea_case", help="Output case directory")

    # --- optimize ---
    opt_parser = subparsers.add_parser("optimize", help="Run blade design optimization")
    opt_parser.add_argument("--profile", required=True, help="Base profile CSV")
    opt_parser.add_argument("--pitch", type=float, default=0.05, help="Blade pitch")
    opt_parser.add_argument("--n-profiles", type=int, default=3, help="Span profiles to optimize")
    opt_parser.add_argument("--generations", type=int, default=50, help="Optimization generations")
    opt_parser.add_argument("--population", type=int, default=20, help="Population size")
    opt_parser.add_argument("-o", "--output", default=None, help="Save best design CSV")

    # --- yplus ---
    yp_parser = subparsers.add_parser("yplus", help="y+ calculator for mesh design")
    yp_parser.add_argument("--velocity", type=float, required=True, help="Freestream velocity (m/s)")
    yp_parser.add_argument("--chord", type=float, required=True, help="Reference chord length (m)")
    yp_parser.add_argument("--density", type=float, default=1.225, help="Fluid density (kg/m3)")
    yp_parser.add_argument("--viscosity", type=float, default=1.8e-5, help="Dynamic viscosity (Pa.s)")
    yp_parser.add_argument("--target-yplus", type=float, default=1.0, help="Target y+ value")
    yp_parser.add_argument("--cell-height", type=float, default=None, help="First cell height (m) to check")

    # --- formats ---
    subparsers.add_parser("formats", help="List all supported file formats")

    # --- material-advisor ---
    ma_parser = subparsers.add_parser("material-advisor", help="Recommend materials for engine components")
    ma_parser.add_argument("--component", default=None,
                           help="Component type (fan_blade, turbine_blade, combustor_liner, ...)")
    ma_parser.add_argument("--temperature", type=float, default=None, help="Operating temperature (K)")
    ma_parser.add_argument("--stress", type=float, default=0.0, help="Expected stress (MPa)")
    ma_parser.add_argument("--engine", action="store_true",
                           help="Show full engine material map with default temperatures")
    ma_parser.add_argument("--t-fan", type=float, default=350.0, help="Fan temperature (K)")
    ma_parser.add_argument("--t-compressor", type=float, default=750.0, help="HP compressor exit temp (K)")
    ma_parser.add_argument("--t-combustor", type=float, default=1400.0, help="Combustor liner temp (K)")
    ma_parser.add_argument("--t-turbine", type=float, default=1350.0, help="Turbine inlet temp (K)")
    ma_parser.add_argument("--t-nozzle", type=float, default=1000.0, help="Nozzle temp (K)")

    # --- multistage ---
    ms_parser = subparsers.add_parser("multistage", help="Generate multi-stage mesh")
    ms_parser.add_argument("--profiles", nargs="+", required=True, help="Profile CSVs (one per row)")
    ms_parser.add_argument("--pitches", nargs="+", type=float, required=True, help="Pitch per row")
    ms_parser.add_argument("--names", nargs="+", default=None, help="Row names (default: row0, row1...)")
    ms_parser.add_argument("-o", "--output", default="multistage.cgns", help="Output CGNS file")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Execute a CFD/FEA solver")
    run_parser.add_argument("case", help="Path to case directory")
    run_parser.add_argument("--solver", choices=["openfoam", "su2", "calculix"], default="openfoam")
    run_parser.add_argument("--nprocs", type=int, default=1, help="Number of parallel processes")

    # --- throughflow ---
    tf_parser = subparsers.add_parser("throughflow", help="Run throughflow (S2m) solver")
    tf_parser.add_argument("--pr", type=float, default=1.5, help="Pressure ratio")
    tf_parser.add_argument("--mass-flow", type=float, default=10.0, help="Mass flow rate (kg/s)")
    tf_parser.add_argument("--rpm", type=float, default=10000.0, help="Rotational speed (RPM)")
    tf_parser.add_argument("--r-hub", type=float, default=0.1, help="Hub radius (m)")
    tf_parser.add_argument("--r-tip", type=float, default=0.2, help="Tip radius (m)")
    tf_parser.add_argument("--n-streamwise", type=int, default=20, help="Number of streamwise stations")
    tf_parser.add_argument("--n-radial", type=int, default=10, help="Number of radial streamlines")

    # --- smooth ---
    sm_parser = subparsers.add_parser("smooth", help="Apply Laplacian smoothing to a mesh")
    sm_parser.add_argument("--input", required=True, help="Input mesh file (CGNS)")
    sm_parser.add_argument("--iterations", type=int, default=50, help="Smoothing iterations")
    sm_parser.add_argument("--output", default="smoothed.cgns", help="Output file path")

    # --- database ---
    db_parser = subparsers.add_parser("database", help="Design database operations")
    db_sub = db_parser.add_subparsers(dest="db_command")
    db_sub.add_parser("list", help="List all saved designs")
    db_save = db_sub.add_parser("save", help="Save a new design")
    db_save.add_argument("--name", required=True, help="Design name")
    db_save.add_argument("--params", default="{}", help="Parameters as JSON string")
    db_export = db_sub.add_parser("export", help="Export designs to CSV")
    db_export.add_argument("filepath", help="Output CSV file path")

    # --- hpc ---
    hpc_parser = subparsers.add_parser("hpc", help="HPC job management")
    hpc_sub = hpc_parser.add_subparsers(dest="hpc_command")
    hpc_submit = hpc_sub.add_parser("submit", help="Submit a CFD job")
    hpc_submit.add_argument("case", help="Path to case directory")
    hpc_submit.add_argument("--backend", choices=["local", "slurm", "pbs", "aws"], default="local",
                            help="HPC backend (default: local)")
    hpc_submit.add_argument("--solver", choices=["openfoam", "su2"], default="openfoam",
                            help="CFD solver (default: openfoam)")
    hpc_submit.add_argument("--nprocs", type=int, default=1, help="Number of MPI processes")
    hpc_submit.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    hpc_submit.add_argument("--walltime", default="2:00:00", help="Wall time limit (HH:MM:SS)")
    hpc_submit.add_argument("--host", default="", help="Remote hostname (for slurm/pbs)")
    hpc_submit.add_argument("--user", default="", help="Remote username (for slurm/pbs)")
    hpc_submit.add_argument("--ssh-key", default="", help="SSH key path (for slurm/pbs)")
    # AWS Batch options
    hpc_submit.add_argument("--aws-region", default="us-east-1", help="AWS region (for aws backend)")
    hpc_submit.add_argument("--aws-job-queue", default="", help="AWS Batch job queue name")
    hpc_submit.add_argument("--aws-job-definition", default="", help="AWS Batch job definition name/ARN")
    hpc_submit.add_argument("--aws-s3-bucket", default="", help="S3 bucket for case data")
    hpc_submit.add_argument("--aws-container-image", default="", help="Docker image for solver")
    hpc_status = hpc_sub.add_parser("status", help="Check job status")
    hpc_status.add_argument("job_id", help="Job ID to check")
    hpc_cancel = hpc_sub.add_parser("cancel", help="Cancel a running job")
    hpc_cancel.add_argument("job_id", help="Job ID to cancel")
    hpc_download = hpc_sub.add_parser("download", help="Download job results")
    hpc_download.add_argument("job_id", help="Job ID to download results for")
    hpc_download.add_argument("--output-dir", default=".", help="Local directory for results (default: .)")
    hpc_setup = hpc_sub.add_parser("setup-aws", help="Provision AWS Batch infrastructure")
    hpc_setup.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    hpc_setup.add_argument("--platform", choices=["EC2", "FARGATE"], default="EC2", help="Compute platform")
    hpc_setup.add_argument("--bucket-name", default="", help="S3 bucket name (auto-generated if empty)")
    hpc_setup.add_argument("--max-vcpus", type=int, default=256, help="Max vCPUs (default: 256)")
    hpc_teardown = hpc_sub.add_parser("teardown-aws", help="Delete all AstraTurbo AWS resources")
    hpc_teardown.add_argument("--region", default="us-east-1", help="AWS region")

    # --- sweep ---
    sw_parser = subparsers.add_parser("sweep", help="Run a parametric sweep")
    sw_parser.add_argument("--parameter", required=True, help="Parameter name to sweep")
    sw_parser.add_argument("--start", type=float, required=True, help="Start value")
    sw_parser.add_argument("--end", type=float, required=True, help="End value")
    sw_parser.add_argument("--steps", type=int, default=5, help="Number of sweep steps")

    # --- engine-cycle ---
    ec_parser = subparsers.add_parser("engine-cycle", help="Full engine cycle analysis (turbojet/turboshaft)")
    ec_parser.add_argument("--engine-type", choices=["turbojet", "turboshaft"], default="turbojet", help="Engine type")
    ec_parser.add_argument("--altitude", type=float, default=0.0, help="Altitude (m)")
    ec_parser.add_argument("--mach", type=float, default=0.0, help="Flight Mach number")
    ec_parser.add_argument("--opr", type=float, required=True, help="Overall pressure ratio")
    ec_parser.add_argument("--tit", type=float, required=True, help="Turbine inlet temperature (K)")
    ec_parser.add_argument("--mass-flow", type=float, default=20.0, help="Air mass flow (kg/s)")
    ec_parser.add_argument("--rpm", type=float, default=15000.0, help="Shaft speed (RPM)")
    ec_parser.add_argument("--r-hub", type=float, default=0.15, help="Hub radius (m)")
    ec_parser.add_argument("--r-tip", type=float, default=0.30, help="Tip radius (m)")
    ec_parser.add_argument("--compressor-type", choices=["axial", "centrifugal"], default="axial", help="Compressor type")
    ec_parser.add_argument("--n-spools", type=int, default=1, choices=[1, 2], help="Number of spools (1=single, 2=twin)")
    ec_parser.add_argument("--hp-pr", type=float, default=None, help="HP spool pressure ratio (twin-spool; default sqrt(OPR))")
    ec_parser.add_argument("--hp-rpm", type=float, default=None, help="HP spool RPM (twin-spool; default rpm*1.3)")
    ec_parser.add_argument("--hp-r-hub", type=float, default=None, help="HP blade hub radius (twin-spool; default r_hub*0.8)")
    ec_parser.add_argument("--hp-r-tip", type=float, default=None, help="HP blade tip radius (twin-spool; default r_tip*0.8)")
    ec_parser.add_argument("--report", type=str, default=None, help="Generate HTML report to this file")

    # Parse
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "gui":
        _cmd_gui()
    elif args.command == "ai":
        _cmd_ai(args)
    elif args.command == "profile":
        _cmd_profile(args)
    elif args.command == "mesh":
        _cmd_mesh(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "cfd":
        _cmd_cfd(args)
    elif args.command == "meanline":
        _cmd_meanline(args)
    elif args.command == "fea":
        _cmd_fea(args)
    elif args.command == "optimize":
        _cmd_optimize(args)
    elif args.command == "yplus":
        _cmd_yplus(args)
    elif args.command == "formats":
        _cmd_formats()
    elif args.command == "material-advisor":
        _cmd_material_advisor(args)
    elif args.command == "multistage":
        _cmd_multistage(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "throughflow":
        _cmd_throughflow(args)
    elif args.command == "smooth":
        _cmd_smooth(args)
    elif args.command == "database":
        _cmd_database(args)
    elif args.command == "hpc":
        _cmd_hpc(args)
    elif args.command == "sweep":
        _cmd_sweep(args)
    elif args.command == "blade":
        _cmd_blade(args)
    elif args.command == "pipeline":
        _cmd_pipeline(args)
    elif args.command == "centrifugal":
        _cmd_centrifugal(args)
    elif args.command == "turbine":
        _cmd_turbine(args)
    elif args.command == "engine-cycle":
        _cmd_engine_cycle(args)


# ────────────────────────────────────────────────────────────────
# Command implementations
# ────────────────────────────────────────────────────────────────

def _cmd_gui():
    """Launch the GUI."""
    try:
        from astraturbo.gui.app import main as gui_main
        gui_main()
    except ImportError:
        print("ERROR: GUI dependencies not installed.")
        print("Install with:  pip install astraturbo[gui]")
        sys.exit(1)


def _cmd_ai(args):
    """AI design assistant."""
    try:
        from astraturbo.ai import create_assistant, chat_cli
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.prompt:
        # Single-shot mode
        try:
            assistant = create_assistant()
            response = assistant.chat(args.prompt)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive chat mode
        chat_cli()


def _cmd_profile(args):
    """Generate a 2D blade profile."""
    from astraturbo.camberline import create_camberline
    from astraturbo.thickness import create_thickness
    from astraturbo.profile import Superposition

    # Build camber line
    camber_kwargs = {}
    if args.camber == "naca65":
        camber_kwargs["cl0"] = args.cl0
    cl = create_camberline(args.camber, **camber_kwargs)
    cl.sample_rate = args.samples

    # Build thickness
    td = create_thickness(args.thickness, max_thickness=args.max_thickness)

    # Build profile
    profile = Superposition(cl, td)
    coords = profile.as_array()

    print(f"Profile generated: {args.camber} + {args.thickness}")
    print(f"  Points: {len(coords)}")
    print(f"  X range: {coords[:, 0].min():.6f} to {coords[:, 0].max():.6f}")
    print(f"  Y range: {coords[:, 1].min():.6f} to {coords[:, 1].max():.6f}")

    # Output
    if args.output:
        np.savetxt(args.output, coords, delimiter=",", header="x,y", comments="")
        print(f"  Saved to: {args.output}")
    else:
        print("\n  First 10 points:")
        for i in range(min(10, len(coords))):
            print(f"    ({coords[i, 0]:.8f}, {coords[i, 1]:.8f})")
        print(f"    ... ({len(coords)} total)")

    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            camber_pts = cl.as_array()
            plt.figure(figsize=(10, 4))
            plt.plot(coords[:, 0], coords[:, 1], "b-", linewidth=2, label="Profile")
            plt.plot(camber_pts[:, 0], camber_pts[:, 1], "r--", label="Camber line")
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title(f"{args.camber} + {args.thickness}")
            plt.xlabel("x/c")
            plt.ylabel("y/c")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  (matplotlib not installed — skipping plot)")


def _cmd_mesh(args):
    """Generate a mesh from a profile CSV."""
    from astraturbo.mesh.multiblock import (
        generate_blade_passage_mesh,
        generate_blade_passage_mesh_3d,
    )

    # Load profile
    prof_path = Path(args.profile)
    if not prof_path.exists():
        print(f"ERROR: Profile file not found: {prof_path}")
        sys.exit(1)
    profile = np.loadtxt(args.profile, delimiter=",", skiprows=1)
    if profile.shape[1] < 2:
        print(f"ERROR: Profile file must have at least 2 columns (x, y)")
        sys.exit(1)
    profile = profile[:, :2]

    print(f"Profile loaded: {len(profile)} points from {args.profile}")

    # Generate mesh (2D or 3D)
    if args.three_d:
        # Create slightly scaled profiles at each span station
        span_positions = np.linspace(0, args.span, args.n_span).tolist()
        profiles = []
        for i, z in enumerate(span_positions):
            # Slight thickness variation along span (thinner at tip)
            scale = 1.0 - 0.15 * (z / args.span) if args.span > 0 else 1.0
            p = profile.copy()
            p[:, 1] *= scale
            profiles.append(p)

        mesh = generate_blade_passage_mesh_3d(
            profiles=profiles,
            span_positions=span_positions,
            pitch=args.pitch,
            n_blade=args.n_blade,
            n_ogrid=args.n_ogrid,
            n_inlet=args.n_inlet,
            n_outlet=args.n_outlet,
            n_passage=args.n_passage,
        )
        print(f"3D mesh generated: {mesh.n_blocks} blocks, {args.n_span} span stations")
    else:
        mesh = generate_blade_passage_mesh(
            profile=profile,
            pitch=args.pitch,
            n_blade=args.n_blade,
            n_ogrid=args.n_ogrid,
            n_inlet=args.n_inlet,
            n_outlet=args.n_outlet,
            n_passage=args.n_passage,
        )
        print(f"Mesh generated: {mesh.n_blocks} blocks, {mesh.total_cells} cells")

    # Build patches dict for CGNS BC export
    cgns_patches = None
    if args.with_bcs and args.format == "cgns":
        cgns_patches = {}
        for block in mesh.blocks:
            if block.patches:
                cgns_patches[block.name] = block.patches
        if cgns_patches:
            print(f"CGNS boundary conditions: {sum(len(v) for v in cgns_patches.values())} BCs across {len(cgns_patches)} blocks")

    # Export
    output = Path(args.output)
    if args.format == "cgns":
        if not output.suffix:
            output = output.with_suffix(".cgns")
        from astraturbo.export.cgns_writer import write_cgns_structured
        block_arrays = [b.points for b in mesh.blocks]
        block_names = [b.name for b in mesh.blocks]
        write_cgns_structured(output, block_arrays, block_names, patches=cgns_patches)
    elif args.format == "openfoam":
        mesh.export_openfoam(output)
    elif args.format == "vtk":
        if not output.suffix:
            output = output.with_suffix(".vtk")
        from astraturbo.export import export_structured_as_quads
        block_arrays = [b.points for b in mesh.blocks]
        export_structured_as_quads(output, block_arrays, file_format="vtk")

    print(f"Exported: {output} ({args.format})")


def _cmd_info(args):
    """Inspect a mesh or data file."""
    filepath = Path(args.file)

    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    if not filepath.is_file():
        print(f"ERROR: Not a file: {filepath}")
        sys.exit(1)

    size = filepath.stat().st_size
    if size == 0:
        print(f"ERROR: File is empty: {filepath}")
        sys.exit(1)

    # Check if binary
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(min(8192, size))
        is_binary = b"\x00" in chunk
    except OSError as e:
        print(f"ERROR: Cannot read file: {e}")
        sys.exit(1)

    # Try CGNS / HDF5 (binary format)
    if filepath.suffix in (".cgns", ".hdf5", ".h5"):
        try:
            _info_cgns(filepath)
            return
        except Exception:
            print(f"ERROR: File has CGNS/HDF5 extension but cannot be read as HDF5.")
            print(f"  File may be corrupted or in an unsupported format.")
            sys.exit(1)

    if is_binary:
        print(f"ERROR: File appears to be binary: {filepath}")
        print(f"  Size: {size:,} bytes")
        print(f"  Supported binary formats: CGNS (.cgns)")
        print(f"  If this is an OpenFOAM file, it may be in binary format.")
        print(f"  AstraTurbo requires ASCII-format OpenFOAM files.")
        sys.exit(1)

    # Text file — try OpenFOAM first (regardless of extension)
    from astraturbo.export.openfoam_reader import validate_openfoam_file
    is_foam, _ = validate_openfoam_file(filepath)
    if is_foam:
        _info_openfoam_points(filepath)
        return

    # Try CSV profile
    if filepath.suffix in (".csv", ".dat", ".txt"):
        _info_csv(filepath)
        return

    # Unknown
    print(f"File: {filepath}")
    print(f"Size: {size:,} bytes")
    print(f"Type: Unrecognized format")
    print(f"Supported formats:")
    print(f"  - OpenFOAM points (ASCII, with FoamFile header)")
    print(f"  - CGNS (.cgns)")
    print(f"  - CSV (.csv, .dat, .txt)")


def _info_openfoam_points(filepath):
    """Display info about an OpenFOAM points file."""
    from astraturbo.export import read_openfoam_points, openfoam_points_to_cloud
    from astraturbo.export.openfoam_reader import OpenFOAMReadError

    try:
        points = read_openfoam_points(filepath)
        stats = openfoam_points_to_cloud(points)

        print(f"OpenFOAM Points File: {filepath}")
        print(f"  Total points:  {stats['n_points']:,}")
        print(f"  X range:       {stats['x_min']:.6f} to {stats['x_max']:.6f}  ({stats['x_range']*1000:.2f} mm)")
        print(f"  Y range:       {stats['y_min']:.6f} to {stats['y_max']:.6f}  ({stats['y_range']*1000:.2f} mm)")
        print(f"  Z range:       {stats['z_min']:.6f} to {stats['z_max']:.6f}  ({stats['z_range']*1000:.2f} mm)")
        print(f"  Centroid:      ({stats['centroid'][0]:.6f}, {stats['centroid'][1]:.6f}, {stats['centroid'][2]:.6f})")
    except OpenFOAMReadError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _info_cgns(filepath):
    """Display info about a CGNS file."""
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            print(f"CGNS File: {filepath}")
            print(f"  Size: {filepath.stat().st_size:,} bytes")
            print(f"  Top-level groups:")
            for key in f.keys():
                print(f"    {key}")
                if isinstance(f[key], h5py.Group):
                    for subkey in f[key].keys():
                        print(f"      {subkey}")
    except Exception as e:
        print(f"Error reading CGNS: {e}")


def _info_csv(filepath):
    """Display info about a CSV data file."""
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        print(f"CSV File: {filepath}")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {data.shape[1]}")
        print(f"  Rows: {data.shape[0]}")
        for col in range(min(data.shape[1], 5)):
            print(f"  Column {col}: min={data[:, col].min():.6f}, max={data[:, col].max():.6f}")
    except Exception as e:
        print(f"Error reading CSV: {e}")


def _cmd_cfd(args):
    """Set up a CFD case using the workflow engine."""
    from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig

    output = Path(args.output)
    cfg = CFDWorkflowConfig(
        solver=args.solver,
        inlet_velocity=args.velocity,
        turbulence_model=args.turbulence,
        is_rotating=args.rotating,
        omega=args.omega,
        n_procs=args.nprocs,
        compressible=args.compressible,
        total_pressure=args.total_pressure,
        total_temperature=args.total_temperature,
    )

    wf = CFDWorkflow(cfg)
    if args.mesh:
        wf.set_mesh(args.mesh)
    case = wf.setup_case(output)

    print(f"{args.solver.upper()} case created: {case}")
    print(f"  Solver:      {args.solver}")
    if args.compressible:
        print(f"  Mode:        compressible (rhoSimpleFoam)")
        print(f"  Total P:     {args.total_pressure} Pa")
        print(f"  Total T:     {args.total_temperature} K")
    else:
        print(f"  Velocity:    {args.velocity} m/s")
    print(f"  Turbulence:  {args.turbulence}")
    if args.rotating:
        print(f"  Rotating:    omega = {args.omega} rad/s")
    if args.mesh:
        print(f"  Mesh:        {args.mesh}")
    print(f"  Procs:       {args.nprocs}")

    if args.solver == "openfoam":
        print(f"\n  Next steps:")
        print(f"    cd {output} && bash Allrun")
    elif args.solver == "fluent":
        print(f"\n  Next steps:")
        print(f"    cd {output} && fluent 3ddp -i run.jou")
    elif args.solver == "cfx":
        print(f"\n  Next steps:")
        print(f"    cd {output} && bash run_cfx.sh")
    elif args.solver == "su2":
        print(f"\n  Next steps:")
        print(f"    cd {output} && bash run_su2.sh")


def _cmd_meanline(args):
    """Run meanline compressor design."""
    from astraturbo.design import meanline_compressor, meanline_to_blade_parameters
    from astraturbo.design.meanline import blade_angle_to_cl0
    import math

    result = meanline_compressor(
        overall_pressure_ratio=args.pr,
        mass_flow=args.mass_flow,
        rpm=args.rpm,
        r_hub=args.r_hub,
        r_tip=args.r_tip,
        n_stages=args.n_stages,
        eta_poly=args.eta,
        reaction=args.reaction,
        radial_stations=args.radial_stations,
    )

    print(result.summary())
    print()

    params = meanline_to_blade_parameters(result)
    print("Blade Parameters for AstraTurbo:")
    for p in params:
        print(f"  Stage {p['stage']}:")
        print(f"    Rotor:  stagger={p['rotor_stagger_deg']:.1f} deg, "
              f"camber={p['rotor_camber_deg']:.1f} deg, "
              f"solidity={p['rotor_solidity']:.2f}")
        print(f"    Stator: stagger={p['stator_stagger_deg']:.1f} deg, "
              f"camber={p['stator_camber_deg']:.1f} deg, "
              f"solidity={p['stator_solidity']:.2f}")
        print(f"    De Haller: {p['de_haller']:.3f}  "
              f"{'OK' if p['de_haller'] > 0.72 else 'WARNING: below 0.72'}")

    # Show auto-computed cl0 from blade angles
    print()
    print("Auto-computed parameters (for profile generation):")
    for i, (stage, bp) in enumerate(zip(result.stages, params)):
        cl0 = blade_angle_to_cl0(
            stage.rotor_inlet_beta, stage.rotor_outlet_beta, bp["rotor_solidity"]
        )
        print(f"  Stage {stage.stage_number}: cl0 = {cl0:.4f}")

    # Show radial blade angle distribution
    print()
    print("Radial blade angle distribution (free vortex):")
    for stage in result.stages:
        print(f"  Stage {stage.stage_number}:")
        print(f"    {'Radius (m)':>12s}  {'beta_in (deg)':>14s}  {'beta_out (deg)':>15s}  "
              f"{'alpha_in (deg)':>15s}  {'alpha_out (deg)':>16s}")
        for a in stage.radial_blade_angles:
            print(f"    {a['r']:12.4f}  {math.degrees(a['beta_in']):14.2f}  "
                  f"{math.degrees(a['beta_out']):15.2f}  "
                  f"{math.degrees(a['alpha_in']):15.2f}  "
                  f"{math.degrees(a['alpha_out']):16.2f}")

    # Off-design analysis
    if getattr(args, 'off_design', False):
        from astraturbo.design.off_design import off_design_compressor
        print()
        print("=" * 60)
        print("Off-Design Analysis (same conditions as design point):")
        print("=" * 60)
        od_result = off_design_compressor(
            result, mass_flow=args.mass_flow, rpm=args.rpm,
        )
        print(od_result.summary())

    # Compressor map generation
    if getattr(args, 'map', False):
        from astraturbo.design.compressor_map import generate_compressor_map
        print()
        print("=" * 60)
        print("Generating Compressor Map...")
        print("=" * 60)
        rpm_fracs = [float(x) for x in args.rpm_fractions.split(",")]
        cmap = generate_compressor_map(
            result, rpm_fractions=rpm_fracs,
        )
        print(cmap.summary())

        # Surge margin at design speed
        for sl in cmap.speed_lines:
            if abs(sl.rpm_fraction - 1.0) < 0.01 and sl.surge_point_index is not None:
                surge_pr = sl.pressure_ratios[sl.surge_point_index]
                design_pr = cmap.design_point.get("pr", 1.0)
                if design_pr > 1.0:
                    sm = (surge_pr - design_pr) / design_pr
                    print(f"\nSurge margin at design speed: {sm:.4f} ({sm*100:.1f}%)")

    # Report generation
    report_path = getattr(args, 'report', None)
    if report_path:
        from astraturbo.reports import generate_report, ReportConfig
        cfg = ReportConfig(
            title=f"Axial Compressor Design — PR {args.pr}",
            output_path=report_path,
        )
        # Collect what we have
        od_result = None
        cmap_result = None
        if getattr(args, 'off_design', False):
            # Re-run to capture result (already printed above)
            from astraturbo.design.off_design import off_design_compressor as _od
            od_result = _od(result, mass_flow=args.mass_flow, rpm=args.rpm)
        if getattr(args, 'map', False):
            from astraturbo.design.compressor_map import generate_compressor_map as _gm
            rpm_fracs = [float(x) for x in args.rpm_fractions.split(",")]
            cmap_result = _gm(result, rpm_fractions=rpm_fracs)
        path = generate_report(
            config=cfg,
            meanline_result=result,
            off_design_result=od_result,
            compressor_map=cmap_result,
            blade_params=params,
        )
        print(f"\nReport generated: {path}")


def _cmd_centrifugal(args):
    """Design a centrifugal compressor."""
    from astraturbo.design.centrifugal import centrifugal_compressor

    kwargs = {
        "pressure_ratio": args.pr,
        "mass_flow": args.mass_flow,
        "rpm": args.rpm,
        "r1_hub": args.r1_hub,
        "r1_tip": args.r1_tip,
        "beta2_blade_deg": args.backsweep,
        "n_blades": args.n_blades,
    }
    if args.r2 is not None:
        kwargs["r2"] = args.r2

    result = centrifugal_compressor(**kwargs)
    print(result.summary())

    # Report
    report_path = getattr(args, 'report', None)
    if report_path:
        from astraturbo.reports import generate_report, ReportConfig
        cfg = ReportConfig(
            title=f"Centrifugal Compressor — PR {args.pr}",
            output_path=report_path,
        )
        generate_report(config=cfg, centrifugal_result=result)
        print(f"\nReport generated: {report_path}")


def _cmd_turbine(args):
    """Run axial turbine meanline design."""
    from astraturbo.design.turbine import meanline_turbine, meanline_to_turbine_blade_parameters

    result = meanline_turbine(
        overall_expansion_ratio=args.expansion_ratio,
        mass_flow=args.mass_flow,
        rpm=args.rpm,
        r_hub=args.r_hub,
        r_tip=args.r_tip,
        n_stages=args.n_stages,
        eta_poly=args.eta,
        reaction=args.reaction,
        T_inlet=args.inlet_temp,
        P_inlet=args.inlet_pressure,
        radial_stations=args.radial_stations,
    )

    print(result.summary())
    print()

    params = meanline_to_turbine_blade_parameters(result)
    print("Turbine Blade Parameters:")
    for p in params:
        print(f"  Stage {p['stage']}:")
        print(f"    NGV:   stagger={p['ngv_stagger_deg']:.1f} deg, "
              f"camber={p['ngv_camber_deg']:.1f} deg, "
              f"solidity={p['ngv_solidity']:.2f}")
        print(f"    Rotor: stagger={p['rotor_stagger_deg']:.1f} deg, "
              f"camber={p['rotor_camber_deg']:.1f} deg, "
              f"solidity={p['rotor_solidity']:.2f}")
        print(f"    Zweifel = {p['zweifel']:.3f}")

    # Off-design analysis
    if getattr(args, 'off_design', False):
        from astraturbo.design.turbine_off_design import turbine_off_design
        print()
        print("=" * 60)
        print("Turbine Off-Design Analysis (same conditions as design point):")
        print("=" * 60)
        od_result = turbine_off_design(
            result, mass_flow=args.mass_flow, rpm=args.rpm,
        )
        print(od_result.summary())

    # Turbine map generation
    if getattr(args, 'map', False):
        from astraturbo.design.turbine_off_design import generate_turbine_map
        print()
        print("=" * 60)
        print("Generating Turbine Map...")
        print("=" * 60)
        rpm_fracs = [float(x) for x in args.rpm_fractions.split(",")]
        tmap = generate_turbine_map(
            result, rpm_fractions=rpm_fracs,
        )
        print(tmap.summary())

        # Choke margin at design speed
        for sl in tmap.speed_lines:
            if abs(sl.rpm_fraction - 1.0) < 0.01 and sl.choke_point_index is not None:
                choke_er = sl.expansion_ratios[sl.choke_point_index]
                design_er = tmap.design_point.get("er", 1.0)
                if design_er > 1.0:
                    cm = (choke_er - design_er) / design_er
                    print(f"\nChoke margin at design speed: {cm:.4f} ({cm*100:.1f}%)")

    # Report
    report_path = getattr(args, 'report', None)
    if report_path:
        from astraturbo.reports import generate_report, ReportConfig
        cfg = ReportConfig(
            title=f"Axial Turbine — ER {args.expansion_ratio}",
            output_path=report_path,
        )
        # Collect off-design results for report
        od_result = None
        tmap_result = None
        if getattr(args, 'off_design', False):
            from astraturbo.design.turbine_off_design import turbine_off_design as _tod
            od_result = _tod(result, mass_flow=args.mass_flow, rpm=args.rpm)
        if getattr(args, 'map', False):
            from astraturbo.design.turbine_off_design import generate_turbine_map as _gtm
            rpm_fracs = [float(x) for x in args.rpm_fractions.split(",")]
            tmap_result = _gtm(result, rpm_fractions=rpm_fracs)
        generate_report(
            config=cfg, turbine_result=result,
            turbine_off_design_result=od_result,
            turbine_map=tmap_result,
        )
        print(f"\nReport generated: {report_path}")


def _cmd_engine_cycle(args):
    """Run full engine cycle analysis."""
    from astraturbo.design.engine_cycle import engine_cycle

    result = engine_cycle(
        engine_type=args.engine_type,
        altitude=args.altitude,
        mach_flight=args.mach,
        overall_pressure_ratio=args.opr,
        turbine_inlet_temp=args.tit,
        mass_flow=args.mass_flow,
        rpm=args.rpm,
        r_hub=args.r_hub,
        r_tip=args.r_tip,
        compressor_type=args.compressor_type,
        n_spools=args.n_spools,
        hp_pressure_ratio=args.hp_pr,
        hp_rpm=args.hp_rpm,
        hp_r_hub=args.hp_r_hub,
        hp_r_tip=args.hp_r_tip,
    )

    print(result.summary())

    report_path = getattr(args, 'report', None)
    if report_path:
        from astraturbo.reports import generate_report, ReportConfig
        cfg = ReportConfig(
            title=f"Engine Cycle — {args.engine_type.upper()} OPR={args.opr} TIT={args.tit}K",
            output_path=report_path,
        )
        generate_report(config=cfg, engine_cycle_result=result)
        print(f"\nReport generated: {report_path}")


def _cmd_blade(args):
    """Build a 3D blade from hub-to-tip profiles."""
    import numpy as np
    from astraturbo.camberline import NACA65
    from astraturbo.thickness import NACA65Series
    from astraturbo.profile import Superposition
    from astraturbo.blade import BladeRow

    profiles = [
        Superposition(NACA65(cl0=args.cl0_hub), NACA65Series(max_thickness=args.thickness_hub)),
        Superposition(NACA65(cl0=args.cl0_mid), NACA65Series(max_thickness=args.thickness_mid)),
        Superposition(NACA65(cl0=args.cl0_tip), NACA65Series(max_thickness=args.thickness_tip)),
    ]

    hub_pts = np.array([[0.0, args.r_hub], [args.axial_chord, args.r_hub]])
    shroud_pts = np.array([[0.0, args.r_tip], [args.axial_chord, args.r_tip]])

    row = BladeRow(hub_points=hub_pts, shroud_points=shroud_pts)
    row.number_blades = args.n_blades
    for p in profiles:
        row.add_profile(p)

    staggers = np.deg2rad([args.stagger_hub, args.stagger_mid, args.stagger_tip])
    chords = np.array([args.axial_chord * 0.8, args.axial_chord, args.axial_chord * 1.2])
    row.compute(stagger_angles=staggers, chord_lengths=chords)

    print(f"3D Blade Built:")
    print(f"  Blades: {args.n_blades}")
    print(f"  Hub: {args.r_hub:.4f} m, Tip: {args.r_tip:.4f} m")
    print(f"  CL0: {args.cl0_hub:.2f} / {args.cl0_mid:.2f} / {args.cl0_tip:.2f}")
    print(f"  Thickness: {args.thickness_hub:.0%} / {args.thickness_mid:.0%} / {args.thickness_tip:.0%}")
    print(f"  Stagger: {args.stagger_hub:.1f} / {args.stagger_mid:.1f} / {args.stagger_tip:.1f} deg")
    if row.leading_edge is not None:
        print(f"  Leading edge: {row.leading_edge.shape}")
    if row.trailing_edge is not None:
        print(f"  Trailing edge: {row.trailing_edge.shape}")
    print(f"  3D profiles: {len(row.profiles_3d) if row.profiles_3d else 0} sections")

    if args.output:
        from astraturbo.mesh.multiblock import generate_blade_passage_mesh
        mid_profile = profiles[1].as_array()
        pitch = 2 * np.pi * (args.r_hub + args.r_tip) / 2.0 / args.n_blades
        mesh = generate_blade_passage_mesh(
            profile=mid_profile, pitch=pitch,
            n_blade=30, n_ogrid=8, n_inlet=10, n_outlet=10, n_passage=15,
        )
        mesh.export_cgns(args.output)
        print(f"\n  Mesh exported to {args.output}: {mesh.total_cells} cells")


def _cmd_pipeline(args):
    """Run the full design pipeline."""
    from astraturbo.foundation.design_chain import DesignChain

    chain = DesignChain()
    params = {
        "pressure_ratio": args.pr,
        "mass_flow": args.mass_flow,
        "rpm": args.rpm,
    }
    if args.cl0 is not None:
        params["cl0"] = args.cl0
    if args.cfd_output:
        params["cfd_output"] = args.cfd_output
    if args.compressible:
        params["cfd_compressible"] = True

    print(f"Running full design pipeline...")
    print(f"  PR={args.pr}, mass_flow={args.mass_flow} kg/s, RPM={args.rpm}")
    print()

    result = chain.set_parameters(params)

    if result is None:
        print("ERROR: Design chain returned no result")
        sys.exit(1)

    print(f"Pipeline: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Total time: {result.total_time:.3f}s\n")
    for stage in result.stages:
        status = "OK" if stage.success else f"FAIL: {stage.error}"
        print(f"  {stage.stage_name:12s}  {stage.elapsed_time:.3f}s  {status}")
        if stage.data:
            for key, val in stage.data.items():
                if isinstance(val, (int, float, str, bool)):
                    print(f"    {key}: {val}")

    if not result.success:
        sys.exit(1)


def _cmd_fea(args):
    """Set up FEA structural analysis."""
    from astraturbo.fea import (
        FEAWorkflow, FEAWorkflowConfig,
        get_material, list_materials,
    )

    if args.list_materials:
        from astraturbo.fea.material import list_categories
        print("Available materials (32):\n")
        for cat in list_categories():
            cat_materials = list_materials(category=cat)
            print(f"  [{cat.upper()}]")
            for name in cat_materials:
                mat = get_material(name)
                temp_marker = " [T(K)]" if mat.youngs_modulus_table else ""
                print(f"    {name:20s}  E={mat.youngs_modulus/1e9:>6.0f} GPa, "
                      f"yield={mat.yield_strength/1e6:>5.0f} MPa, "
                      f"Tmax={mat.max_service_temperature:>5.0f} K  "
                      f"({mat.description}){temp_marker}")
            print()
        print("  [T(K)] = temperature-dependent data available")
        return

    try:
        material = get_material(args.material)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Show temperature-dependent properties if temperature specified
    op_temp = getattr(args, 'temperature', None)
    if op_temp is not None:
        props = material.properties_at(op_temp)
        print(f"Material: {material.name} at {op_temp:.0f} K")
        print(f"  E  = {props['youngs_modulus_GPa']:.1f} GPa "
              f"(room temp: {material.youngs_modulus/1e9:.1f} GPa, "
              f"ratio: {props['youngs_modulus_Pa']/material.youngs_modulus:.2f})")
        print(f"  Sy = {props['yield_strength_MPa']:.0f} MPa "
              f"(room temp: {material.yield_strength/1e6:.0f} MPa, "
              f"ratio: {props['yield_strength_Pa']/material.yield_strength:.2f})")
        print(f"  k  = {props['thermal_conductivity_W_mK']:.1f} W/m-K "
              f"(room temp: {material.thermal_conductivity:.1f} W/m-K)")
        if not props['has_temp_data']:
            print("  WARNING: No temperature-dependent data — using room-temp values")
        print()

    output = Path(args.output)
    cfg = FEAWorkflowConfig(
        material=material,
        omega=args.omega,
        blade_thickness=args.thickness,
        analysis_type=args.analysis,
    )

    fea = FEAWorkflow(cfg)

    if args.surface:
        surface_data = np.loadtxt(args.surface, delimiter=",", skiprows=1)
        if surface_data.shape[1] < 3:
            print("ERROR: Surface file must have 3 columns (x, y, z)")
            sys.exit(1)
        fea.set_blade_surface(surface_data[:, :3], args.ni, args.nj)
        case = fea.setup(output)

        print(f"FEA case created: {case}")
        print(f"  Material:    {material.name}")
        print(f"  Analysis:    {args.analysis}")
        print(f"  Omega:       {args.omega} rad/s")
        print(f"  Thickness:   {args.thickness*1000:.1f} mm")
        print(f"  Input file:  {case / 'blade.inp'}")

        # Quick stress estimate
        estimate = fea.estimate_stress_analytical()
        yield_at_T = material.yield_strength_at(op_temp) if op_temp else material.yield_strength
        safety_at_T = yield_at_T / (estimate['centrifugal_stress_MPa'] * 1e6) if estimate['centrifugal_stress_MPa'] > 0 else float('inf')
        print(f"\n  Analytical stress estimate:")
        print(f"    Centrifugal: {estimate['centrifugal_stress_MPa']:.1f} MPa")
        if op_temp and material.yield_strength_table:
            print(f"    Yield at {op_temp:.0f} K: {yield_at_T/1e6:.0f} MPa "
                  f"(room temp: {material.yield_strength/1e6:.0f} MPa)")
            print(f"    Safety (at temp): {safety_at_T:.2f}")
            print(f"    Safety (room):    {estimate['safety_factor']:.2f}")
        else:
            print(f"    Yield:       {material.yield_strength/1e6:.0f} MPa")
            print(f"    Safety:      {estimate['safety_factor']:.2f}")

        print(f"\n  Next steps:")
        print(f"    cd {output} && bash run_fea.sh  (requires CalculiX)")
    else:
        print("No --surface file provided. A blade surface CSV is required for FEA setup.")
        print("Example:")
        print(f"  python -m astraturbo fea --material {args.material} "
              f"--omega {args.omega} --surface blade_surface.csv -o fea_case")
        print("\nUse --list-materials to see available material options.")


def _cmd_optimize(args):
    """Run blade design optimization."""
    from astraturbo.optimization import (
        Optimizer, OptimizationConfig, create_blade_design_space,
    )
    from astraturbo.mesh.multiblock import generate_blade_passage_mesh

    if not Path(args.profile).exists():
        print(f"ERROR: Profile file not found: {args.profile}")
        sys.exit(1)
    profile = np.loadtxt(args.profile, delimiter=",", skiprows=1)[:, :2]
    print(f"Base profile: {len(profile)} points from {args.profile}")
    print(f"Running optimization: {args.generations} generations, population {args.population}")

    design_space = create_blade_design_space(n_profiles=args.n_profiles)

    def evaluate(x):
        # Simplified objective: minimize negative of a smoothness metric
        params = design_space.decode(x)
        # Use parameter variation as proxy for design quality
        penalty = sum((v - 0.5)**2 for v in x / (design_space.upper_bounds - design_space.lower_bounds + 1e-10))
        return np.array([-1.0 + penalty]), np.array([])

    optimizer = Optimizer(design_space, evaluate, n_objectives=1)
    result = optimizer.run(OptimizationConfig(
        n_generations=args.generations,
        population_size=args.population,
    ))

    print(f"\nOptimization complete:")
    print(f"  Evaluations: {result.n_evaluations}")
    print(f"  Best objective: {result.best_f}")
    if result.best_x is not None:
        print(f"  Best design: {result.best_x[:5]}...")
        if args.output:
            np.savetxt(args.output, result.best_x, delimiter=",")
            print(f"  Saved to: {args.output}")


def _cmd_yplus(args):
    """y+ calculator."""
    from astraturbo.mesh import estimate_yplus, first_cell_height_for_yplus

    if args.cell_height is not None:
        yp = estimate_yplus(
            args.cell_height, args.density, args.velocity,
            args.viscosity, args.chord,
        )
        print(f"y+ Calculator")
        print(f"  Velocity:    {args.velocity} m/s")
        print(f"  Chord:       {args.chord*1000:.1f} mm")
        print(f"  Cell height: {args.cell_height*1000:.4f} mm")
        print(f"  Estimated y+: {yp:.2f}")
    else:
        dy = first_cell_height_for_yplus(
            args.target_yplus, args.density, args.velocity,
            args.viscosity, args.chord,
        )
        yp_check = estimate_yplus(
            dy, args.density, args.velocity, args.viscosity, args.chord,
        )
        print(f"y+ Calculator")
        print(f"  Velocity:       {args.velocity} m/s")
        print(f"  Chord:          {args.chord*1000:.1f} mm")
        print(f"  Target y+:      {args.target_yplus}")
        print(f"  Required cell:  {dy*1e6:.2f} um ({dy*1000:.4f} mm)")
        print(f"  Verification:   y+ = {yp_check:.2f}")


def _cmd_formats():
    """List all supported file formats."""
    from astraturbo.export import list_supported_formats

    fmts = list_supported_formats()
    print(f"Supported file formats: {len(fmts)}\n")
    for name, info in sorted(fmts.items()):
        rw = ("R" if info["read"] else "-") + ("W" if info["write"] else "-")
        exts = ", ".join(info["extensions"])
        print(f"  [{rw}] {name:22s} {exts:28s} {info['description']}")


def _cmd_material_advisor(args):
    """Recommend materials for engine components."""
    from astraturbo.fea.material import (
        recommend_material, recommend_engine_materials,
        COMPONENT_MATERIAL_MAP,
    )

    if args.engine:
        # Full engine material map
        recs = recommend_engine_materials(
            t_fan=args.t_fan,
            t_compressor=args.t_compressor,
            t_combustor=args.t_combustor,
            t_turbine=args.t_turbine,
            t_nozzle=args.t_nozzle,
        )
        print("Engine Material Map")
        print("=" * 80)
        print(f"  Fan inlet:    {args.t_fan:.0f} K")
        print(f"  Compressor:   {args.t_compressor:.0f} K")
        print(f"  Combustor:    {args.t_combustor:.0f} K")
        print(f"  Turbine:      {args.t_turbine:.0f} K")
        print(f"  Nozzle:       {args.t_nozzle:.0f} K")
        print()

        for comp, rec in recs.items():
            mat = rec.primary_material
            margin = mat.max_service_temperature - rec.operating_temperature
            coats = ", ".join(c[1].name for c in rec.coatings) if rec.coatings else "none"
            print(f"  {comp:22s}  {rec.primary_key:16s}  "
                  f"Tmax={mat.max_service_temperature:>5.0f} K  "
                  f"margin={margin:>+4.0f} K  "
                  f"coatings: {coats}")
            for w in rec.warnings:
                print(f"    WARNING: {w}")

        print(f"\n{'=' * 80}")
        print("Temperature transitions:")
        print("  [Fan] Ti --> [Compressor] Ti/Ni --> [Combustor] Ni/Co "
              "--> [Turbine] SC Ni/CMC --> [Nozzle] Ni")

    elif args.component and args.temperature:
        rec = recommend_material(args.component, args.temperature, args.stress)
        mat = rec.primary_material
        margin = mat.max_service_temperature - rec.operating_temperature

        print(f"Material Recommendation")
        print(f"  Component:    {rec.component}")
        print(f"  Temperature:  {rec.operating_temperature:.0f} K")
        if args.stress > 0:
            print(f"  Stress:       {args.stress:.0f} MPa")
        print()
        print(f"  Primary:      {rec.primary_key}")
        print(f"    {mat.name}: {mat.category}")
        print(f"    E={mat.youngs_modulus/1e9:.0f} GPa, "
              f"yield={mat.yield_strength/1e6:.0f} MPa, "
              f"Tmax={mat.max_service_temperature:.0f} K")
        print(f"    Margin: {margin:+.0f} K")
        print(f"    {mat.description}")

        if rec.alternatives:
            print(f"\n  Alternatives:")
            for key, alt in rec.alternatives:
                print(f"    {key:20s}  Tmax={alt.max_service_temperature:.0f} K  "
                      f"({alt.description})")

        if rec.coatings:
            print(f"\n  Recommended coatings:")
            for key, coat in rec.coatings:
                print(f"    {key:20s}  k={coat.thermal_conductivity:.1f} W/mK  "
                      f"({coat.description})")

        for w in rec.warnings:
            print(f"\n  WARNING: {w}")

    else:
        print("Usage:")
        print("  astraturbo material-advisor --engine")
        print("  astraturbo material-advisor --engine --t-turbine 1500")
        print("  astraturbo material-advisor --component turbine_blade --temperature 1350")
        print("  astraturbo material-advisor --component shaft --temperature 700 --stress 500")
        print()
        print("Available components:")
        for comp in sorted(COMPONENT_MATERIAL_MAP.keys()):
            print(f"  {comp}")
        sys.exit(1)


def _cmd_multistage(args):
    """Generate multi-stage mesh."""
    from astraturbo.mesh.multistage import MultistageGenerator, RowMeshConfig

    if len(args.profiles) != len(args.pitches):
        print(f"ERROR: Number of profiles ({len(args.profiles)}) must match pitches ({len(args.pitches)})")
        sys.exit(1)

    names = args.names or [f"row{i}" for i in range(len(args.profiles))]
    if len(names) != len(args.profiles):
        print(f"ERROR: Number of names ({len(names)}) must match profiles ({len(args.profiles)})")
        sys.exit(1)

    gen = MultistageGenerator()
    for name, prof_path, pitch in zip(names, args.profiles, args.pitches):
        if not Path(prof_path).exists():
            print(f"ERROR: Profile file not found: {prof_path}")
            sys.exit(1)
        profile = np.loadtxt(prof_path, delimiter=",", skiprows=1)[:, :2]
        print(f"  {name}: {len(profile)} points, pitch={pitch}")
        gen.add_row(name, RowMeshConfig(
            profile=profile, pitch=pitch,
            n_blade=20, n_ogrid=5, n_inlet=8, n_outlet=8, n_passage=10,
        ))

    result = gen.generate()
    output = Path(args.output)
    if not output.suffix:
        output = output.with_suffix(".cgns")
    result.export_cgns(output)

    print(f"\nMulti-stage mesh: {result.n_rows} rows, {result.total_cells} cells")
    print(f"Exported: {output}")


def _cmd_run(args):
    """Execute a CFD or FEA solver."""
    case = Path(args.case)

    if not case.exists():
        print(f"ERROR: Case directory not found: {case}")
        sys.exit(1)

    if args.solver == "openfoam":
        from astraturbo.cfd.runner import run_openfoam, RunConfig
        cfg = RunConfig(solver="simpleFoam", case_dir=str(case), n_procs=args.nprocs)
        print(f"Running OpenFOAM in {case}...")
        result = run_openfoam(cfg)
    elif args.solver == "su2":
        from astraturbo.cfd.runner import run_su2
        cfg_file = case / "astraturbo.cfg"
        if not cfg_file.exists():
            print(f"ERROR: SU2 config not found: {cfg_file}")
            sys.exit(1)
        print(f"Running SU2 in {case}...")
        result = run_su2(cfg_file, args.nprocs)
    elif args.solver == "calculix":
        import subprocess
        inp_file = case / "blade.inp"
        if not inp_file.exists():
            print(f"ERROR: CalculiX input not found: {inp_file}")
            sys.exit(1)
        print(f"Running CalculiX in {case}...")
        try:
            proc = subprocess.run(
                ["ccx", "blade"], cwd=str(case),
                capture_output=True, text=True, timeout=3600,
            )
            if proc.returncode == 0:
                print("CalculiX completed successfully.")
            else:
                print(f"CalculiX failed (exit code {proc.returncode})")
                print(proc.stderr[:500] if proc.stderr else "")
        except FileNotFoundError:
            from astraturbo.cfd.runner import _solver_install_hint
            hint = _solver_install_hint("ccx")
            print(f"ERROR: CalculiX (ccx) not found in PATH.\n\n{hint}")
            sys.exit(1)
        return  # CalculiX handled above

    if hasattr(result, "success"):  # noqa: F821 — result set in openfoam/su2 branches
        if result.success:
            print(f"Solver completed successfully.")
            if result.log_file:
                print(f"Log: {result.log_file}")
        else:
            print(f"Solver failed: {result.error_message}")
            sys.exit(1)


def _cmd_throughflow(args):
    """Run the throughflow (S2m) solver."""
    from astraturbo.solver.throughflow import (
        ThroughflowSolver, ThroughflowConfig, BladeRowSpec,
    )

    n_stations = args.n_streamwise
    n_streamlines = args.n_radial + 1  # +1 because n_radial is cells

    config = ThroughflowConfig(
        n_stations=n_stations,
        n_streamlines=n_streamlines,
        max_iterations=200,
    )
    solver = ThroughflowSolver(config)

    # Set up annulus
    hub_r = np.full(n_stations, args.r_hub)
    tip_r = np.full(n_stations, args.r_tip)
    axial = np.linspace(0.0, 0.2, n_stations)
    solver.set_annulus(hub_r, tip_r, axial)

    # Add a single rotor row
    omega = args.rpm * 2.0 * np.pi / 60.0
    rotor = BladeRowSpec(
        row_type="rotor",
        n_blades=36,
        inlet_station=n_stations // 4,
        outlet_station=n_stations // 2,
        omega=omega,
    )
    solver.add_blade_row(rotor)

    # Set inlet conditions
    solver.set_inlet_conditions(total_pressure=101325.0, total_temperature=288.15)

    print(f"Running throughflow solver...")
    print(f"  Pressure ratio target: {args.pr}")
    print(f"  Mass flow: {args.mass_flow} kg/s")
    print(f"  RPM: {args.rpm}")
    print(f"  Hub radius: {args.r_hub} m")
    print(f"  Tip radius: {args.r_tip} m")
    print(f"  Stations: {n_stations}")
    print(f"  Streamlines: {n_streamlines}")

    result = solver.solve()

    print(f"\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    if result.total_pressure is not None and result.total_temperature is not None:
        pr_actual = result.total_pressure[-1, :].mean() / result.total_pressure[0, :].mean()
        tr_actual = result.total_temperature[-1, :].mean() / result.total_temperature[0, :].mean()
        print(f"  Pressure ratio (outlet/inlet): {pr_actual:.4f}")
        print(f"  Temperature ratio: {tr_actual:.4f}")
    if result.mach_number is not None:
        print(f"  Max Mach number: {result.mach_number.max():.4f}")
    if result.residual_history:
        print(f"  Final residual: {result.residual_history[-1]:.2e}")


def _cmd_smooth(args):
    """Apply Laplacian smoothing to a mesh file."""
    from astraturbo.mesh.smoothing import laplacian_smooth

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Load mesh from CGNS
    try:
        import h5py
        with h5py.File(input_path, "r") as f:
            # Try to find a zone with grid coordinates
            print(f"Loading mesh from: {input_path}")
            # Navigate CGNS structure
            block = None
            for base_name in f.keys():
                base = f[base_name]
                if isinstance(base, h5py.Group):
                    for zone_name in base.keys():
                        zone = base[zone_name]
                        if isinstance(zone, h5py.Group):
                            for gc_name in zone.keys():
                                gc = zone[gc_name]
                                if isinstance(gc, h5py.Group):
                                    if "CoordinateX" in gc and "CoordinateY" in gc:
                                        # CGNS stores coordinates as groups
                                        # with a " data" child dataset
                                        cx_node = gc["CoordinateX"]
                                        cy_node = gc["CoordinateY"]
                                        if isinstance(cx_node, h5py.Group):
                                            cx = cx_node[" data"][:].squeeze()
                                            cy = cy_node[" data"][:].squeeze()
                                        else:
                                            cx = cx_node[:].squeeze()
                                            cy = cy_node[:].squeeze()
                                        if cx.ndim == 2:
                                            ni, nj = cx.shape
                                            block = np.zeros((ni, nj, 2))
                                            block[:, :, 0] = cx
                                            block[:, :, 1] = cy
                                            break
                        if block is not None:
                            break
                if block is not None:
                    break

            if block is None:
                print("ERROR: Could not find 2D coordinate data in CGNS file.")
                sys.exit(1)
    except ImportError:
        print("ERROR: h5py is required to read CGNS files.")
        print("Install with: pip install h5py")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load mesh: {e}")
        sys.exit(1)

    print(f"Mesh loaded: {block.shape[0]}x{block.shape[1]} points")

    # Run smoothing
    smoothed, metrics = laplacian_smooth(block, n_iterations=args.iterations)

    print(f"\nSmoothing results ({args.iterations} iterations):")
    print(f"  Before: aspect_ratio_max={metrics['before_aspect_ratio_max']:.3f}, "
          f"skewness_max={metrics['before_skewness_max']:.3f}")
    print(f"  After:  aspect_ratio_max={metrics['after_aspect_ratio_max']:.3f}, "
          f"skewness_max={metrics['after_skewness_max']:.3f}")

    # Export smoothed mesh
    output_path = Path(args.output)
    try:
        from astraturbo.export import write_cgns_2d
        write_cgns_2d(output_path, smoothed)
        print(f"  Saved smoothed mesh to: {output_path}")
    except Exception as e:
        print(f"  Warning: Could not save to CGNS: {e}")
        # Fallback: save as numpy
        np_path = output_path.with_suffix(".npy")
        np.save(np_path, smoothed)
        print(f"  Saved as numpy: {np_path}")


def _cmd_database(args):
    """Manage the design database."""
    import json as _json
    from astraturbo.database.design_db import DesignDatabase

    db = DesignDatabase()

    if args.db_command == "list":
        designs = db.list_designs()
        if not designs:
            print("No designs in database.")
        else:
            print(f"Designs ({len(designs)}):")
            for d in designs:
                print(f"  [{d['id']}] {d['name']}  "
                      f"created={d['created_at']}  tags={','.join(d['tags']) if d['tags'] else 'none'}")
    elif args.db_command == "save":
        try:
            params = _json.loads(args.params)
        except _json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON for --params: {e}")
            sys.exit(1)
        design_id = db.save_design(name=args.name, parameters=params)
        print(f"Design saved: ID={design_id}, name='{args.name}'")
    elif args.db_command == "export":
        count = db.export_csv(args.filepath)
        print(f"Exported {count} designs to {args.filepath}")
    else:
        print("Usage: astraturbo database {list|save|export}")
        sys.exit(1)

    db.close()


def _cmd_hpc(args):
    """HPC job management."""
    from astraturbo.hpc.job_manager import HPCJobManager, HPCConfig, JobStatus

    # Job registry file for persisting job state across CLI invocations
    job_registry = Path.home() / ".astraturbo" / "jobs.json"

    def _load_registry() -> dict:
        if job_registry.exists():
            import json
            try:
                return json.loads(job_registry.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_registry(registry: dict) -> None:
        import json
        job_registry.parent.mkdir(parents=True, exist_ok=True)
        job_registry.write_text(json.dumps(registry, indent=2))

    def _config_from_registry(info: dict) -> HPCConfig:
        """Reconstruct HPCConfig from persisted job registry entry."""
        return HPCConfig(
            backend=info.get("backend", "local"),
            host=info.get("host", ""),
            user=info.get("user", ""),
            ssh_key=info.get("ssh_key", ""),
            aws_region=info.get("aws_region", "us-east-1"),
            aws_job_queue=info.get("aws_job_queue", ""),
            aws_job_definition=info.get("aws_job_definition", ""),
            aws_s3_bucket=info.get("aws_s3_bucket", ""),
            aws_container_image=info.get("aws_container_image", ""),
        )

    if args.hpc_command == "submit":
        config = HPCConfig(
            backend=args.backend,
            max_nodes=args.nodes,
            walltime=args.walltime,
            host=args.host,
            user=args.user,
            ssh_key=args.ssh_key,
            aws_region=args.aws_region,
            aws_job_queue=args.aws_job_queue,
            aws_job_definition=args.aws_job_definition,
            aws_s3_bucket=args.aws_s3_bucket,
            aws_container_image=args.aws_container_image,
        )
        manager = HPCJobManager(config)
        try:
            job_id = manager.submit_job(
                case_dir=args.case,
                solver=args.solver,
                n_procs=args.nprocs,
                walltime=args.walltime,
            )
            # Persist job info for later status/cancel/download calls
            registry = _load_registry()
            registry[job_id] = {
                "backend": args.backend,
                "case": str(Path(args.case).resolve()),
                "solver": args.solver,
                "nprocs": args.nprocs,
                "host": args.host,
                "user": args.user,
                "ssh_key": args.ssh_key,
                "aws_region": args.aws_region,
                "aws_job_queue": args.aws_job_queue,
                "aws_job_definition": args.aws_job_definition,
                "aws_s3_bucket": args.aws_s3_bucket,
                "aws_container_image": args.aws_container_image,
            }
            _save_registry(registry)

            print(f"Job submitted: {job_id}")
            print(f"  Backend:  {args.backend}")
            print(f"  Solver:   {args.solver}")
            print(f"  Procs:    {args.nprocs}")
            print(f"  Nodes:    {args.nodes}")
            print(f"  Walltime: {args.walltime}")
            print(f"  Case:     {args.case}")
            if args.backend == "aws":
                print(f"  Region:   {args.aws_region}")
                print(f"  S3:       {args.aws_s3_bucket}")
        except Exception as e:
            print(f"ERROR: Job submission failed: {e}")
            sys.exit(1)

    elif args.hpc_command == "status":
        registry = _load_registry()
        info = registry.get(args.job_id, {})
        config = _config_from_registry(info)
        manager = HPCJobManager(config)
        status = manager.check_status(args.job_id)
        print(f"Job {args.job_id}: {status.value}")
        if info:
            print(f"  Backend: {info.get('backend')}")
            print(f"  Case:    {info.get('case')}")
            print(f"  Solver:  {info.get('solver')}")

    elif args.hpc_command == "cancel":
        registry = _load_registry()
        info = registry.get(args.job_id, {})
        config = _config_from_registry(info)
        manager = HPCJobManager(config)
        success = manager.cancel_job(args.job_id)
        if success:
            print(f"Job {args.job_id} cancelled.")
        else:
            print(f"ERROR: Could not cancel job {args.job_id}")
            sys.exit(1)

    elif args.hpc_command == "download":
        registry = _load_registry()
        info = registry.get(args.job_id, {})
        config = _config_from_registry(info)
        manager = HPCJobManager(config)
        success = manager.download_results(args.job_id, args.output_dir)
        if success:
            print(f"Results downloaded to: {args.output_dir}")
        else:
            print(f"ERROR: Could not download results for job {args.job_id}")
            sys.exit(1)

    elif args.hpc_command == "setup-aws":
        from astraturbo.hpc.aws_setup import AWSBatchProvisioner
        try:
            provisioner = AWSBatchProvisioner(
                region=args.region,
                platform=args.platform,
                max_vcpus=args.max_vcpus,
                bucket_name=args.bucket_name,
            )
            provisioner.setup()
        except Exception as e:
            print(f"ERROR: AWS setup failed: {e}")
            sys.exit(1)

    elif args.hpc_command == "teardown-aws":
        from astraturbo.hpc.aws_setup import AWSBatchProvisioner
        try:
            provisioner = AWSBatchProvisioner(region=args.region)
            provisioner.teardown()
        except Exception as e:
            print(f"ERROR: AWS teardown failed: {e}")
            sys.exit(1)

    else:
        print("Usage: astraturbo hpc {submit|status|cancel|download|setup-aws|teardown-aws}")
        sys.exit(1)


def _cmd_sweep(args):
    """Run a parametric sweep."""
    from astraturbo.foundation.design_chain import DesignChain

    chain = DesignChain()
    print(f"Running parametric sweep:")
    print(f"  Parameter: {args.parameter}")
    print(f"  Range: {args.start} to {args.end}")
    print(f"  Steps: {args.steps}")

    results = chain.sweep(args.parameter, start=args.start, end=args.end, steps=args.steps)

    values = np.linspace(args.start, args.end, args.steps)
    print(f"\nSweep Results ({len(results)} evaluations):")
    for i, (val, res) in enumerate(zip(values, results)):
        status = "OK" if res.success else "FAILED"
        stage_info = ", ".join(
            f"{s.stage_name}={s.elapsed_time:.3f}s" for s in res.stages
        )
        print(f"  [{i+1}] {args.parameter}={val:.4f} -> {status} ({stage_info})")

    n_success = sum(1 for r in results if r.success)
    print(f"\nSummary: {n_success}/{len(results)} successful")


if __name__ == "__main__":
    main()
