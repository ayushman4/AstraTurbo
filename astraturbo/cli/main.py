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
    from astraturbo.mesh.multiblock import generate_blade_passage_mesh

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

    # Generate mesh
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

    # Export
    output = Path(args.output)
    if args.format == "cgns":
        if not output.suffix:
            output = output.with_suffix(".cgns")
        mesh.export_cgns(output)
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
    )

    wf = CFDWorkflow(cfg)
    if args.mesh:
        wf.set_mesh(args.mesh)
    case = wf.setup_case(output)

    print(f"{args.solver.upper()} case created: {case}")
    print(f"  Solver:      {args.solver}")
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

    result = meanline_compressor(
        overall_pressure_ratio=args.pr,
        mass_flow=args.mass_flow,
        rpm=args.rpm,
        r_hub=args.r_hub,
        r_tip=args.r_tip,
        n_stages=args.n_stages,
        eta_poly=args.eta,
        reaction=args.reaction,
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


def _cmd_fea(args):
    """Set up FEA structural analysis."""
    from astraturbo.fea import (
        FEAWorkflow, FEAWorkflowConfig,
        get_material, list_materials,
    )

    if args.list_materials:
        print("Available materials:")
        for name in list_materials():
            mat = get_material(name)
            print(f"  {name:20s}  E={mat.youngs_modulus/1e9:.0f} GPa, "
                  f"yield={mat.yield_strength/1e6:.0f} MPa, "
                  f"Tmax={mat.max_service_temperature:.0f} K")
        return

    try:
        material = get_material(args.material)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

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
        print(f"\n  Analytical stress estimate:")
        print(f"    Centrifugal: {estimate['centrifugal_stress_MPa']:.1f} MPa")
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
            print("ERROR: CalculiX (ccx) not found in PATH. Install CalculiX first.")
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
