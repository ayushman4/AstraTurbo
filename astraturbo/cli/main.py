"""CLI entry point for AstraTurbo."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="astraturbo",
        description="AstraTurbo — Turbomachinery Design & Simulation Platform",
    )
    parser.add_argument("--version", action="version", version="astraturbo 0.1.0")

    subparsers = parser.add_subparsers(dest="command")

    # gui subcommand
    subparsers.add_parser("gui", help="Launch the graphical interface")

    # compute subcommand
    compute_parser = subparsers.add_parser("compute", help="Compute blade geometry")
    compute_parser.add_argument("project", help="Path to project YAML file")

    # mesh subcommand
    mesh_parser = subparsers.add_parser("mesh", help="Generate mesh")
    mesh_parser.add_argument("project", help="Path to project YAML file")
    mesh_parser.add_argument(
        "--format", choices=["cgns", "openfoam", "cgx", "vtk"], default="cgns",
        help="Output mesh format",
    )

    # export subcommand
    export_parser = subparsers.add_parser("export", help="Export geometry")
    export_parser.add_argument("project", help="Path to project YAML file")
    export_parser.add_argument(
        "--format", choices=["step", "iges", "stl"], default="step",
        help="Output CAD format",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "gui":
        try:
            from astraturbo.gui.app import main as gui_main
            gui_main()
        except ImportError:
            print("GUI dependencies not installed. Run: pip install astraturbo[gui]")
            sys.exit(1)
    elif args.command == "compute":
        print(f"Computing blade geometry from {args.project}...")
        # TODO: implement
    elif args.command == "mesh":
        print(f"Generating {args.format} mesh from {args.project}...")
        # TODO: implement
    elif args.command == "export":
        print(f"Exporting {args.format} from {args.project}...")
        # TODO: implement


if __name__ == "__main__":
    main()
