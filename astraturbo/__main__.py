"""Entry point for `python -m astraturbo`."""

import sys


def main():
    """Launch AstraTurbo CLI or GUI depending on arguments."""
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        try:
            from astraturbo.gui.app import main as gui_main
            gui_main()
        except ImportError:
            print("GUI dependencies not installed. Run: pip install astraturbo[gui]")
            sys.exit(1)
    else:
        from astraturbo.cli.main import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
