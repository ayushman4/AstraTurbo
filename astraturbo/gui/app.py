"""AstraTurbo GUI application entry point.

Launches the PySide6-based graphical interface.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Launch the AstraTurbo GUI."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is required for the GUI. Install with:")
        print("  pip install astraturbo[gui]")
        sys.exit(1)

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AstraTurbo")
    app.setOrganizationName("AstraTurbo")
    app.setApplicationVersion("0.1.0")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
