"""Turbomachine container — the top-level domain object.

Ported from V1 turboMachine.py. A TurboMachine holds one or more
BladeRows and orchestrates the full computation pipeline.
"""

from __future__ import annotations

from ..baseclass import Node
from ..foundation import NumericProperty, StringProperty
from ..blade import BladeRow


class TurboMachine(Node):
    """Top-level container for a turbomachine.

    Holds blade rows and machine-level parameters.

    Properties:
        machine_type: 'axial', 'radial', or 'mixed'.
    """

    machine_type = StringProperty(default="axial")

    def __init__(self) -> None:
        super().__init__()
        self.name = "TurboMachine"
        self._blade_rows: list[BladeRow] = []

    @property
    def blade_rows(self) -> list[BladeRow]:
        return self._blade_rows

    def add_blade_row(self, row: BladeRow) -> None:
        """Add a blade row to the machine."""
        self._blade_rows.append(row)

    def remove_blade_row(self, index: int) -> BladeRow:
        """Remove and return a blade row by index."""
        return self._blade_rows.pop(index)

    def compute_all(self, **kwargs) -> None:
        """Compute 3D geometry for all blade rows."""
        for row in self._blade_rows:
            row.compute(**kwargs)

    @property
    def n_stages(self) -> int:
        """Return estimated number of stages (pairs of rows)."""
        return max(1, len(self._blade_rows) // 2)
