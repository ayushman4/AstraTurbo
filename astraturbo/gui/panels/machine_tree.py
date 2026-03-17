"""Machine structure tree panel.

Displays the turbomachine hierarchy (machine → rows → profiles)
in a tree view for navigation.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ...machine import TurboMachine


class MachineTreePanel(QWidget):
    """Tree view showing the turbomachine hierarchy."""

    item_selected = Signal(object)

    def __init__(self, machine: TurboMachine | None = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("Machine Structure")
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

        self._machine = machine
        if machine:
            self.set_machine(machine)

    def set_machine(self, machine: TurboMachine) -> None:
        """Rebuild the tree from a TurboMachine object."""
        self._machine = machine
        self._tree.clear()

        root = QTreeWidgetItem(self._tree, [machine.name or "TurboMachine"])
        root.setData(0, Qt.UserRole, machine)

        for i, row in enumerate(machine.blade_rows):
            row_item = QTreeWidgetItem(root, [f"Row {i}: {row.name}"])
            row_item.setData(0, Qt.UserRole, row)

            for j, profile in enumerate(row.profiles):
                prof_item = QTreeWidgetItem(row_item, [f"Profile {j}"])
                prof_item.setData(0, Qt.UserRole, profile)

        self._tree.expandAll()

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        obj = item.data(0, Qt.UserRole)
        if obj is not None:
            self.item_selected.emit(obj)
