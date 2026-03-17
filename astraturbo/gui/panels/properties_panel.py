"""Properties panel — generic property editor.

Reads ATObject Property descriptors and renders editable fields.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QWidget,
    QScrollArea,
    QVBoxLayout,
)

from ...foundation.properties import (
    Property,
    NumericProperty,
    BoundedNumericProperty,
    StringProperty,
)


class PropertiesPanel(QWidget):
    """Generic property editor that reads ATObject descriptors."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self._form_widget = QWidget()
        self._form_layout = QFormLayout(self._form_widget)
        scroll.setWidget(self._form_widget)

        self._current_obj = None
        self._widgets: dict[str, QWidget] = {}

    def set_object(self, obj) -> None:
        """Populate the panel with properties from an ATObject."""
        self._current_obj = obj
        self._clear()

        if not hasattr(obj, "properties"):
            return

        for prop in obj.properties:
            widget = self._create_widget(prop, obj)
            if widget is not None:
                self._form_layout.addRow(prop.name.replace("_", " ").title(), widget)
                self._widgets[prop.name] = widget

    def _clear(self) -> None:
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()

    def _create_widget(self, prop: Property, obj) -> QWidget | None:
        value = prop.get(obj)

        if isinstance(prop, BoundedNumericProperty):
            spin = QDoubleSpinBox()
            spin.setRange(prop.lb, prop.ub)
            spin.setDecimals(6)
            if value is not None:
                spin.setValue(float(value))
            spin.valueChanged.connect(
                lambda v, p=prop: self._on_value_changed(p, v)
            )
            return spin

        elif isinstance(prop, NumericProperty):
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(6)
            if value is not None:
                spin.setValue(float(value))
            spin.valueChanged.connect(
                lambda v, p=prop: self._on_value_changed(p, v)
            )
            return spin

        elif isinstance(prop, StringProperty):
            edit = QLineEdit()
            if value is not None:
                edit.setText(str(value))
            edit.textChanged.connect(
                lambda v, p=prop: self._on_value_changed(p, v)
            )
            return edit

        return None

    def _on_value_changed(self, prop: Property, value) -> None:
        if self._current_obj is not None:
            try:
                setattr(self._current_obj, prop.name, value)
            except (ValueError, TypeError):
                pass
