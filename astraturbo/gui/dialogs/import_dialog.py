"""Import dialog for legacy XML project files."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QFileDialog,
    QPushButton, QTextEdit, QDialogButtonBox,
)


class ImportDialog(QDialog):
    """Dialog for importing legacy XML projects."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Legacy XML Project")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a legacy XML project file to import."))

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse)
        layout.addWidget(self._browse_btn)

        self._path_label = QLabel("No file selected")
        layout.addWidget(self._path_label)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setMaximumHeight(200)
        layout.addWidget(self._preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.selected_path: str | None = None

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select XML File", "", "XML Files (*.xml)"
        )
        if path:
            self.selected_path = path
            self._path_label.setText(path)
            self._preview.setPlainText(f"Selected: {path}")
