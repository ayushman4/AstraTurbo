"""AI Chat panel for the AstraTurbo GUI.

Provides an interactive chat interface to the Claude-powered design
assistant directly within the GUI.
"""

from __future__ import annotations

import os
import sys
import threading

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QLineEdit, QPushButton, QLabel, QMessageBox,
)


class _WorkerSignals(QObject):
    """Signals for the background AI worker thread."""
    response_ready = Signal(str)
    tool_called = Signal(str)
    error = Signal(str)


class AIChatPanel(QWidget):
    """Chat panel for interacting with the AI design assistant."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._assistant = None
        self._worker_signals = _WorkerSignals()
        self._worker_signals.response_ready.connect(self._on_response)
        self._worker_signals.tool_called.connect(self._on_tool_called)
        self._worker_signals.error.connect(self._on_error)
        self._busy = False

        layout = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("AI Design Assistant (Claude)"))
        header.addStretch()

        self._status_label = QLabel("")
        header.addWidget(self._status_label)

        self._reset_btn = QPushButton("Reset Chat")
        self._reset_btn.clicked.connect(self._reset)
        header.addWidget(self._reset_btn)
        layout.addLayout(header)

        # Chat history
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: monospace; font-size: 13px; padding: 8px; }"
        )
        self._chat_display.setPlaceholderText(
            "Ask the AI to help design turbomachinery...\n\n"
            "Examples:\n"
            '  "Design a 5-stage axial compressor with PR=8, mass flow 25 kg/s at 15000 RPM"\n'
            '  "What first cell height do I need for y+=1 at Mach 0.6 with 100mm chord?"\n'
            '  "Generate a NACA 65 profile with 10% thickness and export to CSV"\n'
            '  "Set up an OpenFOAM case for a rotor at 1200 rad/s"\n'
            '  "What materials are available for HP turbine blades?"\n'
        )
        layout.addWidget(self._chat_display)

        # Input area
        input_layout = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type your design request...")
        self._input.returnPressed.connect(self._send)
        self._input.setStyleSheet(
            "QLineEdit { padding: 8px; font-size: 14px; }"
        )
        input_layout.addWidget(self._input)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._send)
        self._send_btn.setFixedWidth(80)
        input_layout.addWidget(self._send_btn)
        layout.addLayout(input_layout)

    def _ensure_assistant(self) -> bool:
        """Initialize the assistant — API mode only for GUI."""
        if self._assistant is not None:
            return True

        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            self._append_system(
                "API key not set. The AI assistant requires an Anthropic API key.\n"
                "\n"
                "Setup (one time):\n"
                "  1. Get a key at https://console.anthropic.com/\n"
                "  2. Run: pip install anthropic\n"
                "  3. Run: export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  4. Restart the GUI\n"
                "\n"
                "Or use the CLI instead (works without the GUI):\n"
                "  python -m astraturbo ai\n"
                "\n"
                "You can still use all other AstraTurbo features without the AI."
            )
            return False

        try:
            from astraturbo.ai.assistant import AstraTurboAssistant
            self._assistant = AstraTurboAssistant(api_key=api_key)
            self._append_system("Connected: Claude API (tool use enabled)")
            return True
        except ImportError:
            self._append_system(
                "The 'anthropic' package is not installed.\n"
                "Run: pip install anthropic\n"
                "Then restart the GUI."
            )
            return False
        except Exception as e:
            self._append_system(f"Error connecting to Claude API: {e}")
            return False

    def _send(self) -> None:
        """Send the user's message to the AI assistant."""
        text = self._input.text().strip()
        if not text or self._busy:
            return

        if not self._ensure_assistant():
            return

        self._input.clear()
        self._append_message("You", text, color="#569cd6")
        self._set_busy(True)

        # Run in background thread to avoid freezing GUI
        thread = threading.Thread(target=self._run_chat, args=(text,), daemon=True)
        thread.start()

    def _run_chat(self, message: str) -> None:
        """Run the AI chat in a background thread."""
        import traceback
        original_stderr = sys.stderr
        try:
            class ToolCapture:
                def __init__(self, signals):
                    self._signals = signals
                    self._original = original_stderr

                def write(self, text):
                    if "[Executing:" in text:
                        tool_name = text.strip().strip("[]").replace("Executing: ", "").replace("...", "")
                        self._signals.tool_called.emit(tool_name)
                    self._original.write(text)

                def flush(self):
                    self._original.flush()

            sys.stderr = ToolCapture(self._worker_signals)
            response = self._assistant.chat(message)
            sys.stderr = original_stderr

            if response and response.strip():
                self._worker_signals.response_ready.emit(response)
            else:
                self._worker_signals.error.emit(
                    "AI returned empty response. The backend may not be working.\n\n"
                    "If using Claude CLI fallback, try API mode instead:\n"
                    "  pip install anthropic\n"
                    "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                    "  Then restart the GUI."
                )
        except Exception as e:
            sys.stderr = original_stderr
            self._worker_signals.error.emit(
                f"{type(e).__name__}: {e}\n\n"
                f"If using Claude CLI fallback, it may not work in this environment.\n"
                f"Use API mode: pip install anthropic + set ANTHROPIC_API_KEY"
            )

    def _on_response(self, response: str) -> None:
        """Handle AI response (called on main thread via signal)."""
        self._append_message("Assistant", response, color="#4ec9b0")
        self._set_busy(False)

    def _on_tool_called(self, tool_name: str) -> None:
        """Show tool execution in the chat."""
        self._append_system(f"Executing: {tool_name}")

    def _on_error(self, error: str) -> None:
        """Handle error from AI thread."""
        self._append_message("Error", error, color="#f44747")
        self._set_busy(False)

    def _append_message(self, sender: str, text: str, color: str = "#d4d4d4") -> None:
        """Append a message to the chat display."""
        html = (
            f'<div style="margin-bottom: 12px;">'
            f'<b style="color: {color};">{sender}:</b><br>'
            f'<span style="color: #d4d4d4; white-space: pre-wrap;">{self._escape_html(text)}</span>'
            f'</div>'
        )
        self._chat_display.append(html)
        # Scroll to bottom
        scrollbar = self._chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_system(self, text: str) -> None:
        """Append a system/status message."""
        html = (
            f'<div style="margin-bottom: 4px;">'
            f'<i style="color: #808080;">{self._escape_html(text)}</i>'
            f'</div>'
        )
        self._chat_display.append(html)
        scrollbar = self._chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_busy(self, busy: bool) -> None:
        """Toggle busy state."""
        self._busy = busy
        self._send_btn.setEnabled(not busy)
        self._input.setEnabled(not busy)
        self._status_label.setText("Thinking..." if busy else "")

    def _reset(self) -> None:
        """Reset the conversation."""
        if self._assistant:
            self._assistant.reset()
        self._chat_display.clear()
        self._status_label.setText("")
        self._busy = False
        self._send_btn.setEnabled(True)
        self._input.setEnabled(True)

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
