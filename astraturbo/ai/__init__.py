"""AstraTurbo AI module — Claude-powered design assistant.

Uses the Claude API with tool use to provide a natural language interface
to all AstraTurbo turbomachinery design tools.

Requires: pip install anthropic
Set ANTHROPIC_API_KEY environment variable.

Usage::

    from astraturbo.ai import create_assistant

    assistant = create_assistant()
    response = assistant.chat(
        "Design a 5-stage axial compressor with PR=8, "
        "mass flow 25 kg/s at 15000 RPM"
    )
    print(response)
"""

from .assistant import AstraTurboAssistant, create_assistant, chat_cli
from .tools import TOOLS, execute_tool

__all__ = [
    "AstraTurboAssistant",
    "create_assistant",
    "chat_cli",
    "TOOLS",
    "execute_tool",
]
