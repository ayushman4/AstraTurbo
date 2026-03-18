"""AstraTurbo AI Assistant — Claude-powered design copilot.

Uses the Claude API with tool use to translate natural language into
AstraTurbo operations. Engineers describe what they want, and the
assistant calls the appropriate AstraTurbo functions.

Requires: pip install anthropic
Set ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import os
import sys

from .tools import TOOLS, execute_tool


SYSTEM_PROMPT = """You are AstraTurbo Assistant, an expert turbomachinery design engineer with access to the AstraTurbo simulation platform.

You help engineers design compressor and turbine blades by translating their requirements into AstraTurbo operations. You have deep knowledge of:
- Aerodynamic design: velocity triangles, meanline analysis, blade loading, De Haller ratio
- Off-design analysis: incidence, deviation, loss models, stall/choke detection
- Compressor maps: speed lines, surge line, surge margin, part-load performance
- Blade geometry: camber lines (NACA 65, circular arc, polynomial), thickness distributions
- Mesh generation: structured O-grid meshes, boundary layer resolution, y+ requirements
- CFD simulation: OpenFOAM, ANSYS Fluent, ANSYS CFX, SU2
- Structural analysis: centrifugal stress, material selection, CalculiX/Abaqus
- Turbomachinery materials: Inconel 718, Ti-6Al-4V, CMSX-4, etc.

When an engineer describes what they want, you:
1. Identify which AstraTurbo tools to call
2. Extract the parameters from their description
3. Call the tools in the right order
4. Explain the results in engineering terms
5. Suggest next steps or flag any concerns (e.g., De Haller < 0.72, high y+)

Design rules of thumb you should apply:
- De Haller ratio should be > 0.72 (below indicates excessive diffusion/separation risk)
- Loading coefficient psi < 0.45 for conservative design, < 0.55 for aggressive
- Flow coefficient phi typically 0.3-0.8
- Degree of reaction 0.5 = symmetric stage (most common)
- y+ = 1 for resolved boundary layer, 30-100 for wall functions
- NACA 65 camber line is standard for compressors
- k-omega SST is the recommended turbulence model for turbomachinery
- Diffusion factor DF < 0.6 to avoid stall; DF > 0.6 = stalled
- Surge margin > 15% is typically required for safe operation
- At off-design: reduced mass flow increases incidence and DF (toward stall)
- At off-design: lower RPM reduces pressure ratio and efficiency
- Inconel 718 for compressor discs/blades up to 700°C
- Ti-6Al-4V for fan and LP compressor blades
- CMSX-4 (single crystal) for HP turbine blades up to 1100°C

Always be direct and specific. Give actual numbers, not vague advice.
If a design parameter is questionable, say so explicitly with the reasoning."""


class AstraTurboAssistant:
    """Claude API-powered assistant with automatic tool execution.

    Usage::

        assistant = AstraTurboAssistant()
        response = assistant.chat("Design a 5-stage compressor with PR=8")
        print(response)

        # Multi-turn conversation
        response2 = assistant.chat("Now set up an OpenFOAM case for the first stage")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required for the AI assistant.\n"
                "Install with: pip install anthropic\n"
                "Then set ANTHROPIC_API_KEY environment variable."
            )

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set.\n"
                "Get one at https://console.anthropic.com/\n"
                "Then: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._model = model
        self._messages: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Send a message and get a response with automatic tool execution.

        The assistant will call AstraTurbo tools as needed, then return
        a final text response explaining what was done.

        Args:
            user_message: The engineer's request in natural language.

        Returns:
            The assistant's response text.
        """
        self._messages.append({"role": "user", "content": user_message})

        while True:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self._messages,
            )

            if response.stop_reason == "end_turn":
                break

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            self._messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool in tool_use_blocks:
                print(f"  [Executing: {tool.name}...]", file=sys.stderr)
                result = execute_tool(tool.name, tool.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool.id,
                    "content": result,
                })

            self._messages.append({"role": "user", "content": tool_results})

        self._messages.append({"role": "assistant", "content": response.content})
        text_parts = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_parts)

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []


def create_assistant(api_key: str | None = None) -> AstraTurboAssistant:
    """Create an AI assistant.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.

    Returns:
        AstraTurboAssistant instance.

    Raises:
        ImportError: If anthropic package is not installed.
        ValueError: If no API key is available.
    """
    return AstraTurboAssistant(api_key=api_key)


def chat_cli() -> None:
    """Interactive CLI chat with the AstraTurbo assistant."""
    print("AstraTurbo AI Assistant (Claude API — tool use enabled)")
    print("Type your turbomachinery design requests. Type 'quit' to exit.\n")

    try:
        assistant = create_assistant()
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            assistant.reset()
            print("Conversation reset.\n")
            continue

        response = assistant.chat(user_input)
        print(f"\nAssistant: {response}\n")
