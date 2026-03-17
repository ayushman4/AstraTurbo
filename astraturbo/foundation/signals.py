"""Signal/event system for AstraTurbo.

Provides named signals that components can send and receive to communicate
state changes without direct coupling. Uses `blinker` for cross-platform
signal dispatch.
"""

from blinker import Signal


# Emitted when any Property descriptor value changes.
# Keyword args: node (the object), name (property name), value (new value)
property_changed = Signal()

# Emitted when the undo stack changes (push, undo, redo, clear).
undo_stack_changed = Signal()

# Emitted when a model computation completes.
computation_finished = Signal()
