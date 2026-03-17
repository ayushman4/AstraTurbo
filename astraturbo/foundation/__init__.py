"""Foundation layer for AstraTurbo.

Provides the core infrastructure that all domain objects build upon:
  - Property descriptors with change tracking, validation, and undo support
  - Signal system for decoupled component communication
  - Undo/redo stack
  - Collection types (ChildrenList, ObserverSet)
  - Caching decorators
  - Serialization (YAML + legacy XML import)
  - Unit conversions
"""

from .containers import ChildrenList, ObserverSet
from .decorators import memoize
from .properties import (
    BoundedNumericProperty,
    ChildProperty,
    NumericProperty,
    Property,
    SharedBoundedNumericProperty,
    SharedNumericProperty,
    SharedProperty,
    StringProperty,
)
from .signals import computation_finished, property_changed, undo_stack_changed
from .undo import Stack, group, setstack, stack, undoable

__all__ = [
    # Properties
    "Property",
    "SharedProperty",
    "NumericProperty",
    "SharedNumericProperty",
    "BoundedNumericProperty",
    "SharedBoundedNumericProperty",
    "StringProperty",
    "ChildProperty",
    # Signals
    "property_changed",
    "undo_stack_changed",
    "computation_finished",
    # Undo
    "undoable",
    "group",
    "Stack",
    "stack",
    "setstack",
    # Containers
    "ChildrenList",
    "ObserverSet",
    # Decorators
    "memoize",
]
