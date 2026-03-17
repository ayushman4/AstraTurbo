"""Descriptor-based property system for AstraTurbo domain objects.

Modernized for Python 3.10+:
  - Uses __set_name__ for automatic name discovery
  - Type hints throughout

Properties automatically:
  - Track changes via signals
  - Support undo/redo
  - Validate types and bounds
  - Propagate shared values to children
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from .signals import property_changed
from .undo import undoable


class Property:
    """Base descriptor with change notification and undo support.

    When set on an ATObject/Node subclass, changing the value will:
    1. Validate via check()
    2. Record the change for undo/redo
    3. Emit property_changed signal
    4. Call instance.update() to propagate
    """

    def __init__(self, default: Any = None):
        self.default = default
        self.name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return self
        return self.get(instance)

    def __set__(self, instance: Any, new_val: Any) -> None:
        self.check(instance, new_val)
        old_val = self.get(instance)
        if new_val == old_val:
            return
        if old_val is not None:
            self.set(instance, new_val, old_val)
        else:
            self._set(instance, new_val, broadcast=False)

    def __delete__(self, instance: Any) -> None:
        raise AttributeError(f"Cannot delete property '{self.name}'")

    def get(self, instance: Any) -> Any:
        return instance.__dict__.get(self.name, self.default)

    @undoable
    def set(self, instance: Any, new_val: Any, old_val: Any):
        self.will_redo(instance, new_val, old_val)
        self._set(instance, new_val)
        property_changed.send(self, node=instance, name=self.name, value=new_val)
        yield f"{self.name} change"
        self.will_undo(instance, new_val, old_val)
        self._set(instance, old_val)
        property_changed.send(self, node=instance, name=self.name, value=old_val)

    def _set(self, instance: Any, val: Any, broadcast: bool = True) -> None:
        instance.__dict__[self.name] = val
        if hasattr(instance, "update"):
            instance.update(broadcast)

    def check(self, instance: Any, val: Any) -> None:
        """Override to validate value before setting."""

    def will_undo(self, instance: Any, new_val: Any, old_val: Any) -> None:
        """Hook called before undo."""

    def will_redo(self, instance: Any, new_val: Any, old_val: Any) -> None:
        """Hook called before redo."""


class SharedProperty(Property):
    """Property that propagates its value to all children of the instance."""

    def _set(self, instance: Any, new_val: Any, broadcast: bool = True) -> None:
        if hasattr(instance, "children"):
            for child in instance.children:
                descriptor = getattr(type(child), self.name, None)
                if descriptor is not None and isinstance(descriptor, Property):
                    descriptor._set(child, new_val, broadcast=False)
        super()._set(instance, new_val, broadcast)


class NumericProperty(Property):
    """Property that only accepts int or float values."""

    def __init__(self, default: int | float = 0):
        super().__init__(default)

    def check(self, instance: Any, val: Any) -> None:
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"{type(instance).__name__}.{self.name} accepts only int/float "
                f"(got {type(val).__name__})"
            )


class SharedNumericProperty(SharedProperty, NumericProperty):
    """Numeric property that propagates to children."""


class BoundedNumericProperty(NumericProperty):
    """Numeric property constrained to [lb, ub]."""

    def __init__(self, lb: float, ub: float, default: int | float = 0):
        super().__init__(default)
        self.lb = lb
        self.ub = ub

    def check(self, instance: Any, val: Any) -> None:
        super().check(instance, val)
        if val < self.lb:
            raise ValueError(
                f"{type(instance).__name__}.{self.name} = {val} "
                f"is below the lower bound ({self.lb})"
            )
        if val > self.ub:
            raise ValueError(
                f"{type(instance).__name__}.{self.name} = {val} "
                f"is above the upper bound ({self.ub})"
            )


class SharedBoundedNumericProperty(SharedProperty, BoundedNumericProperty):
    """Bounded numeric property that propagates to children."""


class StringProperty(Property):
    """Property that only accepts string values."""

    def __init__(self, default: str = ""):
        super().__init__(default)

    def check(self, instance: Any, val: Any) -> None:
        if not isinstance(val, str):
            raise TypeError(
                f"{type(instance).__name__}.{self.name} accepts only str "
                f"(got {type(val).__name__})"
            )


class ChildProperty(Property):
    """Property that manages a child node in the parent's children list.

    The child_index maps this property to a specific position in the
    children list.
    """

    def __init__(self, child_index: int, default: Any = None):
        super().__init__(default)
        self.child_index = child_index

    def get(self, instance: Any) -> Any:
        try:
            return instance.children[self.child_index]
        except (IndexError, AttributeError):
            return self.default

    def _set(self, instance: Any, val: Any, broadcast: bool = True) -> None:
        # Propagate shared properties from parent to new child
        if hasattr(val, "properties"):
            for p in val.properties:
                if isinstance(p, SharedProperty):
                    with suppress(AttributeError):
                        p._set(val, getattr(instance, p.name), broadcast=False)
        # Insert or replace in children list
        if hasattr(instance, "children"):
            try:
                instance.children[self.child_index] = val
            except IndexError:
                instance.children.append(val)
        if hasattr(instance, "update"):
            instance.update(broadcast)

    def will_redo(self, instance: Any, new_val: Any, old_val: Any) -> None:
        with suppress(AttributeError):
            visible_desc = getattr(type(new_val), "visible", None)
            if visible_desc is not None:
                visible_desc._set(new_val, old_val.visible)
            visible_desc = getattr(type(old_val), "visible", None)
            if visible_desc is not None:
                visible_desc._set(old_val, False)

    def will_undo(self, instance: Any, new_val: Any, old_val: Any) -> None:
        with suppress(AttributeError):
            visible_desc = getattr(type(old_val), "visible", None)
            if visible_desc is not None:
                visible_desc._set(old_val, new_val.visible)
            visible_desc = getattr(type(new_val), "visible", None)
            if visible_desc is not None:
                visible_desc._set(new_val, False)
