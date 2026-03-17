"""Hierarchical tree node for AstraTurbo domain objects.

Nodes form a tree structure (parent/children) with observer-based change
propagation. When a node's properties change, its observers (typically the
parent node) are notified.
"""

from __future__ import annotations

from ..foundation.containers import ChildrenList, ObserverSet
from ..foundation.properties import StringProperty
from .atobject import ATObject


class Node(ATObject):
    """Tree node with parent/children and observer-based change notification.

    Attributes:
        name: Display name for this node.
        children: Ordered list of child nodes.
        observers: Set of objects notified on changes.
    """

    name = StringProperty()

    def __init__(self) -> None:
        self._parent: Node | None = None
        self.children = ChildrenList(parent=self)
        self.observers = ObserverSet()

    @property
    def idx(self) -> int | None:
        """Return this node's index in its parent's children list."""
        if self._parent is None:
            return None
        return self._parent.children.index(self)

    @property
    def parent(self) -> Node | None:
        """Return the parent node."""
        return self._parent

    @parent.setter
    def parent(self, parent: Node) -> None:
        if not isinstance(parent, Node):
            raise TypeError(
                f"{type(self).__name__}.parent accepts only Node objects "
                f"(got {type(parent).__name__})"
            )
        # Detach from old parent first
        if self._parent is not None:
            self.observers.discard(self._parent)
        self.observers.add(parent)
        self._parent = parent

    @parent.deleter
    def parent(self) -> None:
        if self._parent is not None:
            self.observers.discard(self._parent)
            self._parent = None

    def update(self, broadcast: bool = True) -> None:
        """Handle property changes.

        Clears the computation cache and optionally notifies observers.
        Subclasses should call super().update(broadcast) after their own logic.
        """
        self.invalidate_cache()
        if broadcast:
            self.observers.notify()
