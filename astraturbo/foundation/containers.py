"""Collection data structures for AstraTurbo.

Modernized for Python 3.10+:
  - Uses collections.abc instead of collections for ABCs
  - Modern super() calls
  - Type hints
"""

from __future__ import annotations

import collections.abc
from typing import Any, Iterator


def hascallableattr(obj: Any, name: str) -> bool:
    """Check if an object has a callable attribute with the given name."""
    return hasattr(obj, name) and callable(getattr(obj, name))


class ChildrenList(collections.abc.MutableSequence):
    """Ordered list of child nodes that automatically sets parent references.

    When a node is added to this list, its parent is set to the owner.
    When removed, the parent reference is cleared.
    """

    def __init__(self, parent: Any):
        self._children: list = []
        self._parent = parent

    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            for child in self._children[index]:
                child._parent = None
            del self._children[index]
        else:
            self._children[index]._parent = None
            del self._children[index]

    def __getitem__(self, index: int | slice) -> Any:
        return self._children[index]

    def __setitem__(self, index: int | slice, item: Any) -> None:
        if isinstance(index, slice):
            raise NotImplementedError("Slice assignment not supported")
        old = self._children[index]
        old._parent = None
        item._parent = self._parent
        self._children[index] = item

    def __len__(self) -> int:
        return len(self._children)

    def __repr__(self) -> str:
        return repr(self._children)

    def __str__(self) -> str:
        return str(self._children)

    def insert(self, index: int, item: Any) -> None:
        item._parent = self._parent
        self._children.insert(index, item)


class ObserverSet(collections.abc.MutableSet):
    """Set of observer objects that can be notified of changes.

    Each observer must have a callable `update()` method.
    """

    def __init__(self) -> None:
        super().__init__()
        self._observers: set = set()

    def __contains__(self, element: Any) -> bool:
        return element in self._observers

    def __iter__(self) -> Iterator:
        return iter(self._observers)

    def __len__(self) -> int:
        return len(self._observers)

    def __repr__(self) -> str:
        return repr(self._observers)

    def __str__(self) -> str:
        return str(self._observers)

    def add(self, value: Any) -> None:
        if not hascallableattr(value, "update"):
            raise TypeError(
                f"'{type(value).__name__}' object has no callable attribute 'update'"
            )
        self._observers.add(value)

    def discard(self, value: Any) -> None:
        self._observers.discard(value)

    def notify(self) -> None:
        """Call update() on all registered observers."""
        for observer in self._observers:
            observer.update()
