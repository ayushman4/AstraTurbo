"""Undo/redo framework for AstraTurbo.

Originally by David Townshend. Modernized for Python 3.10+.

Usage::

    @undoable
    def change_value(obj, new, old):
        obj.value = new
        yield 'change value'
        obj.value = old

    # Undo last action
    stack().undo()

    # Group multiple actions
    with group('batch edit'):
        change_value(obj, 1, 0)
        change_value(obj2, 2, 0)
    # Single undo reverses both
    stack().undo()
"""

from __future__ import annotations

import contextlib
from collections import deque
from typing import Any, Callable, Generator

from .signals import undo_stack_changed


class _Action:
    """Represents a single undoable/redoable action.

    Wraps a generator function where code before `yield` is the "do" action
    and code after `yield` is the "undo" action.
    """

    def __init__(self, generator: Callable, args: tuple, kwargs: dict):
        self._generator = generator
        self.args = args
        self.kwargs = kwargs
        self._text = ""
        self._runner: Generator | None = None

    def do(self) -> Any:
        """Execute (or re-execute) the action."""
        self._runner = self._generator(*self.args, **self.kwargs)
        rets = next(self._runner)
        if isinstance(rets, tuple):
            self._text = rets[0]
            return rets[1:]
        elif rets is None:
            self._text = ""
            return None
        else:
            self._text = rets
            return None

    def undo(self) -> None:
        """Undo the action by advancing the generator past yield."""
        if self._runner is not None:
            try:
                next(self._runner)
            except StopIteration:
                pass
            self._runner = None

    def text(self) -> str:
        """Return descriptive text for this action."""
        return self._text


def undoable(generator: Callable) -> Callable:
    """Decorator that makes a generator function undoable.

    The decorated generator should yield once::

        @undoable
        def operation(obj, new_val, old_val):
            obj.value = new_val       # do
            yield 'description'
            obj.value = old_val       # undo
    """

    def inner(*args: Any, **kwargs: Any) -> Any:
        action = _Action(generator, args, kwargs)
        ret = action.do()
        stack().append(action)
        if isinstance(ret, tuple):
            if len(ret) == 1:
                return ret[0]
            elif len(ret) == 0:
                return None
        return ret

    inner.__name__ = generator.__name__
    inner.__doc__ = generator.__doc__
    return inner


class _Group:
    """Context manager that groups multiple undoable actions into one."""

    def __init__(self, desc: str):
        self._desc = desc
        self._stack: list[_Action | _Group] = []

    def __enter__(self) -> _Group:
        stack().setreceiver(self._stack)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is None:
            stack().resetreceiver()
            stack().append(self)
        return False

    def undo(self) -> None:
        for action in reversed(self._stack):
            action.undo()

    def do(self) -> None:
        for action in self._stack:
            action.do()

    def text(self) -> str:
        return self._desc.format(count=len(self._stack))


def group(desc: str) -> _Group:
    """Return a context manager for grouping undoable actions.

    All actions within the group are undone/redone as a single unit::

        with group('batch edit ({count} actions)'):
            action_a()
            action_b()
        stack().undo()  # undoes both action_a and action_b
    """
    return _Group(desc)


class Stack:
    """The main undo/redo stack.

    Manages a history of undoable actions with undo/redo capability.
    If an exception occurs during undo/redo, the stack is cleared to
    prevent data corruption.
    """

    def __init__(self) -> None:
        self._undos: deque[_Action | _Group] = deque()
        self._redos: deque[_Action | _Group] = deque()
        self._receiver: Any = self._undos
        self._savepoint: int | None = None

    def canundo(self) -> bool:
        """Return True if undos are available."""
        return len(self._undos) > 0

    def canredo(self) -> bool:
        """Return True if redos are available."""
        return len(self._redos) > 0

    def redo(self) -> None:
        """Redo the last undone action."""
        if self.canredo():
            action = self._redos.pop()
            with self._pausereceiver():
                try:
                    action.do()
                except Exception:
                    self.clear()
                    raise
                else:
                    self._undos.append(action)
            undo_stack_changed.send(self)

    def undo(self) -> None:
        """Undo the last action."""
        if self.canundo():
            action = self._undos.pop()
            with self._pausereceiver():
                try:
                    action.undo()
                except Exception:
                    self.clear()
                    raise
                else:
                    self._redos.append(action)
            undo_stack_changed.send(self)

    def clear(self) -> None:
        """Clear all undo/redo history."""
        self._undos.clear()
        self._redos.clear()
        self._savepoint = None
        self._receiver = self._undos
        undo_stack_changed.send(self)

    def undocount(self) -> int:
        """Return the number of available undos."""
        return len(self._undos)

    def redocount(self) -> int:
        """Return the number of available redos."""
        return len(self._redos)

    def undotext(self) -> str | None:
        """Return description of the next available undo."""
        if self.canundo():
            return ("Undo " + self._undos[-1].text()).strip()
        return None

    def redotext(self) -> str | None:
        """Return description of the next available redo."""
        if self.canredo():
            return ("Redo " + self._redos[-1].text()).strip()
        return None

    @contextlib.contextmanager
    def _pausereceiver(self):
        """Temporarily pause the receiver during undo/redo."""
        self.setreceiver([])
        yield
        self.resetreceiver()

    def setreceiver(self, receiver: Any = None) -> None:
        """Set an object to receive pushed actions (must have append())."""
        assert hasattr(receiver, "append")
        self._receiver = receiver

    def resetreceiver(self) -> None:
        """Reset receiver to the internal undo stack."""
        self._receiver = self._undos

    def append(self, action: _Action | _Group) -> None:
        """Add an action to the stack."""
        if self._receiver is not None:
            self._receiver.append(action)
        if self._receiver is self._undos:
            self._redos.clear()
            undo_stack_changed.send(self)

    def savepoint(self) -> None:
        """Mark the current position as a save point."""
        self._savepoint = self.undocount()
        undo_stack_changed.send(self)

    def haschanged(self) -> bool:
        """Return True if the state has changed since the last save point."""
        return self._savepoint is None or self._savepoint != self.undocount()


_stack: Stack | None = None


def stack() -> Stack:
    """Return the current undo stack (creates one if needed)."""
    global _stack
    if _stack is None:
        _stack = Stack()
    return _stack


def setstack(new_stack: Stack) -> None:
    """Set a specific Stack object as the current undo stack."""
    global _stack
    _stack = new_stack
