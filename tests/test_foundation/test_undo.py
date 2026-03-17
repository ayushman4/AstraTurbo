"""Tests for undo/redo framework."""

from astraturbo.foundation.undo import Stack, undoable, group, stack, setstack


class TestUndo:
    def setup_method(self):
        setstack(Stack())

    def test_undoable_action(self):
        container = {"value": 0}

        @undoable
        def set_value(new, old):
            container["value"] = new
            yield "set value"
            container["value"] = old

        set_value(10, 0)
        assert container["value"] == 10

        stack().undo()
        assert container["value"] == 0

        stack().redo()
        assert container["value"] == 10

    def test_undo_text(self):
        @undoable
        def action():
            yield "test action"

        action()
        assert stack().undotext() == "Undo test action"

    def test_group(self):
        container = {"value": 0}

        @undoable
        def increment(amount):
            container["value"] += amount
            yield f"add {amount}"
            container["value"] -= amount

        with group("batch"):
            increment(1)
            increment(2)
            increment(3)

        assert container["value"] == 6

        stack().undo()  # Undoes all three at once
        assert container["value"] == 0

    def test_clear(self):
        @undoable
        def noop():
            yield "noop"

        noop()
        assert stack().canundo()
        stack().clear()
        assert not stack().canundo()

    def test_savepoint(self):
        @undoable
        def noop():
            yield "noop"

        stack().savepoint()
        assert not stack().haschanged()
        noop()
        assert stack().haschanged()
