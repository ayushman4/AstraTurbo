"""Tests for foundation property system."""

import pytest
from astraturbo.foundation.properties import (
    BoundedNumericProperty,
    NumericProperty,
    Property,
    StringProperty,
)
from astraturbo.baseclass import Node


class DummyNode(Node):
    value = NumericProperty(default=0)
    name_prop = StringProperty(default="test")
    bounded = BoundedNumericProperty(lb=0, ub=100, default=50)


class TestProperty:
    def test_default_value(self):
        n = DummyNode()
        assert n.value == 0

    def test_set_and_get(self):
        n = DummyNode()
        n.value = 42
        assert n.value == 42

    def test_numeric_rejects_string(self):
        n = DummyNode()
        with pytest.raises(TypeError):
            n.value = "not a number"

    def test_string_rejects_int(self):
        n = DummyNode()
        with pytest.raises(TypeError):
            n.name_prop = 123

    def test_bounded_within_range(self):
        n = DummyNode()
        n.bounded = 75
        assert n.bounded == 75

    def test_bounded_below_min(self):
        n = DummyNode()
        with pytest.raises(ValueError):
            n.bounded = -1

    def test_bounded_above_max(self):
        n = DummyNode()
        with pytest.raises(ValueError):
            n.bounded = 101

    def test_same_value_no_change(self):
        n = DummyNode()
        n.value = 10
        n.value = 10  # Should not raise or change

    def test_property_descriptor_access(self):
        desc = DummyNode.__dict__["value"]
        assert isinstance(desc, NumericProperty)

    def test_properties_iterator(self):
        n = DummyNode()
        prop_names = {p.name for p in n.properties}
        assert "value" in prop_names
        assert "name_prop" in prop_names
        assert "bounded" in prop_names
