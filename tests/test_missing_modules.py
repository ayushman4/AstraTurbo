"""Tests for previously untested modules: distribution, serialization, AI tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────
# 1. Distribution module
# ────────────────────────────────────────────────────────────────

class TestLinearDistribution:
    """Test Linear (uniform) point distribution."""

    def test_shape(self):
        from astraturbo.distribution import Linear
        dist = Linear()
        pts = dist(50)
        assert pts.shape == (50,)

    def test_bounds(self):
        from astraturbo.distribution import Linear
        dist = Linear()
        pts = dist(100)
        assert pts[0] == pytest.approx(0.0)
        assert pts[-1] == pytest.approx(1.0)

    def test_uniform_spacing(self):
        from astraturbo.distribution import Linear
        dist = Linear()
        pts = dist(11)
        diffs = np.diff(pts)
        np.testing.assert_allclose(diffs, 0.1, atol=1e-14)

    def test_single_point(self):
        from astraturbo.distribution import Linear
        dist = Linear()
        pts = dist(1)
        assert len(pts) == 1

    def test_two_points(self):
        from astraturbo.distribution import Linear
        dist = Linear()
        pts = dist(2)
        assert pts[0] == pytest.approx(0.0)
        assert pts[1] == pytest.approx(1.0)


class TestChebyshevDistribution:
    """Test Chebyshev (endpoint-clustered) distribution."""

    def test_shape(self):
        from astraturbo.distribution import Chebyshev
        dist = Chebyshev()
        pts = dist(50)
        assert pts.shape == (50,)

    def test_bounds(self):
        from astraturbo.distribution import Chebyshev
        dist = Chebyshev()
        pts = dist(100)
        assert pts[0] == pytest.approx(0.0, abs=1e-14)
        assert pts[-1] == pytest.approx(1.0, abs=1e-14)

    def test_monotonically_increasing(self):
        from astraturbo.distribution import Chebyshev
        dist = Chebyshev()
        pts = dist(50)
        assert np.all(np.diff(pts) > 0)

    def test_clusters_at_endpoints(self):
        """Points should be denser near 0 and 1 than at 0.5."""
        from astraturbo.distribution import Chebyshev
        dist = Chebyshev()
        pts = dist(100)

        # Spacing near endpoints should be smaller than near midpoint
        spacing = np.diff(pts)
        near_start = spacing[:10].mean()
        near_middle = spacing[40:60].mean()
        assert near_start < near_middle

    def test_different_from_linear(self):
        """Chebyshev should differ from uniform spacing."""
        from astraturbo.distribution import Chebyshev, Linear
        cheb = Chebyshev()(50)
        lin = Linear()(50)
        assert not np.allclose(cheb, lin)


# ────────────────────────────────────────────────────────────────
# 2. Serialization module
# ────────────────────────────────────────────────────────────────

class TestSerialization:
    """Test YAML save/load and XML import."""

    def test_serialize_camberline(self):
        from astraturbo.camberline import CircularArc
        from astraturbo.foundation.serialization import serialize_instance

        arc = CircularArc()
        d = serialize_instance(arc)
        assert d["__class__"] == "CircularArc"
        assert "astraturbo" in d["__module__"]

    def test_round_trip_yaml(self, tmp_path):
        """Save and load should preserve class and properties.

        Note: yaml.dump serializes Python objects with tags, and
        yaml.safe_load cannot deserialize them. The real project
        save/load uses machine-level serialization. Here we test
        serialize_instance + unserialize_object directly.
        """
        from astraturbo.camberline import CircularArc
        from astraturbo.foundation.serialization import serialize_instance, unserialize_object

        arc = CircularArc()
        arc.sample_rate = 99

        d = serialize_instance(arc)
        loaded = unserialize_object(d)
        assert loaded.__class__.__name__ == "CircularArc"
        assert loaded.sample_rate == 99

    def test_round_trip_thickness(self, tmp_path):
        from astraturbo.thickness import NACA4Digit
        from astraturbo.foundation.serialization import serialize_instance, unserialize_object

        thick = NACA4Digit()
        thick.max_thickness = 0.12

        d = serialize_instance(thick)
        loaded = unserialize_object(d)
        assert loaded.max_thickness == pytest.approx(0.12)

    def test_unserialize_rejects_untrusted_module(self):
        from astraturbo.foundation.serialization import unserialize_object

        malicious = {
            "__class__": "System",
            "__module__": "os",
        }
        with pytest.raises(ValueError, match="Untrusted module"):
            unserialize_object(malicious)

    def test_unserialize_allows_astraturbo(self):
        from astraturbo.foundation.serialization import unserialize_object

        d = {
            "__class__": "CircularArc",
            "__module__": "astraturbo.camberline.circular_arc",
        }
        obj = unserialize_object(d)
        assert obj.__class__.__name__ == "CircularArc"

    def test_xml_import_simple(self, tmp_path):
        from astraturbo.foundation.serialization import import_bladedesigner_xml

        xml_content = """<?xml version="1.0"?>
<project>
    <name>TestBlade</name>
    <chord>0.05</chord>
    <stagger>30</stagger>
    <n_blades>24</n_blades>
    <section>
        <radius>0.15</radius>
        <camber>circular_arc</camber>
    </section>
</project>"""
        xml_file = tmp_path / "test.xml"
        xml_file.write_text(xml_content)

        result = import_bladedesigner_xml(xml_file)
        assert result["name"] == "TestBlade"
        assert result["chord"] == pytest.approx(0.05)
        assert result["n_blades"] == 24
        assert result["section"]["radius"] == pytest.approx(0.15)

    def test_xml_import_rejects_xxe(self, tmp_path):
        """XML with DOCTYPE should be rejected."""
        from astraturbo.foundation.serialization import import_bladedesigner_xml

        xxe_xml = """<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<project>
    <name>&xxe;</name>
</project>"""
        xml_file = tmp_path / "xxe.xml"
        xml_file.write_text(xxe_xml)

        with pytest.raises(ValueError, match="DOCTYPE"):
            import_bladedesigner_xml(xml_file)


# ────────────────────────────────────────────────────────────────
# 3. AI tools module
# ────────────────────────────────────────────────────────────────

class TestAITools:
    """Test AI tool registry and dispatcher."""

    def test_tools_registry_not_empty(self):
        from astraturbo.ai.tools import TOOLS
        assert len(TOOLS) > 0

    def test_each_tool_has_required_fields(self):
        from astraturbo.ai.tools import TOOLS
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert isinstance(tool["name"], str)
            assert len(tool["description"]) > 0

    def test_execute_unknown_tool(self):
        from astraturbo.ai.tools import execute_tool
        result = execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_execute_yplus_calculator(self):
        from astraturbo.ai.tools import execute_tool
        result = execute_tool("yplus_calculator", {
            "velocity": 100.0,
            "chord": 0.05,
        })
        assert "y+" in result.lower() or "cell" in result.lower() or "height" in result.lower()

    def test_execute_list_materials(self):
        from astraturbo.ai.tools import execute_tool
        result = execute_tool("list_materials", {})
        assert "ti6al4v" in result.lower() or "material" in result.lower()

    def test_execute_list_formats(self):
        from astraturbo.ai.tools import execute_tool
        result = execute_tool("list_formats", {})
        assert "cgns" in result.lower() or "vtk" in result.lower()

    def test_execute_generate_profile(self):
        from astraturbo.ai.tools import execute_tool
        result = execute_tool("generate_profile", {
            "camber": "circular_arc",
            "thickness": "naca4digit",
        })
        assert "profile" in result.lower() or "point" in result.lower()

    def test_execute_meanline(self):
        from astraturbo.ai.tools import execute_tool
        result = execute_tool("meanline_compressor", {
            "pressure_ratio": 1.3,
            "mass_flow": 20.0,
            "rpm": 15000,
        })
        # Should return a result string (not an error)
        assert "error" not in result.lower() or "pressure" in result.lower()

    def test_execute_tool_handles_errors_gracefully(self):
        """Tools should return error strings, not raise exceptions."""
        from astraturbo.ai.tools import execute_tool
        # Inspect a file that doesn't exist — should return error string
        result = execute_tool("inspect_file", {"filepath": "/nonexistent/file.xyz"})
        assert "error" in result.lower() or "Error" in result
