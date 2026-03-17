"""Tests for the 5 adaptation gaps: polyline, vertex extraction,
grading projection, multi-block mesher, multi-stage orchestration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from astraturbo.profile import Superposition
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA4Digit


def _make_profile():
    """Helper: create a standard test profile."""
    return Superposition(NACA65(cl0=1.0), NACA4Digit(max_thickness=0.10)).as_array()


# ============================================================
# Gap 1: Polyline and Arc
# ============================================================

from astraturbo.mesh.polyline import Polyline, Arc, BlockEdge


class TestPolyline:
    def test_create(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
        pl = Polyline(pts)
        assert pl.n_points == 4
        assert pl.dim == 2

    def test_total_length(self):
        pts = np.array([[0, 0], [3, 0], [3, 4]], dtype=np.float64)
        pl = Polyline(pts)
        assert pl.total_length() == pytest.approx(7.0, abs=1e-10)  # 3 + 4

    def test_normalized_parameters(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        pl = Polyline(pts)
        params = pl.normalized_parameters()
        np.testing.assert_allclose(params, [0.0, 0.5, 1.0], atol=1e-10)

    def test_evaluate_at_midpoint(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        pl = Polyline(pts)
        mid = pl.evaluate_at(0.5)
        np.testing.assert_allclose(mid, [1.0, 0.0], atol=1e-10)

    def test_resample(self):
        pts = np.array([[0, 0], [10, 0]], dtype=np.float64)
        pl = Polyline(pts)
        resampled = pl.resample(11)
        assert resampled.n_points == 11
        np.testing.assert_allclose(resampled.points[5], [5.0, 0.0], atol=1e-10)

    def test_resample_graded(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
                       dtype=np.float64)
        pl = Polyline(pts)
        graded = pl.resample_graded(11, grading_ratio=3.0)
        assert graded.n_points == 11
        # First segment should be smaller than last
        segs = graded.segment_lengths()
        assert segs[-1] > segs[0]

    def test_reverse(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        pl = Polyline(pts, start_vertex=0, end_vertex=1)
        rev = pl.reverse()
        np.testing.assert_allclose(rev.start, [2, 0])
        assert rev.start_vertex == 1
        assert rev.end_vertex == 0

    def test_split_at(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        pl = Polyline(pts)
        left, right = pl.split_at(0.5)
        assert left.end[0] == pytest.approx(1.0, abs=0.01)
        assert right.start[0] == pytest.approx(1.0, abs=0.01)


class TestArc:
    def test_create_2d(self):
        start = np.array([1, 0], dtype=np.float64)
        end = np.array([-1, 0], dtype=np.float64)
        mid = np.array([0, 1], dtype=np.float64)
        arc = Arc(start, end, mid)
        assert arc.radius == pytest.approx(1.0, abs=0.01)
        np.testing.assert_allclose(arc.center, [0, 0], atol=0.01)

    def test_to_polyline(self):
        start = np.array([1, 0], dtype=np.float64)
        end = np.array([0, 1], dtype=np.float64)
        mid = np.array([np.sqrt(2)/2, np.sqrt(2)/2], dtype=np.float64)
        arc = Arc(start, end, mid)
        pl = arc.to_polyline(50)
        assert pl.n_points == 50
        # All points should be ~radius=1 from center
        dists = np.sqrt(np.sum((pl.points - arc.center)**2, axis=1))
        np.testing.assert_allclose(dists, 1.0, atol=0.02)

    def test_create_3d(self):
        start = np.array([1, 0, 0], dtype=np.float64)
        end = np.array([0, 1, 0], dtype=np.float64)
        mid = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0], dtype=np.float64)
        arc = Arc(start, end, mid)
        assert arc.radius == pytest.approx(1.0, abs=0.01)
        pl = arc.to_polyline(30)
        assert pl.n_points == 30


class TestBlockEdge:
    def test_polyline_edge(self):
        pts = np.array([[0, 0], [0.5, 0.5], [1, 1]], dtype=np.float64)
        edge = BlockEdge(curve=Polyline(pts), edge_type="polyline")
        pl = edge.resample_graded(10, grading=1.0)
        assert pl.n_points == 10

    def test_arc_edge(self):
        arc = Arc(np.array([1, 0.0]), np.array([0, 1.0]),
                  np.array([np.sqrt(2)/2, np.sqrt(2)/2]))
        edge = BlockEdge(curve=arc, edge_type="arc")
        pl = edge.resample_graded(20, grading=2.0)
        assert pl.n_points == 20


# ============================================================
# Gap 2: Vertex extraction
# ============================================================

from astraturbo.mesh.vertex_extraction import (
    extract_vertices_from_profile,
    extract_polylines_from_profile,
    build_passage_vertices,
    build_passage_edges,
    build_block_topology_from_profile,
)


class TestVertexExtraction:
    def test_extract_vertices(self):
        profile = _make_profile()
        verts = extract_vertices_from_profile(profile)
        assert "te" in verts
        assert "le" in verts
        assert "ss_peak" in verts
        assert "ps_peak" in verts
        # LE should have min x
        assert verts["le"].coords[0] == pytest.approx(profile[:, 0].min(), abs=0.01)

    def test_extract_polylines(self):
        profile = _make_profile()
        polylines = extract_polylines_from_profile(profile)
        assert "suction" in polylines
        assert "pressure" in polylines
        assert "full" in polylines
        # Suction + pressure should cover the whole profile
        n_ss = polylines["suction"].n_points
        n_ps = polylines["pressure"].n_points
        assert n_ss + n_ps - 1 == len(profile)  # -1 for shared LE point

    def test_build_passage_vertices(self):
        profile = _make_profile()
        verts = build_passage_vertices(profile, pitch=0.05, inlet_offset=0.2, outlet_offset=0.2)
        assert len(verts) == 8
        # Check vertex indices
        indices = {v.index for v in verts}
        assert indices == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_build_passage_edges(self):
        profile = _make_profile()
        verts = build_passage_vertices(profile, pitch=0.05, inlet_offset=0.2, outlet_offset=0.2)
        edges = build_passage_edges(verts, profile)
        assert len(edges) == 6  # suction, pressure, inlet, outlet, lower, upper

    def test_build_topology(self):
        profile = _make_profile()
        topo = build_block_topology_from_profile(profile, pitch=0.05)
        assert len(topo.vertices) == 8
        assert len(topo.edges) == 6
        assert len(topo.faces) == 5


# ============================================================
# Gap 3: Grading projection
# ============================================================

from astraturbo.mesh.grading import (
    compute_graded_parameters,
    compute_double_sided_grading,
    compute_boundary_layer_grading,
    project_grading_onto_polyline,
    project_grading_onto_arc,
    project_boundary_layer_onto_polyline,
)


class TestGrading:
    def test_uniform(self):
        params = compute_graded_parameters(11, 1.0)
        assert len(params) == 11
        diffs = np.diff(params)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-10)

    def test_growing(self):
        params = compute_graded_parameters(11, 5.0)
        diffs = np.diff(params)
        assert diffs[-1] > diffs[0]

    def test_shrinking(self):
        params = compute_graded_parameters(11, 0.2)
        diffs = np.diff(params)
        assert diffs[-1] < diffs[0]

    def test_double_sided(self):
        params = compute_double_sided_grading(21, start_grading=0.5, end_grading=0.5)
        assert len(params) == 21
        assert params[0] == pytest.approx(0.0, abs=1e-10)
        assert params[-1] == pytest.approx(1.0, abs=1e-10)

    def test_boundary_layer(self):
        params = compute_boundary_layer_grading(20, 0.001, 0.1, 1.2)
        assert len(params) == 20
        diffs = np.diff(params)
        # First cell should be smallest
        assert diffs[0] < diffs[-1]

    def test_project_onto_polyline(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float64)
        pl = Polyline(pts)
        graded = project_grading_onto_polyline(pl, 11, grading_ratio=3.0)
        assert graded.n_points == 11
        segs = graded.segment_lengths()
        assert segs[-1] > segs[0]

    def test_project_onto_arc(self):
        arc = Arc(np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                  np.array([np.sqrt(2)/2, np.sqrt(2)/2]))
        graded = project_grading_onto_arc(arc, 20, grading_ratio=2.0)
        assert graded.n_points == 20

    def test_boundary_layer_onto_polyline(self):
        pts = np.array([[0, 0], [0, 0.1]], dtype=np.float64)
        pl = Polyline(pts)
        bl = project_boundary_layer_onto_polyline(pl, 15, first_cell_height=0.001, growth_rate=1.2)
        assert bl.n_points == 15
        # First segment should be smallest
        segs = bl.segment_lengths()
        assert segs[0] < segs[-1]


# ============================================================
# Gap 4: Multi-block structured mesher
# ============================================================

from astraturbo.mesh.multiblock import (
    MultiBlockGenerator,
    MultiBlockMesh,
    generate_blade_passage_mesh,
)


class TestMultiBlockGenerator:
    def test_single_block(self):
        bottom = np.array([[0, 0], [1, 0]], dtype=np.float64)
        top = np.array([[0, 1], [1, 1]], dtype=np.float64)
        left = np.array([[0, 0], [0, 1]], dtype=np.float64)
        right = np.array([[1, 0], [1, 1]], dtype=np.float64)

        gen = MultiBlockGenerator()
        gen.add_block("test", bottom, top, left, right, n_i=10, n_j=10)
        mesh = gen.generate()

        assert mesh.n_blocks == 1
        assert mesh.blocks[0].points.shape == (11, 11, 2)
        assert mesh.total_cells == 100

    def test_multi_block(self):
        gen = MultiBlockGenerator()

        # Block 1
        gen.add_block("b1",
            np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 1]]),
            np.array([[0, 0], [0, 1]]), np.array([[1, 0], [1, 1]]),
            n_i=5, n_j=5)

        # Block 2 (shares right edge with block 1)
        gen.add_block("b2",
            np.array([[1, 0], [2, 0]]), np.array([[1, 1], [2, 1]]),
            np.array([[1, 0], [1, 1]]), np.array([[2, 0], [2, 1]]),
            n_i=5, n_j=5)

        mesh = gen.generate()
        assert mesh.n_blocks == 2
        assert mesh.total_cells == 50

    def test_graded_block(self):
        gen = MultiBlockGenerator()
        gen.add_block("graded",
            np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 1]]),
            np.array([[0, 0], [0, 1]]), np.array([[1, 0], [1, 1]]),
            n_i=10, n_j=10, grading_i=3.0)
        mesh = gen.generate()
        # Check that i-spacing is non-uniform
        block = mesh.blocks[0].points
        dx_first = np.linalg.norm(block[1, 0] - block[0, 0])
        dx_last = np.linalg.norm(block[-1, 0] - block[-2, 0])
        assert dx_last > dx_first

    def test_export_cgns(self, tmp_path):
        gen = MultiBlockGenerator()
        gen.add_block("test",
            np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 1]]),
            np.array([[0, 0], [0, 1]]), np.array([[1, 0], [1, 1]]),
            n_i=5, n_j=5)
        mesh = gen.generate()
        path = tmp_path / "test_multiblock.cgns"
        mesh.export_cgns(path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestBladePassageMesh:
    def test_generate(self):
        profile = _make_profile()
        mesh = generate_blade_passage_mesh(
            profile, pitch=0.05,
            n_blade=20, n_ogrid=5, n_inlet=5, n_outlet=5, n_passage=10,
            ogrid_thickness=0.005,
        )
        assert mesh.n_blocks > 0
        assert mesh.total_cells > 0

    def test_export_cgns(self, tmp_path):
        profile = _make_profile()
        mesh = generate_blade_passage_mesh(
            profile, pitch=0.05,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        )
        path = tmp_path / "blade_passage.cgns"
        mesh.export_cgns(path)
        assert path.exists()


# ============================================================
# Gap 5: Multi-stage orchestration
# ============================================================

from astraturbo.mesh.multistage import (
    RowMeshConfig,
    MultistageGenerator,
)


class TestMultistage:
    def test_single_row(self):
        profile = _make_profile()
        gen = MultistageGenerator()
        gen.add_row("rotor", RowMeshConfig(
            profile=profile, pitch=0.05,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        ))
        result = gen.generate()
        assert result.n_rows == 1
        assert result.total_cells > 0

    def test_rotor_stator(self):
        profile = _make_profile()
        gen = MultistageGenerator()
        gen.add_row("rotor", RowMeshConfig(
            profile=profile, pitch=0.05, is_rotor=True,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        ))
        gen.add_row("stator", RowMeshConfig(
            profile=profile, pitch=0.06, is_rotor=False,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        ))
        result = gen.generate()
        assert result.n_rows == 2
        assert result.total_blocks > 0

    def test_export_single_cgns(self, tmp_path):
        profile = _make_profile()
        gen = MultistageGenerator()
        gen.add_row("rotor", RowMeshConfig(
            profile=profile, pitch=0.05,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        ))
        gen.add_row("stator", RowMeshConfig(
            profile=profile, pitch=0.06,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        ))
        result = gen.generate()
        path = tmp_path / "stage.cgns"
        result.export_cgns(path)
        assert path.exists()

    def test_export_per_row(self, tmp_path):
        profile = _make_profile()
        gen = MultistageGenerator()
        gen.add_row("rotor", RowMeshConfig(
            profile=profile, pitch=0.05,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        ))
        gen.add_row("stator", RowMeshConfig(
            profile=profile, pitch=0.06,
            n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
        ))
        result = gen.generate()
        paths = result.export_cgns_per_row(tmp_path / "per_row")
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_three_row_compressor(self):
        profile = _make_profile()
        gen = MultistageGenerator()
        for name in ["igv", "rotor1", "stator1"]:
            gen.add_row(name, RowMeshConfig(
                profile=profile, pitch=0.05,
                n_blade=10, n_ogrid=3, n_inlet=3, n_outlet=3, n_passage=6,
            ))
        result = gen.generate()
        assert result.n_rows == 3
