"""Tests for FEA integration."""

import pytest
import numpy as np
from pathlib import Path

from astraturbo.fea import (
    Material,
    get_material,
    list_materials,
    write_calculix_input,
    blade_surface_to_solid_mesh,
    map_cfd_pressure_to_fea,
    identify_root_nodes,
    export_fea_mesh_abaqus,
    FEAWorkflow,
    FEAWorkflowConfig,
)


class TestMaterialDatabase:
    def test_list_materials(self):
        mats = list_materials()
        assert len(mats) >= 6
        assert "inconel_718" in mats
        assert "ti_6al_4v" in mats

    def test_get_material(self):
        mat = get_material("inconel_718")
        assert mat.density == pytest.approx(8190, abs=10)
        assert mat.youngs_modulus > 100e9

    def test_unknown_material(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_material("unobtanium")

    def test_calculix_format(self):
        mat = get_material("ti_6al_4v")
        text = mat.to_calculix_format()
        assert "*MATERIAL" in text
        assert "*ELASTIC" in text
        assert "*DENSITY" in text
        assert "Ti-6Al-4V" in text


class TestSolidMesh:
    def _make_surface(self):
        ni, nj = 10, 5
        x = np.linspace(0, 0.1, ni)
        z = np.linspace(0, 0.05, nj)
        xx, zz = np.meshgrid(x, z, indexing="ij")
        yy = 0.01 * np.sin(np.pi * xx / 0.1)  # Curved blade
        pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        return pts, ni, nj

    def test_extrude_to_solid(self):
        pts, ni, nj = self._make_surface()
        nodes, elements = blade_surface_to_solid_mesh(pts, ni, nj, thickness=0.002)

        assert nodes.shape[1] == 3
        assert nodes.shape[0] == 2 * ni * nj  # Outer + inner
        assert elements.shape[1] == 8  # Hex elements
        assert len(elements) == (ni - 1) * (nj - 1)

    def test_identify_root(self):
        pts, ni, nj = self._make_surface()
        nodes, _ = blade_surface_to_solid_mesh(pts, ni, nj)
        root = identify_root_nodes(nodes, axis=2)
        assert len(root) > 0
        for nid in root:
            assert nodes[nid, 2] == pytest.approx(0.0, abs=1e-3)

    def test_pressure_mapping(self):
        cfd_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        cfd_pressure = np.array([100000, 120000, 110000], dtype=np.float64)
        fea_pts = np.array([[0.1, 0.1, 0], [0.9, 0.1, 0]], dtype=np.float64)

        mapped = map_cfd_pressure_to_fea(cfd_pts, cfd_pressure, fea_pts)
        assert len(mapped) == 2
        assert all(p > 0 for p in mapped)


class TestCalculiXOutput:
    def test_write_input(self, tmp_path):
        nodes = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

        path = tmp_path / "test.inp"
        write_calculix_input(
            path, nodes, elements,
            omega=1200.0, fixed_nodes=[0, 1, 2, 3],
        )

        assert path.exists()
        content = path.read_text()
        assert "*NODE" in content
        assert "*ELEMENT" in content
        assert "*BOUNDARY" in content
        assert "CENTRIF" in content
        assert "1200" in content

    def test_abaqus_mesh_export(self, tmp_path):
        nodes = np.random.rand(20, 3)
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        path = tmp_path / "mesh.inp"
        export_fea_mesh_abaqus(path, nodes, elements)
        assert path.exists()
        content = path.read_text()
        assert "*NODE" in content
        assert "C3D8" in content


class TestFEAWorkflow:
    def test_setup(self, tmp_path):
        ni, nj = 8, 4
        x = np.linspace(0, 0.1, ni)
        z = np.linspace(0, 0.05, nj)
        xx, zz = np.meshgrid(x, z, indexing="ij")
        yy = 0.005 * np.sin(np.pi * xx / 0.1)
        pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        cfg = FEAWorkflowConfig(omega=1200.0)
        wf = FEAWorkflow(cfg)
        wf.set_blade_surface(pts, ni, nj)
        case = wf.setup(tmp_path / "fea_case")

        assert (case / "blade.inp").exists()
        assert (case / "run_fea.sh").exists()
        content = (case / "blade.inp").read_text()
        assert "*NODE" in content
        assert "Inconel_718" in content

    def test_analytical_stress(self, tmp_path):
        ni, nj = 8, 4
        x = np.linspace(0, 0.1, ni)
        z = np.linspace(0.1, 0.3, nj)  # Radial span from 100mm to 300mm
        xx, zz = np.meshgrid(x, z, indexing="ij")
        yy = np.zeros_like(xx)
        pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        cfg = FEAWorkflowConfig(omega=1200.0)
        wf = FEAWorkflow(cfg)
        wf.set_blade_surface(pts, ni, nj)

        estimate = wf.estimate_stress_analytical()
        assert "centrifugal_stress_MPa" in estimate
        assert "safety_factor" in estimate
        assert estimate["centrifugal_stress_MPa"] > 0
        assert estimate["material"] == "Inconel_718"
