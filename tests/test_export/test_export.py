"""Tests for export writers."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import h5py

from astraturbo.export.cgns_writer import write_cgns_structured, write_cgns_2d
from astraturbo.export.openfoam_writer import write_blockmeshdict


class TestCGNSWriter:
    def test_write_single_block(self, tmp_path):
        block = np.zeros((5, 5, 3), dtype=np.float64)
        for i in range(5):
            for j in range(5):
                block[i, j] = [float(i), float(j), 0.0]

        filepath = tmp_path / "test.cgns"
        write_cgns_structured(filepath, [block], ["TestZone"])

        assert filepath.exists()
        assert filepath.stat().st_size > 0

        # Verify it's valid HDF5
        with h5py.File(filepath, "r") as f:
            assert "AstraTurbo" in f

    def test_write_2d_blocks(self, tmp_path):
        b1 = np.zeros((6, 4, 2), dtype=np.float64)
        for i in range(6):
            for j in range(4):
                b1[i, j] = [float(i) * 0.01, 0.1 + float(j) * 0.01]

        filepath = tmp_path / "test_2d.cgns"
        write_cgns_2d(filepath, [b1], ["Inlet"])

        assert filepath.exists()

    def test_write_multiple_blocks(self, tmp_path):
        blocks = []
        for k in range(3):
            b = np.zeros((4, 4, 3), dtype=np.float64)
            for i in range(4):
                for j in range(4):
                    b[i, j] = [float(i) + k * 4, float(j), 0.0]
            blocks.append(b)

        filepath = tmp_path / "multi.cgns"
        write_cgns_structured(filepath, blocks)
        assert filepath.exists()


class TestOpenFOAMWriter:
    def test_write_blockmeshdict(self, tmp_path):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)

        blocks = [{
            "vertices": [0, 1, 2, 3, 4, 5, 6, 7],
            "cells": [10, 10, 1],
            "grading": [1, 1, 1],
        }]

        patches = [
            {"name": "inlet", "type": "patch", "faces": [[0, 3, 7, 4]]},
            {"name": "outlet", "type": "patch", "faces": [[1, 2, 6, 5]]},
            {"name": "walls", "type": "wall", "faces": [[0, 1, 5, 4], [3, 2, 6, 7]]},
            {"name": "frontAndBack", "type": "empty", "faces": [[0, 1, 2, 3], [4, 5, 6, 7]]},
        ]

        filepath = tmp_path / "blockMeshDict"
        write_blockmeshdict(filepath, vertices, blocks, patches)

        assert filepath.exists()
        content = filepath.read_text()
        assert "FoamFile" in content
        assert "vertices" in content
        assert "blocks" in content
        assert "boundary" in content
        assert "AstraTurbo" in content
