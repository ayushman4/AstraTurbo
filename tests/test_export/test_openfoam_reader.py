"""Tests for OpenFOAM reader validation and error handling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from astraturbo.export.openfoam_reader import (
    OpenFOAMReadError,
    validate_openfoam_file,
    read_openfoam_points,
    openfoam_points_to_cloud,
)


class TestValidation:
    def test_nonexistent_file(self):
        valid, msg = validate_openfoam_file("/nonexistent/path/points")
        assert not valid
        assert "not found" in msg.lower()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty"
        f.write_text("")
        valid, msg = validate_openfoam_file(f)
        assert not valid
        assert "empty" in msg.lower()

    def test_binary_file(self, tmp_path):
        f = tmp_path / "binary"
        f.write_bytes(b"\x00\x01\x02\x03\x04binary data")
        valid, msg = validate_openfoam_file(f)
        assert not valid
        assert "binary" in msg.lower()

    def test_random_text_file(self, tmp_path):
        f = tmp_path / "random.txt"
        f.write_text("This is just a random text file\nwith no OpenFOAM content\n")
        valid, msg = validate_openfoam_file(f)
        assert not valid
        assert "OpenFOAM" in msg

    def test_valid_openfoam_header(self, tmp_path):
        f = tmp_path / "points"
        f.write_text(
            "FoamFile\n{\n    version 2.0;\n    class vectorField;\n}\n"
            "2\n(\n(1 2 3)\n(4 5 6)\n)\n"
        )
        valid, msg = validate_openfoam_file(f)
        assert valid

    def test_directory_not_file(self, tmp_path):
        valid, msg = validate_openfoam_file(tmp_path)
        assert not valid
        assert "Not a file" in msg


class TestReadErrors:
    def test_read_nonexistent(self):
        with pytest.raises(OpenFOAMReadError, match="not found"):
            read_openfoam_points("/nonexistent/file")

    def test_read_binary(self, tmp_path):
        f = tmp_path / "binary"
        f.write_bytes(b"\x00\x01\x02data")
        with pytest.raises(OpenFOAMReadError, match="binary"):
            read_openfoam_points(f)

    def test_read_no_points(self, tmp_path):
        f = tmp_path / "points"
        f.write_text("FoamFile\n{\n    version 2.0;\n}\n0\n(\n)\n")
        with pytest.raises(OpenFOAMReadError, match="No point data"):
            read_openfoam_points(f)

    def test_read_valid_file(self, tmp_path):
        f = tmp_path / "points"
        f.write_text(
            "FoamFile\n{\n    version 2.0;\n    class vectorField;\n}\n"
            "3\n(\n(0 0 0)\n(1 0 0)\n(0 1 0)\n)\n"
        )
        pts = read_openfoam_points(f)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[1], [1, 0, 0])

    def test_read_with_count_mismatch(self, tmp_path):
        f = tmp_path / "points"
        f.write_text(
            "FoamFile\n{\n    version 2.0;\n}\n"
            "5\n(\n(0 0 0)\n(1 0 0)\n)\n"
        )
        with pytest.warns(UserWarning, match="expected 5.*read 2"):
            pts = read_openfoam_points(f)
        assert len(pts) == 2


class TestPointCloudErrors:
    def test_empty_array(self):
        with pytest.raises(OpenFOAMReadError, match="empty"):
            openfoam_points_to_cloud(np.empty((0, 3)))

    def test_wrong_shape(self):
        with pytest.raises(OpenFOAMReadError, match="shape"):
            openfoam_points_to_cloud(np.array([[1, 2], [3, 4]]))

    def test_nan_handling(self):
        pts = np.array([[1, 2, 3], [float("nan"), 0, 0], [4, 5, 6]])
        with pytest.warns(UserWarning, match="NaN"):
            stats = openfoam_points_to_cloud(pts)
        assert stats["n_points"] == 2  # NaN row excluded

    def test_valid_cloud(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        stats = openfoam_points_to_cloud(pts)
        assert stats["n_points"] == 3
        assert stats["x_range"] == pytest.approx(1.0)


class TestRealFile:
    """Test against the actual OpenFOAM points file if available."""

    REAL_FILE = Path("/Users/ayudutta/Downloads/points.txt")

    @pytest.mark.skipif(
        not REAL_FILE.exists(), reason="Real test file not available"
    )
    def test_validate_real_file(self):
        valid, msg = validate_openfoam_file(self.REAL_FILE)
        assert valid

    @pytest.mark.skipif(
        not REAL_FILE.exists(), reason="Real test file not available"
    )
    def test_read_real_file(self):
        pts = read_openfoam_points(self.REAL_FILE)
        assert pts.shape == (59809, 3)
        stats = openfoam_points_to_cloud(pts)
        assert stats["n_points"] == 59809
        assert stats["x_range"] > 0
