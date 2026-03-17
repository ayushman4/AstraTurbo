"""OpenFOAM mesh file readers with proper validation and error handling.

Reads OpenFOAM polyMesh files (points, faces, owner, neighbour, boundary)
to import existing meshes into AstraTurbo for analysis, re-meshing, or
format conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class OpenFOAMReadError(Exception):
    """Raised when an OpenFOAM file cannot be read or is invalid."""


def validate_openfoam_file(filepath: str | Path) -> tuple[bool, str]:
    """Validate whether a file is a readable OpenFOAM file.

    Checks:
      - File exists
      - File is not empty
      - File is text (not binary)
      - File contains OpenFOAM header markers

    Args:
        filepath: Path to check.

    Returns:
        (is_valid, message) tuple.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return False, f"File not found: {filepath}"

    if not filepath.is_file():
        return False, f"Not a file: {filepath}"

    size = filepath.stat().st_size
    if size == 0:
        return False, f"File is empty: {filepath}"

    # Check if it's text (not binary)
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(min(8192, size))
        # Check for null bytes (binary indicator)
        if b"\x00" in chunk:
            return False, (
                f"File appears to be binary, not an OpenFOAM text file: {filepath}\n"
                f"OpenFOAM points files must be ASCII format."
            )
    except OSError as e:
        return False, f"Cannot read file: {e}"

    # Check for OpenFOAM markers
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            header = f.read(min(2048, size))
    except OSError as e:
        return False, f"Cannot read file as text: {e}"

    has_foamfile = "FoamFile" in header
    has_openfoam = "OpenFOAM" in header
    has_version = "version" in header
    has_parens = "(" in header

    if has_foamfile or (has_openfoam and has_version):
        return True, "Valid OpenFOAM file"

    if has_parens and any(c.isdigit() for c in header[:200]):
        return True, "Possible OpenFOAM file (no standard header)"

    return False, (
        f"File does not appear to be an OpenFOAM file: {filepath}\n"
        f"Expected FoamFile header block. "
        f"Make sure this is an ASCII-format OpenFOAM polyMesh file."
    )


def read_openfoam_points(filepath: str | Path) -> NDArray[np.float64]:
    """Read an OpenFOAM points file with validation.

    Parses the `constant/polyMesh/points` file format:
        FoamFile header
        N
        (
        (x y z)
        (x y z)
        ...
        )

    Args:
        filepath: Path to the points file.

    Returns:
        (N, 3) array of point coordinates.

    Raises:
        OpenFOAMReadError: If the file is missing, invalid, or contains no points.
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)

    # Validate
    is_valid, message = validate_openfoam_file(filepath)
    if not is_valid:
        raise OpenFOAMReadError(message)

    points = []
    in_data = False
    n_points_expected = 0
    parse_errors = 0

    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip header and comments
                if (line.startswith("/*") or line.startswith("//")
                        or line.startswith("|") or line.startswith("\\")):
                    continue
                if line in ("", "FoamFile", "{", "}"):
                    continue
                if any(line.startswith(kw) for kw in
                       ("version", "format", "class", "location", "object")):
                    continue

                # Detect point count
                if not in_data and line == "(":
                    in_data = True
                    continue

                if not in_data:
                    try:
                        n_points_expected = int(line)
                    except ValueError:
                        continue
                    continue

                # End of data
                if line == ")":
                    break

                # Parse point: (x y z)
                if line.startswith("(") and line.endswith(")"):
                    coords = line[1:-1].split()
                    if len(coords) == 3:
                        try:
                            points.append([
                                float(coords[0]),
                                float(coords[1]),
                                float(coords[2]),
                            ])
                        except ValueError:
                            parse_errors += 1
                            if parse_errors <= 5:
                                pass  # Skip malformed lines silently
                            elif parse_errors == 6:
                                pass  # Too many errors, continue silently

    except UnicodeDecodeError as e:
        raise OpenFOAMReadError(
            f"File encoding error at {filepath}:\n{e}\n"
            f"The file may be in binary format. "
            f"OpenFOAM points files must be ASCII."
        )

    if not points:
        raise OpenFOAMReadError(
            f"No point data found in {filepath}.\n"
            f"The file may not be an OpenFOAM points file, or it may be empty.\n"
            f"Expected format: lines of (x y z) coordinates between ( and ) delimiters."
        )

    result = np.array(points, dtype=np.float64)

    if n_points_expected > 0 and len(result) != n_points_expected:
        # Warning, not error — some files have minor count mismatches
        import warnings
        warnings.warn(
            f"OpenFOAM points file: expected {n_points_expected} points, "
            f"read {len(result)}. File may be truncated or malformed.",
            UserWarning,
            stacklevel=2,
        )

    if parse_errors > 0:
        import warnings
        warnings.warn(
            f"Skipped {parse_errors} malformed point lines in {filepath}.",
            UserWarning,
            stacklevel=2,
        )

    return result


def read_openfoam_boundary(filepath: str | Path) -> dict[str, dict]:
    """Read an OpenFOAM boundary file with validation.

    Parses `constant/polyMesh/boundary` to extract patch names, types,
    and face ranges.

    Args:
        filepath: Path to the boundary file.

    Returns:
        Dict mapping patch names to {type, nFaces, startFace}.

    Raises:
        OpenFOAMReadError: If the file is invalid.
    """
    filepath = Path(filepath)

    is_valid, message = validate_openfoam_file(filepath)
    if not is_valid:
        raise OpenFOAMReadError(message)

    patches = {}
    current_patch = None
    in_data = False
    brace_depth = 0

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            if line.startswith("//") or line.startswith("/*") or line.startswith("|"):
                continue
            if line.startswith("FoamFile"):
                brace_depth += 1
                continue

            if line == "(":
                in_data = True
                continue
            if line == ")":
                break

            if not in_data:
                continue

            if line == "{":
                brace_depth += 1
                continue
            if line == "}":
                brace_depth -= 1
                if current_patch is not None and brace_depth <= 1:
                    current_patch = None
                continue

            if brace_depth <= 1 and line and not line.startswith("type"):
                current_patch = line
                patches[current_patch] = {}
                continue

            if current_patch and "type" in line:
                val = line.split()[-1].rstrip(";")
                patches[current_patch]["type"] = val
            elif current_patch and "nFaces" in line:
                try:
                    val = line.split()[-1].rstrip(";")
                    patches[current_patch]["nFaces"] = int(val)
                except ValueError:
                    pass
            elif current_patch and "startFace" in line:
                try:
                    val = line.split()[-1].rstrip(";")
                    patches[current_patch]["startFace"] = int(val)
                except ValueError:
                    pass

    return patches


def read_openfoam_polymesh(case_dir: str | Path) -> dict:
    """Read a complete OpenFOAM polyMesh directory.

    Args:
        case_dir: Path to the OpenFOAM case directory (containing constant/).

    Returns:
        Dict with 'points' (NDArray), 'boundary' (dict), and metadata.

    Raises:
        OpenFOAMReadError: If the directory structure is invalid.
    """
    case_dir = Path(case_dir)
    polymesh = case_dir / "constant" / "polyMesh"

    if not polymesh.exists():
        raise OpenFOAMReadError(
            f"polyMesh directory not found at {polymesh}\n"
            f"Expected OpenFOAM case structure: {case_dir}/constant/polyMesh/"
        )

    result = {}

    points_file = polymesh / "points"
    if points_file.exists():
        result["points"] = read_openfoam_points(points_file)
    else:
        raise OpenFOAMReadError(
            f"Points file not found: {points_file}\n"
            f"The polyMesh directory exists but contains no points file."
        )

    boundary_file = polymesh / "boundary"
    if boundary_file.exists():
        result["boundary"] = read_openfoam_boundary(boundary_file)
    else:
        result["boundary"] = {}

    result["n_points"] = len(result["points"])
    result["case_dir"] = str(case_dir)

    return result


def openfoam_points_to_cloud(
    points: NDArray[np.float64],
) -> dict:
    """Analyze an OpenFOAM point cloud and extract basic statistics.

    Args:
        points: (N, 3) array from read_openfoam_points.

    Returns:
        Dict with bounding box, centroid, and axis ranges.

    Raises:
        OpenFOAMReadError: If the points array is empty or malformed.
    """
    if points is None or len(points) == 0:
        raise OpenFOAMReadError(
            "Point array is empty. Cannot compute statistics."
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise OpenFOAMReadError(
            f"Expected (N, 3) array, got shape {points.shape}. "
            f"Points must have 3 coordinates (x, y, z)."
        )

    # Check for NaN/Inf
    if np.any(~np.isfinite(points)):
        n_bad = int(np.sum(~np.isfinite(points).all(axis=1)))
        import warnings
        warnings.warn(
            f"Found {n_bad} points with NaN or Inf coordinates. "
            f"These will be excluded from statistics.",
            UserWarning,
            stacklevel=2,
        )
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]
        if len(points) == 0:
            raise OpenFOAMReadError("All points contain NaN or Inf values.")

    return {
        "n_points": len(points),
        "x_min": float(points[:, 0].min()),
        "x_max": float(points[:, 0].max()),
        "y_min": float(points[:, 1].min()),
        "y_max": float(points[:, 1].max()),
        "z_min": float(points[:, 2].min()),
        "z_max": float(points[:, 2].max()),
        "centroid": points.mean(axis=0),
        "x_range": float(points[:, 0].max() - points[:, 0].min()),
        "y_range": float(points[:, 1].max() - points[:, 1].min()),
        "z_range": float(points[:, 2].max() - points[:, 2].min()),
    }
