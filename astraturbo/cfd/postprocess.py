"""CFD post-processing — read and analyze simulation results."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_openfoam_residuals(log_file: str | Path) -> dict[str, NDArray[np.float64]]:
    """Parse residuals from an OpenFOAM solver log file.

    Args:
        log_file: Path to the solver log.

    Returns:
        Dict mapping field names to arrays of residual values per iteration.
    """
    log_file = Path(log_file)
    residuals: dict[str, list[float]] = {}

    if not log_file.exists():
        return {}

    with open(log_file) as f:
        for line in f:
            # Pattern: "Solving for U, Initial residual = 0.123..."
            if "Solving for" in line and "Initial residual" in line:
                parts = line.split(",")
                field = parts[0].split("Solving for")[-1].strip()
                for part in parts:
                    if "Initial residual" in part:
                        try:
                            val = float(part.split("=")[-1].strip())
                            residuals.setdefault(field, []).append(val)
                        except ValueError:
                            pass

    return {k: np.array(v) for k, v in residuals.items()}


def find_latest_time_dir(case_dir: str | Path) -> Path | None:
    """Find the latest time directory in an OpenFOAM case.

    Scans for numeric directory names (e.g. 100, 200, 1000) and returns
    the one with the highest value. Skips '0' (initial conditions).

    Returns:
        Path to latest time directory, or None if no solution found.
    """
    case_dir = Path(case_dir)
    time_dirs = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    time_dirs.append((t, d))
            except ValueError:
                continue
    if not time_dirs:
        return None
    time_dirs.sort(key=lambda x: x[0])
    return time_dirs[-1][1]


def read_openfoam_field(field_path: str | Path) -> NDArray[np.float64] | None:
    """Read an OpenFOAM field file (volScalarField or volVectorField).

    Handles both 'uniform' and 'nonuniform List<scalar/vector>' formats.

    Returns:
        numpy array of shape (N,) for scalar or (N,3) for vector fields.
        Returns None if the file cannot be parsed.
    """
    field_path = Path(field_path)
    if not field_path.exists():
        return None

    text = field_path.read_text()

    # Detect field class
    is_vector = "volVectorField" in text

    # Find internalField
    # Pattern: internalField   nonuniform List<scalar>  N  (  ... );
    # or:      internalField   uniform 101325;
    # or:      internalField   uniform (0 0 0);
    m = re.search(r"internalField\s+uniform\s+\(([^)]+)\)", text)
    if m and is_vector:
        vals = [float(x) for x in m.group(1).split()]
        return np.array([vals])

    m = re.search(r"internalField\s+uniform\s+([\d.eE+-]+)", text)
    if m and not is_vector:
        return np.array([float(m.group(1))])

    # Nonuniform list
    m = re.search(r"internalField\s+nonuniform\s+List<(?:scalar|vector)>\s+(\d+)\s*\(", text)
    if not m:
        return None

    n_values = int(m.group(1))
    start = m.end()

    if is_vector:
        # Parse vector list: (vx vy vz)
        pattern = re.compile(r"\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)")
        matches = pattern.findall(text[start:])
        if len(matches) < n_values:
            return None
        data = np.array([[float(a), float(b), float(c)] for a, b, c in matches[:n_values]])
        return data
    else:
        # Parse scalar list: one value per line
        # Find all numbers after the opening paren
        remaining = text[start:]
        # Find closing paren
        end = remaining.find(")")
        if end == -1:
            return None
        block = remaining[:end]
        values = re.findall(r"[\d.eE+-]+", block)
        if len(values) < n_values:
            return None
        return np.array([float(v) for v in values[:n_values]])


def read_openfoam_points(case_dir: str | Path) -> NDArray[np.float64] | None:
    """Read mesh points from an OpenFOAM case (constant/polyMesh/points).

    Returns:
        (N, 3) array of point coordinates, or None.
    """
    pts_path = Path(case_dir) / "constant" / "polyMesh" / "points"
    if not pts_path.exists():
        return None

    text = pts_path.read_text()
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        return None

    n_points = int(m.group(1))
    start = m.end()
    pattern = re.compile(r"\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)")
    matches = pattern.findall(text[start:])
    if len(matches) < n_points:
        return None
    return np.array([[float(a), float(b), float(c)] for a, b, c in matches[:n_points]])


def read_openfoam_solution(case_dir: str | Path) -> dict | None:
    """Read a complete OpenFOAM solution from the latest time directory.

    Returns a dict with:
        - 'points': (N, 3) mesh point coordinates
        - 'p': (N,) pressure field
        - 'U': (N, 3) velocity field
        - 'T': (N,) temperature field (if present)
        - 'time': the solution time
        - 'case_dir': path to the case

    Returns None if no solution is found.
    """
    case_dir = Path(case_dir)
    time_dir = find_latest_time_dir(case_dir)
    if time_dir is None:
        return None

    points = read_openfoam_points(case_dir)

    result = {
        "time": float(time_dir.name),
        "case_dir": str(case_dir),
        "points": points,
    }

    for field_name in ("p", "U", "T", "k", "omega"):
        field_path = time_dir / field_name
        if field_path.exists():
            result[field_name] = read_openfoam_field(field_path)

    return result


def compute_performance_map(
    pressure_ratios: NDArray[np.float64],
    mass_flows: NDArray[np.float64],
    efficiencies: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Structure performance map data for plotting.

    Args:
        pressure_ratios: Array of total pressure ratios.
        mass_flows: Array of corrected mass flows.
        efficiencies: Array of isentropic efficiencies.

    Returns:
        Dict with 'pressure_ratio', 'mass_flow', 'efficiency' arrays.
    """
    return {
        "pressure_ratio": np.asarray(pressure_ratios),
        "mass_flow": np.asarray(mass_flows),
        "efficiency": np.asarray(efficiencies),
    }
