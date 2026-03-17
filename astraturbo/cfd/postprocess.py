"""CFD post-processing — read and analyze simulation results."""

from __future__ import annotations

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
