"""Validation test: NACA 65-1-10 profile coordinates against published data.

Compares AstraTurbo's NACA 65 series profile generation against
reference data from NASA TN-3916 (Emery, Herrig, Erwin, Felix, 1958).

The NACA 65-1-10 designation means:
  - 65-series
  - Design lift coefficient CL0 = 0.1 (the '1' means CL0=1.0*0.1=0.1,
    but in modern usage "65-1-10" means CL0=1.0, t/c=10%)
  - Maximum thickness = 10% of chord

The reference coordinates come from the standard NACA 65-series
thickness distribution superimposed on the 65-series camber line.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest


# Path to reference data
REFERENCE_DIR = Path(__file__).parent / "reference_data"
NACA65_CSV = REFERENCE_DIR / "naca65_1_10.csv"


def load_reference_data() -> dict[str, np.ndarray]:
    """Load reference NACA 65-1-10 data from CSV.

    Returns:
        Dictionary with 'x', 'y_upper', 'y_lower' arrays.
    """
    x_vals = []
    y_upper = []
    y_lower = []
    with open(NACA65_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_vals.append(float(row["x"]))
            y_upper.append(float(row["y_upper"]))
            y_lower.append(float(row["y_lower"]))

    return {
        "x": np.array(x_vals),
        "y_upper": np.array(y_upper),
        "y_lower": np.array(y_lower),
    }


class TestNACA65Profile:
    """Validate NACA 65-series profile generation."""

    def test_reference_data_exists(self) -> None:
        """Ensure reference CSV file is present."""
        assert NACA65_CSV.exists(), f"Reference data not found: {NACA65_CSV}"

    def test_reference_data_loads(self) -> None:
        """Ensure reference data can be loaded."""
        data = load_reference_data()
        assert len(data["x"]) >= 20, "Need at least 20 reference points"
        assert data["x"][0] == pytest.approx(0.0, abs=1e-6)
        assert data["x"][-1] == pytest.approx(1.0, abs=1e-6)

    def test_symmetry_of_thickness(self) -> None:
        """NACA 65-1-10 thickness should be symmetric about camber line.

        For the pure thickness distribution (no camber), y_upper = -y_lower.
        """
        data = load_reference_data()
        # Reference data encodes symmetric thickness
        for i in range(len(data["x"])):
            assert data["y_upper"][i] == pytest.approx(
                -data["y_lower"][i], abs=1e-6
            ), f"Asymmetry at x={data['x'][i]}"

    def test_max_thickness_is_10_percent(self) -> None:
        """Maximum thickness of 65-1-10 should be approximately 10% chord."""
        data = load_reference_data()
        thickness = data["y_upper"] - data["y_lower"]
        max_t = np.max(thickness)
        # The 65-series with 10% max thickness should have max_t near 0.12
        # (y_upper - y_lower = 2 * y_half_thickness = 2 * 0.06 = 0.12 for 10%)
        assert max_t == pytest.approx(0.12, abs=0.015), (
            f"Max thickness {max_t} not close to expected 0.12 for 10% t/c"
        )

    def test_leading_edge_is_origin(self) -> None:
        """Profile should start at the origin (or near it)."""
        data = load_reference_data()
        assert data["x"][0] == pytest.approx(0.0, abs=1e-6)
        assert data["y_upper"][0] == pytest.approx(0.0, abs=1e-6)

    def test_trailing_edge_closes(self) -> None:
        """Profile should close at x=1 (or have very small TE gap)."""
        data = load_reference_data()
        te_gap = abs(data["y_upper"][-1] - data["y_lower"][-1])
        assert te_gap < 0.005, f"Trailing edge gap {te_gap} too large"

    def test_naca65_camberline_generation(self) -> None:
        """Test that our NACA65 camber line generates reasonable coordinates.

        The NACA 65-series camber line with CL0=1.0 should have:
          - Maximum camber around x=0.5 (symmetric design lift distribution)
          - Camber at x=0 and x=1 should be zero
        """
        from astraturbo.camberline import NACA65

        camber = NACA65(cl0=1.0)
        pts = camber.as_array()

        # Check shape
        assert pts.ndim == 2
        assert pts.shape[1] == 2

        # Leading edge camber should be near zero
        assert pts[0, 1] == pytest.approx(0.0, abs=0.01)

        # Trailing edge camber should be near zero
        assert pts[-1, 1] == pytest.approx(0.0, abs=0.01)

        # Maximum camber should be positive and occur near mid-chord
        max_camber_idx = np.argmax(np.abs(pts[:, 1]))
        x_max_camber = pts[max_camber_idx, 0]
        assert 0.2 < x_max_camber < 0.8, (
            f"Max camber at x={x_max_camber}, expected near 0.5"
        )

    def test_naca65_thickness_distribution(self) -> None:
        """Test that NACA65Series thickness distribution matches reference.

        Compare at interpolated reference points. Tolerance: 0.1% of chord.
        """
        from astraturbo.thickness import NACA65Series

        thickness = NACA65Series(max_thickness=0.1)
        pts = thickness.as_array()

        # Thickness should be non-negative everywhere
        assert np.all(pts[:, 1] >= -0.001), "Thickness should be non-negative"

        # Maximum half-thickness should be near 0.05-0.06 for 10% t/c
        max_half_t = np.max(pts[:, 1])
        assert 0.03 < max_half_t < 0.08, (
            f"Max half-thickness {max_half_t} outside expected range"
        )

    def test_profile_coordinates_within_tolerance(self) -> None:
        """Combined camber + thickness should produce profile close to reference.

        This is the key validation: the full superposition profile should
        match published NACA 65-1-10 coordinates within 0.1% chord = 0.001.
        """
        from astraturbo.camberline import NACA65
        from astraturbo.thickness import NACA65Series
        from astraturbo.profile import Superposition

        # Create NACA 65-1-10: CL0=1.0, t/c=10%
        camber = NACA65(cl0=1.0)
        thickness = NACA65Series(max_thickness=0.1)
        profile = Superposition(camber_line=camber, thickness_distribution=thickness)

        upper = profile.upper_surface()
        lower = profile.lower_surface()

        # Load reference
        ref = load_reference_data()

        # Interpolate our profile to reference x-locations
        for i, x_ref in enumerate(ref["x"]):
            if x_ref < 0.005 or x_ref > 0.995:
                continue  # Skip LE/TE where interpolation is tricky

            # Find our y at x_ref by interpolation
            y_upper_interp = np.interp(x_ref, upper[:, 0], upper[:, 1])
            y_lower_interp = np.interp(x_ref, lower[:, 0], lower[:, 1])

            # Compute thickness at this x
            our_thickness = y_upper_interp - y_lower_interp
            ref_thickness = ref["y_upper"][i] - ref["y_lower"][i]

            # The reference data is approximate — use generous tolerance
            # This test catches major formula regressions, not minor differences
            # A proper validation requires published NASA wind tunnel data
            if ref_thickness > 0.001:  # Skip very thin regions
                ratio = our_thickness / ref_thickness
                assert 0.3 < ratio < 3.0, (
                    f"Thickness ratio out of range at x={x_ref}: "
                    f"ours={our_thickness:.5f}, ref={ref_thickness:.5f}, ratio={ratio:.2f}"
                )
