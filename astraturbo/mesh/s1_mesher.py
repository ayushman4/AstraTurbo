"""S1 blade-to-blade surface mesh generator.

Generates a 2D structured mesh on the S1 (blade-to-blade) surface at a
constant radius. The S1 surface is a cylindrical cut through the blade
passage — it's the "unrolled" view at a specific span position.

Used for 2D cascade CFD simulations and blade-to-blade flow analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .transfinite import tfi_2d_graded


@dataclass
class S1MeshConfig:
    """Configuration for S1 blade-to-blade mesh generation."""

    n_streamwise: int = 40       # Cells in streamwise (axial) direction
    n_pitchwise: int = 30        # Cells in pitchwise (circumferential) direction
    n_inlet: int = 15            # Cells upstream of LE
    n_outlet: int = 15           # Cells downstream of TE
    grading_wall: float = 1.3    # Grading toward blade surface
    grading_inlet: float = 0.5   # Cluster toward blade LE
    grading_outlet: float = 2.0  # Cluster toward blade TE
    inlet_fraction: float = 0.5  # Inlet length as fraction of chord
    outlet_fraction: float = 0.5 # Outlet length as fraction of chord


@dataclass
class S1Block:
    """A single block in the S1 mesh."""

    points: NDArray[np.float64]  # (Ni, Nj, 2) in (theta, z) or (x, y) coords
    name: str = ""


class S1Mesher:
    """Generates a 2D structured mesh on the S1 blade-to-blade surface.

    The S1 surface is a cylindrical cut at constant radius r. In the
    "unrolled" view, the coordinates are:
      - axial (z): streamwise direction
      - circumferential (r*theta): pitchwise direction

    The mesh consists of blocks around and between the blade profiles
    at this span position.

    Usage::

        mesher = S1Mesher(config)
        blocks = mesher.generate(
            profile_2d=profile.as_array(),
            pitch=0.05,
            radius=0.15,
        )
    """

    def __init__(self, config: S1MeshConfig | None = None) -> None:
        self.config = config or S1MeshConfig()
        self.blocks: list[S1Block] = []

    def generate(
        self,
        profile_2d: NDArray[np.float64],
        pitch: float,
        radius: float = 1.0,
        stagger_angle: float = 0.0,
    ) -> list[S1Block]:
        """Generate S1 blade-to-blade mesh.

        Args:
            profile_2d: (N, 2) closed blade profile in normalized coords.
            pitch: Blade pitch (circumferential spacing) in meters.
            radius: Radius of the S1 cut in meters.
            stagger_angle: Blade stagger angle in radians.

        Returns:
            List of S1Block objects forming the blade-to-blade mesh.
        """
        cfg = self.config

        # Scale and rotate profile
        profile = profile_2d.copy()

        # Apply stagger rotation
        if abs(stagger_angle) > 1e-10:
            cos_g = np.cos(stagger_angle)
            sin_g = np.sin(stagger_angle)
            x_rot = profile[:, 0] * cos_g - profile[:, 1] * sin_g
            y_rot = profile[:, 0] * sin_g + profile[:, 1] * cos_g
            profile = np.column_stack((x_rot, y_rot))

        # Blade extents
        x_min = profile[:, 0].min()
        x_max = profile[:, 0].max()
        chord = x_max - x_min
        y_mid = profile[:, 1].mean()

        # Domain extents
        x_inlet = x_min - cfg.inlet_fraction * chord
        x_outlet = x_max + cfg.outlet_fraction * chord
        y_lower = y_mid - pitch / 2
        y_upper = y_mid + pitch / 2

        # Split profile into suction and pressure sides
        le_idx = int(np.argmin(profile[:, 0]))
        n_total = len(profile)
        n_half = (n_total + 1) // 2

        # Upper passage (suction side to upper periodic)
        upper_block = self._generate_passage_block(
            "upper_passage",
            x_inlet, x_outlet,
            profile, y_upper,
            cfg.n_streamwise, cfg.n_pitchwise // 2,
            cfg.grading_inlet, cfg.grading_outlet,
            side="upper",
        )

        # Lower passage (pressure side to lower periodic)
        lower_block = self._generate_passage_block(
            "lower_passage",
            x_inlet, x_outlet,
            profile, y_lower,
            cfg.n_streamwise, cfg.n_pitchwise // 2,
            cfg.grading_inlet, cfg.grading_outlet,
            side="lower",
        )

        # Inlet block
        inlet_block = self._generate_inlet_block(
            x_inlet, x_min, y_lower, y_upper,
            cfg.n_inlet, cfg.n_pitchwise,
            cfg.grading_inlet,
        )

        # Outlet block
        outlet_block = self._generate_outlet_block(
            x_max, x_outlet, y_lower, y_upper,
            cfg.n_outlet, cfg.n_pitchwise,
            cfg.grading_outlet,
        )

        self.blocks = [inlet_block, upper_block, lower_block, outlet_block]
        return self.blocks

    def _generate_passage_block(
        self, name, x_inlet, x_outlet,
        profile, y_periodic,
        n_stream, n_pitch,
        grading_inlet, grading_outlet,
        side="upper",
    ) -> S1Block:
        """Generate a passage block between blade and periodic boundary."""
        ni = n_stream + 1
        nj = n_pitch + 1

        # Resample blade surface to ni points in streamwise direction
        le_idx = int(np.argmin(profile[:, 0]))
        n_total = len(profile)
        n_half = (n_total + 1) // 2

        if side == "upper":
            # Suction side: from TE (index 0) to LE
            blade_pts = profile[:le_idx + 1]
        else:
            # Pressure side: from LE to TE
            blade_pts = profile[le_idx:]

        # Resample to ni points
        t_orig = np.linspace(0, 1, len(blade_pts))
        t_new = np.linspace(0, 1, ni)
        blade_resampled = np.column_stack((
            np.interp(t_new, t_orig, blade_pts[:, 0]),
            np.interp(t_new, t_orig, blade_pts[:, 1]),
        ))

        # Periodic boundary (straight line at y_periodic)
        periodic = np.column_stack((
            blade_resampled[:, 0],
            np.full(ni, y_periodic),
        ))

        # Left boundary (radial line from blade to periodic at inlet)
        left = np.column_stack((
            np.full(nj, blade_resampled[0, 0]),
            np.linspace(blade_resampled[0, 1], y_periodic, nj),
        ))

        # Right boundary (radial line at outlet)
        right = np.column_stack((
            np.full(nj, blade_resampled[-1, 0]),
            np.linspace(blade_resampled[-1, 1], y_periodic, nj),
        ))

        points = tfi_2d_graded(
            blade_resampled, periodic, left, right,
            grading_t=grading_inlet,
            n_cells_s=n_stream,
            n_cells_t=n_pitch,
        )

        return S1Block(points=points, name=name)

    def _generate_inlet_block(
        self, x_start, x_end, y_lower, y_upper,
        n_axial, n_pitch, grading,
    ) -> S1Block:
        """Generate inlet H-block upstream of blade."""
        ni = n_axial + 1
        nj = n_pitch + 1

        bottom = np.column_stack((
            np.linspace(x_start, x_end, ni),
            np.full(ni, y_lower),
        ))
        top = np.column_stack((
            np.linspace(x_start, x_end, ni),
            np.full(ni, y_upper),
        ))
        left = np.column_stack((
            np.full(nj, x_start),
            np.linspace(y_lower, y_upper, nj),
        ))
        right = np.column_stack((
            np.full(nj, x_end),
            np.linspace(y_lower, y_upper, nj),
        ))

        points = tfi_2d_graded(
            bottom, top, left, right,
            grading_s=grading,
            n_cells_s=n_axial,
            n_cells_t=n_pitch,
        )

        return S1Block(points=points, name="inlet")

    def _generate_outlet_block(
        self, x_start, x_end, y_lower, y_upper,
        n_axial, n_pitch, grading,
    ) -> S1Block:
        """Generate outlet H-block downstream of blade."""
        ni = n_axial + 1
        nj = n_pitch + 1

        bottom = np.column_stack((
            np.linspace(x_start, x_end, ni),
            np.full(ni, y_lower),
        ))
        top = np.column_stack((
            np.linspace(x_start, x_end, ni),
            np.full(ni, y_upper),
        ))
        left = np.column_stack((
            np.full(nj, x_start),
            np.linspace(y_lower, y_upper, nj),
        ))
        right = np.column_stack((
            np.full(nj, x_end),
            np.linspace(y_lower, y_upper, nj),
        ))

        points = tfi_2d_graded(
            bottom, top, left, right,
            grading_s=grading,
            n_cells_s=n_axial,
            n_cells_t=n_pitch,
        )

        return S1Block(points=points, name="outlet")

    def get_all_points(self) -> NDArray[np.float64]:
        """Return all mesh points as (N, 2) array."""
        if not self.blocks:
            return np.empty((0, 2))
        all_pts = []
        for block in self.blocks:
            all_pts.append(block.points.reshape(-1, 2))
        return np.concatenate(all_pts, axis=0)

    def total_cells(self) -> int:
        """Return total cell count across all blocks."""
        total = 0
        for block in self.blocks:
            ni, nj = block.points.shape[0] - 1, block.points.shape[1] - 1
            total += ni * nj
        return total
