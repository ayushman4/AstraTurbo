"""Streamline Curvature Method (SCM) mesher for S2m plane.

Ported from V1 SCMMesher.py. Generates a 2D structured mesh in the
meridional (S2m) plane for throughflow analysis. The mesh consists of
3 blocks per blade row:
  - Block 1: Inlet (upstream boundary to leading edge)
  - Block 2: Blade passage (leading edge to trailing edge)
  - Block 3: Outlet (trailing edge to downstream boundary)

Each block is generated via transfinite interpolation between hub and
shroud contours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .transfinite import tfi_2d_graded


@dataclass
class SCMBlock:
    """A single block in the SCM mesh."""

    points: NDArray[np.float64] | None = None  # (Ni, Nj, 2) in (z, r)
    n_axial: int = 20
    n_radial: int = 10
    grading_axial: float = 1.0
    grading_radial: float = 1.0


@dataclass
class SCMMeshConfig:
    """Configuration for SCM mesh generation."""

    # Cells per block
    n_inlet_axial: int = 15
    n_blade_axial: int = 30
    n_outlet_axial: int = 15
    n_radial: int = 20

    # Grading ratios
    grading_inlet: float = 0.5     # Cluster toward blade LE
    grading_blade: float = 1.0     # Uniform in passage
    grading_outlet: float = 2.0    # Cluster toward blade TE
    grading_radial: float = 1.0    # Radial distribution

    # Domain extents (fraction of blade chord)
    inlet_length_fraction: float = 0.5
    outlet_length_fraction: float = 0.5


class SCMMesher:
    """Generates a structured 2D mesh in the S2m (meridional) plane.

    The S2m plane uses coordinates (z, r) where z is axial and r is radial.
    The mesh captures the meridional flow channel from hub to shroud through
    the blade passage.
    """

    def __init__(self, config: SCMMeshConfig | None = None) -> None:
        self.config = config or SCMMeshConfig()
        self.blocks: list[SCMBlock] = []

    def generate(
        self,
        hub_contour: NDArray[np.float64],
        shroud_contour: NDArray[np.float64],
        le_z: float,
        te_z: float,
    ) -> list[SCMBlock]:
        """Generate the 3-block SCM mesh.

        Args:
            hub_contour: (N, 2) array of [z, r] points for hub.
            shroud_contour: (N, 2) array of [z, r] points for shroud.
            le_z: Axial position of leading edge.
            te_z: Axial position of trailing edge.

        Returns:
            List of 3 SCMBlock objects (inlet, blade, outlet).
        """
        cfg = self.config
        chord = abs(te_z - le_z)

        # Determine domain extents
        z_inlet = le_z - cfg.inlet_length_fraction * chord
        z_outlet = te_z + cfg.outlet_length_fraction * chord

        # Interpolate hub and shroud at boundary z-positions
        hub_z = hub_contour[:, 0]
        hub_r = hub_contour[:, 1]
        shroud_z = shroud_contour[:, 0]
        shroud_r = shroud_contour[:, 1]

        def _hub_r(z):
            return float(np.interp(z, hub_z, hub_r))

        def _shroud_r(z):
            return float(np.interp(z, shroud_z, shroud_r))

        # Block 1: Inlet
        block1 = self._generate_block(
            z_start=z_inlet, z_end=le_z,
            hub_r_func=_hub_r, shroud_r_func=_shroud_r,
            n_axial=cfg.n_inlet_axial,
            n_radial=cfg.n_radial,
            grading_axial=cfg.grading_inlet,
            grading_radial=cfg.grading_radial,
        )

        # Block 2: Blade passage
        block2 = self._generate_block(
            z_start=le_z, z_end=te_z,
            hub_r_func=_hub_r, shroud_r_func=_shroud_r,
            n_axial=cfg.n_blade_axial,
            n_radial=cfg.n_radial,
            grading_axial=cfg.grading_blade,
            grading_radial=cfg.grading_radial,
        )

        # Block 3: Outlet
        block3 = self._generate_block(
            z_start=te_z, z_end=z_outlet,
            hub_r_func=_hub_r, shroud_r_func=_shroud_r,
            n_axial=cfg.n_outlet_axial,
            n_radial=cfg.n_radial,
            grading_axial=cfg.grading_outlet,
            grading_radial=cfg.grading_radial,
        )

        self.blocks = [block1, block2, block3]
        return self.blocks

    def _generate_block(
        self,
        z_start: float,
        z_end: float,
        hub_r_func,
        shroud_r_func,
        n_axial: int,
        n_radial: int,
        grading_axial: float,
        grading_radial: float,
    ) -> SCMBlock:
        """Generate a single TFI block between hub and shroud."""
        # Build boundary curves in (z, r) space
        ni = n_axial + 1
        nj = n_radial + 1

        z_line = np.linspace(z_start, z_end, ni)

        # Bottom = hub contour
        bottom = np.column_stack((z_line, [hub_r_func(z) for z in z_line]))
        # Top = shroud contour
        top = np.column_stack((z_line, [shroud_r_func(z) for z in z_line]))

        # Left = inlet quasi-orthogonal (vertical line at z_start)
        r_hub_start = hub_r_func(z_start)
        r_shroud_start = shroud_r_func(z_start)
        left = np.column_stack((
            np.full(nj, z_start),
            np.linspace(r_hub_start, r_shroud_start, nj),
        ))

        # Right = outlet quasi-orthogonal (vertical line at z_end)
        r_hub_end = hub_r_func(z_end)
        r_shroud_end = shroud_r_func(z_end)
        right = np.column_stack((
            np.full(nj, z_end),
            np.linspace(r_hub_end, r_shroud_end, nj),
        ))

        # TFI with grading
        points = tfi_2d_graded(
            bottom, top, left, right,
            grading_s=grading_axial,
            grading_t=grading_radial,
            n_cells_s=n_axial,
            n_cells_t=n_radial,
        )

        block = SCMBlock(
            points=points,
            n_axial=n_axial,
            n_radial=n_radial,
            grading_axial=grading_axial,
            grading_radial=grading_radial,
        )
        return block

    def get_all_points(self) -> NDArray[np.float64]:
        """Return all mesh points concatenated as (N_total, 2) array."""
        if not self.blocks:
            return np.empty((0, 2))
        all_pts = []
        for block in self.blocks:
            if block.points is not None:
                ni, nj, _ = block.points.shape
                all_pts.append(block.points.reshape(-1, 2))
        return np.concatenate(all_pts, axis=0)

    def get_block_coordinates(self, block_idx: int) -> NDArray[np.float64]:
        """Return (Ni, Nj, 2) coordinate array for a specific block."""
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise IndexError(f"Block index {block_idx} out of range")
        return self.blocks[block_idx].points
