"""O-grid mesh generation orchestrator.

Coordinates the full O-grid mesh generation pipeline:
topology → vertex computation → block filling → patch assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .topology import O10HTopology, create_o10h_topology
from ..transfinite import tfi_2d_vectorized, apply_grading


@dataclass
class OGridMeshConfig:
    """Configuration for O-grid mesh generation."""

    # O-grid parameters
    n_ogrid_normal: int = 10       # Wall-normal cells in O-grid
    ogrid_thickness: float = 0.01  # O-grid thickness (physical units)
    ogrid_grading: float = 1.3     # Wall-normal grading (>1 = finer at wall)

    # Blade discretization
    n_blade_wrap: int = 40         # Cells wrapping around blade

    # Domain
    n_inlet: int = 15
    n_outlet: int = 15
    n_passage: int = 20
    n_span: int = 1

    # Grading
    grading_inlet: float = 0.5
    grading_outlet: float = 2.0

    # Domain extent fractions
    inlet_fraction: float = 0.5
    outlet_fraction: float = 0.5


@dataclass
class OGridMesh:
    """Result of O-grid mesh generation.

    Contains all block coordinates and topology for export.
    """

    topology: O10HTopology
    config: OGridMeshConfig
    block_points: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.block_points)

    def get_all_points(self) -> NDArray[np.float64]:
        """Return all unique mesh points as (N, 3) array."""
        if not self.block_points:
            return np.empty((0, 3))
        all_pts = []
        for pts in self.block_points.values():
            all_pts.append(pts.reshape(-1, pts.shape[-1]))
        return np.concatenate(all_pts, axis=0)


class OGridGenerator:
    """Generates structured O-grid mesh around a turbomachinery blade.

    This is the core mesh generation engine. It takes a blade profile
    and domain parameters and produces a multi-block structured mesh.
    """

    def __init__(self, config: OGridMeshConfig | None = None) -> None:
        self.config = config or OGridMeshConfig()

    def generate(
        self,
        blade_profile: NDArray[np.float64],
        pitch: float,
        inlet_offset: float | None = None,
        outlet_offset: float | None = None,
    ) -> OGridMesh:
        """Generate O-grid mesh for a blade passage.

        Args:
            blade_profile: (N, 2) closed blade profile points.
            pitch: Blade pitch (passage width).
            inlet_offset: Distance from LE to inlet (None = auto).
            outlet_offset: Distance from TE to outlet (None = auto).

        Returns:
            OGridMesh with block coordinates.
        """
        cfg = self.config

        # Determine blade bounds
        x_min = blade_profile[:, 0].min()
        x_max = blade_profile[:, 0].max()
        chord = x_max - x_min

        if inlet_offset is None:
            inlet_offset = chord * cfg.inlet_fraction
        if outlet_offset is None:
            outlet_offset = chord * cfg.outlet_fraction

        # Split blade into suction and pressure sides
        n_pts = len(blade_profile)
        n_half = (n_pts + 1) // 2
        suction_side = blade_profile[:n_half][::-1]  # TE -> LE on upper
        pressure_side = blade_profile[n_half - 1:]    # LE -> TE on lower

        # Generate scaled (offset) profile for O-grid outer boundary
        scaled_profile = self._offset_profile(
            blade_profile, cfg.ogrid_thickness
        )

        # Create topology
        topology = create_o10h_topology(
            n_ogrid_normal=cfg.n_ogrid_normal,
            n_blade_wrap=cfg.n_blade_wrap,
            n_inlet=cfg.n_inlet,
            n_outlet=cfg.n_outlet,
            n_passage=cfg.n_passage,
            n_span=cfg.n_span,
        )

        # Generate block coordinates
        block_points = {}

        # O-grid blocks: use TFI between blade surface and offset curve
        ogrid_pts = self._generate_ogrid_blocks(
            suction_side, pressure_side, cfg
        )
        block_points.update(ogrid_pts)

        # H-grid blocks: use TFI for inlet/outlet/passage
        hgrid_pts = self._generate_hgrid_blocks(
            blade_profile, scaled_profile, pitch,
            x_min - inlet_offset, x_max + outlet_offset, cfg
        )
        block_points.update(hgrid_pts)

        return OGridMesh(
            topology=topology,
            config=cfg,
            block_points=block_points,
        )

    def _offset_profile(
        self,
        profile: NDArray[np.float64],
        thickness: float,
    ) -> NDArray[np.float64]:
        """Offset a closed profile outward by a given thickness.

        Uses normal vectors at each point to push points outward.
        """
        n = len(profile)
        normals = np.zeros_like(profile)

        for i in range(n):
            # Tangent from neighbors
            i_prev = (i - 1) % n
            i_next = (i + 1) % n
            tangent = profile[i_next] - profile[i_prev]
            # Outward normal (rotate tangent 90 degrees)
            normal = np.array([-tangent[1], tangent[0]])
            length = np.sqrt(normal[0] ** 2 + normal[1] ** 2)
            if length > 1e-15:
                normal /= length
            normals[i] = normal

        return profile + normals * thickness

    def _generate_ogrid_blocks(
        self,
        suction: NDArray[np.float64],
        pressure: NDArray[np.float64],
        cfg: OGridMeshConfig,
    ) -> dict[str, NDArray[np.float64]]:
        """Generate O-grid block points around the blade."""
        blocks = {}

        # Resample blade sides to uniform point count
        n_wrap_half = cfg.n_blade_wrap // 2 + 1
        n_normal = cfg.n_ogrid_normal + 1

        for side_name, side_pts in [("o_suction", suction), ("o_pressure", pressure)]:
            # Resample to n_wrap_half points
            t_orig = np.linspace(0, 1, len(side_pts))
            t_new = np.linspace(0, 1, n_wrap_half)
            blade_edge = np.column_stack((
                np.interp(t_new, t_orig, side_pts[:, 0]),
                np.interp(t_new, t_orig, side_pts[:, 1]),
            ))

            # Offset edge
            outer_edge = self._offset_profile(blade_edge, cfg.ogrid_thickness)

            # Left/right boundaries (radial lines at LE and TE)
            left = np.column_stack((
                np.linspace(blade_edge[0, 0], outer_edge[0, 0], n_normal),
                np.linspace(blade_edge[0, 1], outer_edge[0, 1], n_normal),
            ))
            right = np.column_stack((
                np.linspace(blade_edge[-1, 0], outer_edge[-1, 0], n_normal),
                np.linspace(blade_edge[-1, 1], outer_edge[-1, 1], n_normal),
            ))

            block = tfi_2d_vectorized(blade_edge, outer_edge, left, right)
            blocks[side_name] = block

        return blocks

    def _generate_hgrid_blocks(
        self,
        blade_profile: NDArray[np.float64],
        scaled_profile: NDArray[np.float64],
        pitch: float,
        x_inlet: float,
        x_outlet: float,
        cfg: OGridMeshConfig,
    ) -> dict[str, NDArray[np.float64]]:
        """Generate H-grid blocks for inlet, outlet, and passage."""
        blocks = {}

        # Blade extent
        x_min = blade_profile[:, 0].min()
        x_max = blade_profile[:, 0].max()
        y_mid = blade_profile[:, 1].mean()

        # Simple inlet block (rectangular H-grid upstream of blade)
        ni = cfg.n_inlet + 1
        nj = cfg.n_passage + 1

        for suffix, y_range in [
            ("h_inlet_upper", (y_mid, y_mid + pitch / 2)),
            ("h_inlet_lower", (y_mid - pitch / 2, y_mid)),
        ]:
            x_pts = np.linspace(x_inlet, x_min, ni)
            y_pts = np.linspace(y_range[0], y_range[1], nj)
            xx, yy = np.meshgrid(x_pts, y_pts, indexing="ij")
            blocks[suffix] = np.stack((xx, yy), axis=-1)

        # Outlet blocks
        ni = cfg.n_outlet + 1
        for suffix, y_range in [
            ("h_outlet_upper", (y_mid, y_mid + pitch / 2)),
            ("h_outlet_lower", (y_mid - pitch / 2, y_mid)),
        ]:
            x_pts = np.linspace(x_max, x_outlet, ni)
            y_pts = np.linspace(y_range[0], y_range[1], nj)
            xx, yy = np.meshgrid(x_pts, y_pts, indexing="ij")
            blocks[suffix] = np.stack((xx, yy), axis=-1)

        return blocks
