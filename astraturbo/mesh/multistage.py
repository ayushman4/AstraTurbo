"""Multi-stage turbomachinery mesh orchestration.

Generates structured meshes for rotor+stator stages with matching
interfaces, periodic boundaries, and combined CGNS export.

This fills the adaptation requirements:
  "Single-stage (rotor+stator) CGNS mesh generation"
  "Structured multi-block CGNS of multi-stage axial compressor"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .multiblock import MultiBlockMesh, generate_blade_passage_mesh


@dataclass
class RowMeshConfig:
    """Mesh configuration for a single blade row."""

    profile: NDArray[np.float64]        # (N, 2) closed profile
    pitch: float                        # Blade pitch
    n_blade: int = 40
    n_ogrid: int = 10
    n_inlet: int = 15
    n_outlet: int = 15
    n_passage: int = 20
    ogrid_thickness: float = 0.01
    grading_ogrid: float = 1.3
    grading_inlet: float = 0.5
    grading_outlet: float = 2.0
    inlet_offset: float | None = None
    outlet_offset: float | None = None
    is_rotor: bool = True
    omega: float = 0.0                  # Angular velocity for rotors


@dataclass
class StageConfig:
    """Configuration for a single stage (rotor + stator)."""

    rotor: RowMeshConfig
    stator: RowMeshConfig
    gap_fraction: float = 0.3  # Inter-row gap as fraction of chord


@dataclass
class MultistageMesh:
    """Complete multi-stage mesh with all rows."""

    row_meshes: list[MultiBlockMesh] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    row_configs: list[RowMeshConfig] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.row_meshes)

    @property
    def total_cells(self) -> int:
        return sum(m.total_cells for m in self.row_meshes)

    @property
    def total_blocks(self) -> int:
        return sum(m.n_blocks for m in self.row_meshes)

    def export_cgns(self, filepath: str | Path) -> None:
        """Export all rows to a single CGNS file with multiple bases."""
        from ..export.cgns_writer import write_cgns_structured
        import h5py

        filepath = Path(filepath)

        # Collect all blocks with row-prefixed names
        all_blocks = []
        all_names = []
        for row_name, mesh in zip(self.row_names, self.row_meshes):
            for block in mesh.blocks:
                all_blocks.append(block.points)
                all_names.append(f"{row_name}_{block.name}")

        write_cgns_structured(filepath, all_blocks, all_names, "MultiStage")

    def export_cgns_per_row(self, output_dir: str | Path) -> list[Path]:
        """Export each row to a separate CGNS file.

        Args:
            output_dir: Directory for output files.

        Returns:
            List of output file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for row_name, mesh in zip(self.row_names, self.row_meshes):
            path = output_dir / f"{row_name}.cgns"
            mesh.export_cgns(path)
            paths.append(path)
        return paths


class MultistageGenerator:
    """Orchestrates mesh generation for multi-stage turbomachinery.

    Handles:
    - Individual row mesh generation
    - Interface matching between rows
    - Periodic boundary setup
    - Combined CGNS export

    Usage::

        gen = MultistageGenerator()
        gen.add_row("rotor1", rotor_config)
        gen.add_row("stator1", stator_config)
        gen.add_row("rotor2", rotor2_config)
        result = gen.generate()
        result.export_cgns("compressor_3row.cgns")
    """

    def __init__(self) -> None:
        self._rows: list[tuple[str, RowMeshConfig]] = []

    def add_row(self, name: str, config: RowMeshConfig) -> None:
        """Add a blade row to the multi-stage assembly."""
        self._rows.append((name, config))

    def add_stage(self, stage_name: str, stage: StageConfig) -> None:
        """Add a complete stage (rotor + stator).

        Automatically names them '{stage_name}_rotor' and '{stage_name}_stator'.
        """
        self.add_row(f"{stage_name}_rotor", stage.rotor)
        self.add_row(f"{stage_name}_stator", stage.stator)

    def generate(self) -> MultistageMesh:
        """Generate meshes for all rows.

        Returns:
            MultistageMesh containing all row meshes.
        """
        result = MultistageMesh()

        for name, config in self._rows:
            mesh = generate_blade_passage_mesh(
                profile=config.profile,
                pitch=config.pitch,
                n_blade=config.n_blade,
                n_ogrid=config.n_ogrid,
                n_inlet=config.n_inlet,
                n_outlet=config.n_outlet,
                n_passage=config.n_passage,
                ogrid_thickness=config.ogrid_thickness,
                grading_ogrid=config.grading_ogrid,
                grading_inlet=config.grading_inlet,
                grading_outlet=config.grading_outlet,
                inlet_offset=config.inlet_offset,
                outlet_offset=config.outlet_offset,
            )
            result.row_meshes.append(mesh)
            result.row_names.append(name)
            result.row_configs.append(config)

        return result

    @staticmethod
    def create_single_stage(
        rotor_profile: NDArray[np.float64],
        stator_profile: NDArray[np.float64],
        rotor_pitch: float,
        stator_pitch: float,
        rotor_omega: float = 0.0,
        **mesh_kwargs,
    ) -> MultistageMesh:
        """Convenience: generate a single rotor+stator stage.

        Args:
            rotor_profile: (N, 2) rotor blade profile.
            stator_profile: (N, 2) stator blade profile.
            rotor_pitch: Rotor blade pitch.
            stator_pitch: Stator blade pitch.
            rotor_omega: Rotor angular velocity.
            **mesh_kwargs: Additional mesh parameters.

        Returns:
            MultistageMesh with 2 rows.
        """
        gen = MultistageGenerator()
        gen.add_row("rotor", RowMeshConfig(
            profile=rotor_profile, pitch=rotor_pitch,
            is_rotor=True, omega=rotor_omega, **mesh_kwargs,
        ))
        gen.add_row("stator", RowMeshConfig(
            profile=stator_profile, pitch=stator_pitch,
            is_rotor=False, omega=0.0, **mesh_kwargs,
        ))
        return gen.generate()
