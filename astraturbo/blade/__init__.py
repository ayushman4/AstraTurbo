"""3D blade geometry module for AstraTurbo.

Provides blade row construction from 2D profiles through 3D stacking,
NURBS surface lofting, annular array generation, and supporting geometry.
"""

from .blade_row import BladeRow
from .blade_surface import loft_blade_surface, compute_leading_trailing_edges
from .camber_surface import extract_camber_surface, compute_blade_angles
from .hub_shroud import MeridionalContour, compute_stacking_line
from .section import get_blade_section
from .stacking import axial_stacking, radial_stacking, cascade_stacking
from .annular_array import generate_blade_array, generate_blade_array_flat, generate_passage_array

__all__ = [
    "BladeRow",
    "MeridionalContour",
    "loft_blade_surface",
    "compute_leading_trailing_edges",
    "extract_camber_surface",
    "compute_blade_angles",
    "compute_stacking_line",
    "get_blade_section",
    "axial_stacking",
    "radial_stacking",
    "cascade_stacking",
    "generate_blade_array",
    "generate_blade_array_flat",
    "generate_passage_array",
]
