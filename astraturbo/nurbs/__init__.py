"""NURBS curve and surface utilities for AstraTurbo.

Built on geomdl (NURBS-Python), replacing V1's PythonNURBS C++ bindings.

Submodules:
  - curves: Curve interpolation, evaluation, length, parameter search
  - surfaces: Surface interpolation, evaluation
  - operations: Coordinate transforms (xyz, rpz), vector math
  - converters: Convert between geomdl objects and numpy arrays
"""

from .curves import (
    interpolate_2d,
    interpolate_3d,
    approximate_3d,
    evaluate_curve,
    evaluate_curve_array,
    curve_length,
    find_u_from_point,
    find_u_from_z,
)
from .surfaces import (
    interpolate_surface,
    approximate_surface,
    evaluate_surface,
    evaluate_surface_grid,
)
from .operations import (
    xyz_to_rpz,
    rpz_to_xyz,
    norm,
    distance,
    normalize,
    angle_between,
)
from .converters import (
    curve_to_points,
    surface_to_points,
    points_to_curve,
    convert_2d_to_3d_curve,
)

__all__ = [
    "interpolate_2d",
    "interpolate_3d",
    "approximate_3d",
    "evaluate_curve",
    "evaluate_curve_array",
    "curve_length",
    "find_u_from_point",
    "find_u_from_z",
    "interpolate_surface",
    "approximate_surface",
    "evaluate_surface",
    "evaluate_surface_grid",
    "xyz_to_rpz",
    "rpz_to_xyz",
    "norm",
    "distance",
    "normalize",
    "angle_between",
    "curve_to_points",
    "surface_to_points",
    "points_to_curve",
    "convert_2d_to_3d_curve",
]
