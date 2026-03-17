"""Blade row annular array generator.

Generates the full annular array of blades by replicating a single blade
passage around the circumference. Used for visualization and full-annulus
CFD simulations.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def generate_blade_array(
    single_blade_3d: list[NDArray[np.float64]],
    number_blades: int,
    axis: NDArray[np.float64] | None = None,
) -> list[list[NDArray[np.float64]]]:
    """Replicate a single blade around the circumference.

    Takes the 3D profiles of one blade and creates copies rotated at
    equal angular spacing to form the complete blade row.

    Args:
        single_blade_3d: List of M arrays, each (N, 3), representing
            the 3D profile sections of one blade.
        number_blades: Total number of blades in the row.
        axis: Rotation axis as (3,) vector. Default [0, 0, 1] (z-axis).

    Returns:
        List of `number_blades` blade copies, each being a list of
        (N, 3) profile arrays identical in structure to the input.
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)

    delta_theta = 2.0 * math.pi / number_blades
    all_blades = []

    for blade_idx in range(number_blades):
        theta = blade_idx * delta_theta
        rotated_profiles = []

        for profile_3d in single_blade_3d:
            rotated = _rotate_points(profile_3d, axis, theta)
            rotated_profiles.append(rotated)

        all_blades.append(rotated_profiles)

    return all_blades


def generate_blade_array_flat(
    single_blade_3d: list[NDArray[np.float64]],
    number_blades: int,
    axis: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Generate blade array and return all points as a single (N, 3) array.

    Convenience function for visualization — concatenates all blade
    profiles into one point cloud.

    Args:
        single_blade_3d: Single blade profiles.
        number_blades: Number of blades.
        axis: Rotation axis.

    Returns:
        (N_total, 3) array of all blade points.
    """
    all_blades = generate_blade_array(single_blade_3d, number_blades, axis)
    all_points = []
    for blade in all_blades:
        for profile in blade:
            all_points.append(profile)
    return np.concatenate(all_points, axis=0)


def generate_passage_array(
    passage_mesh_points: NDArray[np.float64],
    number_blades: int,
    axis: NDArray[np.float64] | None = None,
) -> list[NDArray[np.float64]]:
    """Replicate a single blade passage mesh around the circumference.

    Args:
        passage_mesh_points: (N, 3) points from one passage mesh.
        number_blades: Number of passages (= number of blades).
        axis: Rotation axis.

    Returns:
        List of `number_blades` point arrays, each (N, 3).
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)

    delta_theta = 2.0 * math.pi / number_blades
    passages = []

    for i in range(number_blades):
        theta = i * delta_theta
        rotated = _rotate_points(passage_mesh_points, axis, theta)
        passages.append(rotated)

    return passages


def _rotate_points(
    points: NDArray[np.float64],
    axis: NDArray[np.float64],
    angle: float,
) -> NDArray[np.float64]:
    """Rotate points around an axis by angle (Rodrigues' formula).

    Args:
        points: (N, 3) array of points.
        axis: (3,) unit rotation axis.
        angle: Rotation angle in radians.

    Returns:
        (N, 3) rotated points.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Rodrigues' rotation formula (vectorized)
    k = axis
    dot = points @ k  # (N,) dot products
    cross = np.cross(k, points)  # (N, 3) cross products

    rotated = (
        points * cos_a
        + cross * sin_a
        + np.outer(dot, k) * (1 - cos_a)
    )
    return rotated
