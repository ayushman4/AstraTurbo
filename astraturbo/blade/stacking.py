"""3D blade stacking algorithms.

Ported from V1 bladeRow.py stacking modes:
  - Mode 0 (axial): Length-preserving transformation for axial machines
  - Mode 1 (radial): Angle-preserving transformation for radial machines
  - Mode 2 (cascade): 2D periodic cascade

Stacking transforms 2D blade profiles at different span positions into
3D coordinates on the blade surface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def axial_stacking(
    profiles_2d: list[NDArray[np.float64]],
    radii: NDArray[np.float64],
    stagger_angles: NDArray[np.float64],
    chord_lengths: NDArray[np.float64],
    stacking_offsets: NDArray[np.float64] | None = None,
) -> list[NDArray[np.float64]]:
    """Length-preserving stacking for axial turbomachinery (Mode 0).

    Transforms 2D profiles into 3D cylindrical coordinates at each
    span position, preserving arc lengths along the blade surface.

    Args:
        profiles_2d: List of (N, 2) arrays, one per span position.
        radii: (M,) array of radial positions for each profile.
        stagger_angles: (M,) array of stagger angles in radians.
        chord_lengths: (M,) array of chord lengths at each span.
        stacking_offsets: (M,) optional circumferential offsets.

    Returns:
        List of (N, 3) arrays in Cartesian (x, y, z) coordinates.
    """
    n_profiles = len(profiles_2d)
    if stacking_offsets is None:
        stacking_offsets = np.zeros(n_profiles)

    profiles_3d = []
    for i in range(n_profiles):
        profile = profiles_2d[i]
        r = radii[i]
        gamma = stagger_angles[i]
        chord = chord_lengths[i]
        offset = stacking_offsets[i]

        # Scale profile by chord length
        x_scaled = profile[:, 0] * chord
        y_scaled = profile[:, 1] * chord

        # Rotate by stagger angle
        x_rot = x_scaled * np.cos(gamma) - y_scaled * np.sin(gamma)
        y_rot = x_scaled * np.sin(gamma) + y_scaled * np.cos(gamma)

        # Map to cylindrical: axial = z, circumferential = r*phi
        z = x_rot
        phi = y_rot / r + offset

        # Convert to Cartesian
        x_3d = r * np.cos(phi)
        y_3d = r * np.sin(phi)
        z_3d = z

        profiles_3d.append(np.column_stack((x_3d, y_3d, z_3d)))

    return profiles_3d


def radial_stacking(
    profiles_2d: list[NDArray[np.float64]],
    radii: NDArray[np.float64],
    stagger_angles: NDArray[np.float64],
    chord_lengths: NDArray[np.float64],
    stacking_offsets: NDArray[np.float64] | None = None,
    r_le: NDArray[np.float64] | None = None,
    r_te: NDArray[np.float64] | None = None,
) -> list[NDArray[np.float64]]:
    """Angle-preserving (conformal) stacking for radial turbomachinery (Mode 1).

    Preserves blade angles by computing the circumferential coordinate
    using the local radius at each chordwise position, rather than a
    single constant radius. This is essential for centrifugal compressors
    and radial turbines where radius varies significantly along the chord.

    The conformal mapping ensures that:
        dphi = dm / r(m)
    so that a square element in the (m, r*phi) plane maps to a square
    element on the actual cylindrical surface at every point.

    Reference: Barthmes, "Entwicklung eines parametrischen CAD-Systems
    für Turbomaschinen", TU München, 2010.

    Args:
        profiles_2d: List of (N, 2) arrays, one per span position.
        radii: (M,) array of mean radial positions for each profile.
        stagger_angles: (M,) array of stagger angles in radians.
        chord_lengths: (M,) array of chord lengths at each span.
        stacking_offsets: (M,) optional circumferential offsets.
        r_le: (M,) leading-edge radii. If None, uses radii (constant r).
        r_te: (M,) trailing-edge radii. If None, uses radii (constant r).

    Returns:
        List of (N, 3) arrays in Cartesian (x, y, z) coordinates.
    """
    n_profiles = len(profiles_2d)
    if stacking_offsets is None:
        stacking_offsets = np.zeros(n_profiles)

    profiles_3d = []
    for i in range(n_profiles):
        profile = profiles_2d[i]
        r_mean = radii[i]
        gamma = stagger_angles[i]
        chord = chord_lengths[i]
        offset = stacking_offsets[i]

        # Scale profile by chord length
        x_scaled = profile[:, 0] * chord
        y_scaled = profile[:, 1] * chord

        # Rotate by stagger angle
        x_rot = x_scaled * np.cos(gamma) - y_scaled * np.sin(gamma)
        y_rot = x_scaled * np.sin(gamma) + y_scaled * np.cos(gamma)

        # Conformal mapping: local radius varies along the meridional
        # coordinate. Interpolate r between LE and TE radii.
        if r_le is not None and r_te is not None:
            r_leading = r_le[i]
            r_trailing = r_te[i]
        else:
            # Fallback: constant radius (degenerates to cylindrical)
            r_leading = r_mean
            r_trailing = r_mean

        # Normalize meridional coordinate to [0, 1] for interpolation
        x_min = np.min(x_rot)
        x_range = np.max(x_rot) - x_min
        if x_range > 1e-12:
            t = (x_rot - x_min) / x_range
        else:
            t = np.zeros_like(x_rot)

        # Local radius at each chordwise position
        r_local = r_leading + t * (r_trailing - r_leading)

        # Map to cylindrical: z = meridional, phi = y / r_local
        z = x_rot
        phi = y_rot / r_local + offset

        # Convert to Cartesian
        x_3d = r_local * np.cos(phi)
        y_3d = r_local * np.sin(phi)
        z_3d = z

        profiles_3d.append(np.column_stack((x_3d, y_3d, z_3d)))

    return profiles_3d


def cascade_stacking(
    profiles_2d: list[NDArray[np.float64]],
    pitches: NDArray[np.float64],
    stagger_angles: NDArray[np.float64],
    chord_lengths: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Cascade stacking for 2D periodic simulations (Mode 2).

    Produces flat (non-curved) blade passage geometry suitable for
    cascade CFD simulations.

    Args:
        profiles_2d: List of (N, 2) arrays.
        pitches: (M,) array of blade pitches at each span.
        stagger_angles: (M,) stagger angles in radians.
        chord_lengths: (M,) chord lengths.

    Returns:
        List of (N, 3) arrays. Z coordinate is the span position index.
    """
    profiles_3d = []
    for i, profile in enumerate(profiles_2d):
        chord = chord_lengths[i]
        gamma = stagger_angles[i]

        # Scale
        x_scaled = profile[:, 0] * chord
        y_scaled = profile[:, 1] * chord

        # Rotate by stagger
        x_rot = x_scaled * np.cos(gamma) - y_scaled * np.sin(gamma)
        y_rot = x_scaled * np.sin(gamma) + y_scaled * np.cos(gamma)

        # Flat: z = span index (normalized)
        z = np.full_like(x_rot, float(i) / max(len(profiles_2d) - 1, 1))

        profiles_3d.append(np.column_stack((x_rot, y_rot, z)))

    return profiles_3d
