"""Blade row: the main 3D blade geometry container.

Ported from V1 bladeRow.py. A BladeRow holds multiple 2D profiles,
hub/shroud contours, and computes the 3D blade surface by stacking
and lofting.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..baseclass import Node
from ..foundation import BoundedNumericProperty, NumericProperty, StringProperty
from ..profile import Superposition
from .blade_surface import loft_blade_surface, compute_leading_trailing_edges
from .hub_shroud import MeridionalContour, compute_stacking_line
from .stacking import axial_stacking, radial_stacking, cascade_stacking


class BladeRow(Node):
    """A single blade row in a turbomachine.

    Contains multiple 2D profiles at different span positions,
    hub/shroud contours, and parameters for 3D blade construction.

    Properties:
        number_blades: Number of blades in the row.
        omega: Angular velocity in rad/s (0 for stators).
        stacking_mode: 0=axial, 1=radial, 2=cascade.
    """

    number_blades = BoundedNumericProperty(lb=1, ub=200, default=20)
    omega = NumericProperty(default=0.0)

    def __init__(
        self,
        profiles: list[Superposition] | None = None,
        hub_points: NDArray[np.float64] | None = None,
        shroud_points: NDArray[np.float64] | None = None,
        stacking_mode: int = 0,
    ) -> None:
        super().__init__()
        self.name = "Blade Row"
        self._profiles = profiles or []
        self._stacking_mode = stacking_mode

        # Hub and shroud contours in (z, r) plane
        if hub_points is not None:
            self._hub = MeridionalContour(hub_points)
        else:
            self._hub = MeridionalContour(np.array([[0.0, 0.1], [0.1, 0.1]]))

        if shroud_points is not None:
            self._shroud = MeridionalContour(shroud_points)
        else:
            self._shroud = MeridionalContour(np.array([[0.0, 0.2], [0.1, 0.2]]))

        # Computed results (lazy)
        self._blade_surface = None
        self._profiles_3d: list[NDArray[np.float64]] | None = None

    @property
    def profiles(self) -> list[Superposition]:
        return self._profiles

    @property
    def hub(self) -> MeridionalContour:
        return self._hub

    @property
    def shroud(self) -> MeridionalContour:
        return self._shroud

    @property
    def stacking_mode(self) -> int:
        return self._stacking_mode

    @stacking_mode.setter
    def stacking_mode(self, value: int) -> None:
        if value not in (0, 1, 2):
            raise ValueError("stacking_mode must be 0 (axial), 1 (radial), or 2 (cascade)")
        self._stacking_mode = value
        self._blade_surface = None
        self._profiles_3d = None

    def add_profile(self, profile: Superposition) -> None:
        """Add a 2D profile at a new span position."""
        self._profiles.append(profile)
        self._blade_surface = None
        self._profiles_3d = None

    def compute(
        self,
        stagger_angles: NDArray[np.float64] | None = None,
        chord_lengths: NDArray[np.float64] | None = None,
    ) -> None:
        """Compute 3D blade geometry from 2D profiles.

        Args:
            stagger_angles: (M,) stagger angles in radians per profile.
                Defaults to zeros.
            chord_lengths: (M,) chord lengths per profile.
                Defaults to ones.
        """
        if not self._profiles:
            raise ValueError("No profiles defined. Add profiles before computing.")

        n = len(self._profiles)
        if n < 2:
            raise ValueError(
                "At least 2 profiles are required for 3D blade computation. "
                "Use Edit > Add Profile to Row to add more span profiles."
            )

        if stagger_angles is None:
            stagger_angles = np.zeros(n)
        if chord_lengths is None:
            chord_lengths = np.ones(n)

        # Get 2D profile arrays
        profiles_2d = [p.as_array() for p in self._profiles]

        # Compute span positions
        radii = compute_stacking_line(self._hub, self._shroud, n)

        # Stack profiles into 3D
        if self._stacking_mode == 0:
            self._profiles_3d = axial_stacking(
                profiles_2d, radii, stagger_angles, chord_lengths
            )
        elif self._stacking_mode == 1:
            self._profiles_3d = radial_stacking(
                profiles_2d, radii, stagger_angles, chord_lengths
            )
        elif self._stacking_mode == 2:
            pitches = 2 * np.pi * radii / self.number_blades
            self._profiles_3d = cascade_stacking(
                profiles_2d, pitches, stagger_angles, chord_lengths
            )

        # Loft surface through 3D profiles (needs at least 2 distinct profiles)
        try:
            self._blade_surface = loft_blade_surface(self._profiles_3d)
        except Exception as e:
            # Surface lofting can fail if profiles are too similar
            logging.getLogger(__name__).warning(
                "Blade surface lofting failed: %s. Keeping 3D profiles for other uses.", e
            )
            self._blade_surface = None

        # Validate blade surface if lofting succeeded
        if self._blade_surface is not None:
            self._validate_blade_surface()

    def _validate_blade_surface(self) -> None:
        """Validate blade surface for self-intersection and minimum thickness."""
        logger = logging.getLogger(__name__)

        if self._profiles_3d is None:
            return

        min_thickness_mm = 0.3  # Minimum acceptable thickness in mm

        for idx, profile in enumerate(self._profiles_3d):
            if profile.shape[1] < 2:
                continue

            n_pts = len(profile)
            if n_pts < 4:
                continue

            # Check minimum thickness: measure distance between points at
            # equivalent stations on upper and lower surfaces
            mid = n_pts // 2
            # Upper surface: first half, lower surface: second half
            for i in range(min(mid, n_pts - mid)):
                if i < mid and (mid + i) < n_pts:
                    dist = np.linalg.norm(profile[i] - profile[-(i + 1)])
                    if dist < min_thickness_mm * 1e-3 and dist > 1e-12:
                        logger.warning(
                            "Span station %d: thickness %.4f mm at point %d "
                            "is below minimum %.1f mm",
                            idx, dist * 1e3, i, min_thickness_mm,
                        )
                        break

            # Check normals consistency for self-intersection detection
            if profile.shape[1] >= 2:
                edges = np.diff(profile[:, :2], axis=0)
                normals = np.column_stack([-edges[:, 1], edges[:, 0]])
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms = np.where(norms < 1e-15, 1.0, norms)
                normals = normals / norms

                # Check for sign flips in normal direction (indicates self-intersection)
                if len(normals) > 2:
                    dots = np.sum(normals[:-1] * normals[1:], axis=1)
                    sign_flips = np.sum(dots < -0.5)
                    if sign_flips > 2:  # Allow a couple at LE/TE
                        logger.warning(
                            "Span station %d: detected %d normal direction reversals, "
                            "possible self-intersection.",
                            idx, sign_flips,
                        )

    @property
    def blade_surface(self):
        """Return the computed NURBS blade surface (None if not computed)."""
        return self._blade_surface

    @property
    def profiles_3d(self) -> list[NDArray[np.float64]] | None:
        """Return the 3D stacked profiles (None if not computed)."""
        return self._profiles_3d

    @property
    def leading_edge(self) -> NDArray[np.float64] | None:
        """Return leading edge curve as (M, 3) array."""
        if self._profiles_3d is None:
            return None
        le, _ = compute_leading_trailing_edges(self._profiles_3d)
        return le

    @property
    def trailing_edge(self) -> NDArray[np.float64] | None:
        """Return trailing edge curve as (M, 3) array."""
        if self._profiles_3d is None:
            return None
        _, te = compute_leading_trailing_edges(self._profiles_3d)
        return te
