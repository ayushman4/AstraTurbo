"""Polyline and Arc geometry structures for structured mesh generation.

These are the fundamental curve types used to define block edges in
multi-block structured meshes. A Polyline is an ordered sequence of
points between two block vertices. An Arc is a circular arc edge.

This fills the key requirement of extracting polyline data and
coordinates between block vertices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class Polyline:
    """Ordered sequence of 2D or 3D points defining a block edge.

    A polyline connects two block vertices through intermediate points.
    It supports resampling, length computation, and graded point distribution.

    Attributes:
        points: (N, D) array of ordered coordinates (D = 2 or 3).
        start_vertex: Index of the starting block vertex.
        end_vertex: Index of the ending block vertex.
    """

    def __init__(
        self,
        points: NDArray[np.float64],
        start_vertex: int = -1,
        end_vertex: int = -1,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def start(self) -> NDArray[np.float64]:
        return self.points[0]

    @property
    def end(self) -> NDArray[np.float64]:
        return self.points[-1]

    def segment_lengths(self) -> NDArray[np.float64]:
        """Return the length of each segment between consecutive points."""
        diffs = np.diff(self.points, axis=0)
        return np.sqrt(np.sum(diffs**2, axis=1))

    def cumulative_lengths(self) -> NDArray[np.float64]:
        """Return cumulative arc length at each point, starting from 0."""
        seg = self.segment_lengths()
        cum = np.zeros(self.n_points)
        cum[1:] = np.cumsum(seg)
        return cum

    def total_length(self) -> float:
        """Return total arc length of the polyline."""
        return float(np.sum(self.segment_lengths()))

    def normalized_parameters(self) -> NDArray[np.float64]:
        """Return normalized [0, 1] arc-length parameter at each point."""
        cum = self.cumulative_lengths()
        total = cum[-1]
        if total < 1e-15:
            return np.linspace(0, 1, self.n_points)
        return cum / total

    def evaluate_at(self, t: float) -> NDArray[np.float64]:
        """Evaluate the polyline at normalized parameter t in [0, 1].

        Uses linear interpolation between stored points.
        """
        params = self.normalized_parameters()
        result = np.empty(self.dim, dtype=np.float64)
        for d in range(self.dim):
            result[d] = np.interp(t, params, self.points[:, d])
        return result

    def resample(self, n_points: int) -> Polyline:
        """Return a new polyline resampled to n_points uniformly in arc length."""
        t_new = np.linspace(0, 1, n_points)
        params = self.normalized_parameters()
        new_pts = np.empty((n_points, self.dim), dtype=np.float64)
        for d in range(self.dim):
            new_pts[:, d] = np.interp(t_new, params, self.points[:, d])
        return Polyline(new_pts, self.start_vertex, self.end_vertex)

    def resample_graded(
        self, n_points: int, grading_ratio: float = 1.0
    ) -> Polyline:
        """Return a new polyline resampled with graded spacing.

        Args:
            n_points: Number of output points.
            grading_ratio: Ratio of last segment to first segment.
                1.0 = uniform, >1.0 = cells grow, <1.0 = cells shrink.
        """
        from .grading import compute_graded_parameters

        t_graded = compute_graded_parameters(n_points, grading_ratio)
        params = self.normalized_parameters()
        new_pts = np.empty((n_points, self.dim), dtype=np.float64)
        for d in range(self.dim):
            new_pts[:, d] = np.interp(t_graded, params, self.points[:, d])
        return Polyline(new_pts, self.start_vertex, self.end_vertex)

    def reverse(self) -> Polyline:
        """Return a new polyline with reversed point order."""
        return Polyline(
            self.points[::-1].copy(), self.end_vertex, self.start_vertex
        )

    def split_at(self, t: float) -> tuple[Polyline, Polyline]:
        """Split the polyline at parameter t into two polylines."""
        params = self.normalized_parameters()
        mid_pt = self.evaluate_at(t)

        mask_before = params <= t
        mask_after = params >= t

        pts_before = np.vstack([self.points[mask_before], mid_pt[None, :]])
        pts_after = np.vstack([mid_pt[None, :], self.points[mask_after]])

        return (
            Polyline(pts_before, self.start_vertex, -1),
            Polyline(pts_after, -1, self.end_vertex),
        )

    def __repr__(self) -> str:
        return (
            f"Polyline({self.n_points} pts, "
            f"length={self.total_length():.6f}, "
            f"v{self.start_vertex}→v{self.end_vertex})"
        )


class Arc:
    """Circular arc between two endpoints, defined by a midpoint.

    Used for curved block edges in turbomachinery meshes (e.g., blade
    surfaces, hub/shroud contours).

    The arc is defined by:
      - start: Starting point
      - end: Ending point
      - midpoint: A point on the arc between start and end

    From these three points, the center and radius are computed.
    """

    def __init__(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        midpoint: NDArray[np.float64],
        start_vertex: int = -1,
        end_vertex: int = -1,
    ) -> None:
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.midpoint = np.asarray(midpoint, dtype=np.float64)
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex

        self._center, self._radius = self._compute_center_radius()

    @property
    def center(self) -> NDArray[np.float64]:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def dim(self) -> int:
        return len(self.start)

    def _compute_center_radius(self) -> tuple[NDArray[np.float64], float]:
        """Compute circle center and radius from 3 points (2D).

        For 3D arcs, projects to the plane of the 3 points first.
        """
        p1, p2, p3 = self.start, self.midpoint, self.end

        if self.dim == 2:
            return self._circle_from_3_points_2d(p1, p2, p3)
        else:
            # 3D: work in the plane of the three points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            n_len = np.linalg.norm(normal)
            if n_len < 1e-15:
                # Degenerate (collinear) — return midpoint as center
                center = (p1 + p3) / 2.0
                radius = np.linalg.norm(p3 - p1) / 2.0
                return center, radius

            # Use circumscribed circle formula
            d = 2.0 * np.linalg.norm(np.cross(p1 - p2, p2 - p3)) ** 2
            if d < 1e-15:
                center = (p1 + p3) / 2.0
                radius = np.linalg.norm(p3 - p1) / 2.0
                return center, radius

            a = np.linalg.norm(p3 - p2) ** 2 * np.dot(p1 - p2, p1 - p3) / d
            b = np.linalg.norm(p1 - p3) ** 2 * np.dot(p2 - p1, p2 - p3) / d
            c = np.linalg.norm(p1 - p2) ** 2 * np.dot(p3 - p1, p3 - p2) / d

            center = a * p1 + b * p2 + c * p3
            radius = float(np.linalg.norm(p1 - center))
            return center, radius

    @staticmethod
    def _circle_from_3_points_2d(
        p1: NDArray, p2: NDArray, p3: NDArray
    ) -> tuple[NDArray[np.float64], float]:
        """Compute circle center from 3 points in 2D."""
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx, cy = p3[0], p3[1]

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-15:
            center = (p1 + p3) / 2.0
            return center, float(np.linalg.norm(p3 - p1) / 2.0)

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
              (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
              (cx**2 + cy**2) * (bx - ax)) / d

        center = np.array([ux, uy], dtype=np.float64)
        radius = float(np.linalg.norm(p1 - center))
        return center, radius

    def to_polyline(self, n_points: int = 50) -> Polyline:
        """Convert the arc to a polyline with n_points.

        Distributes points uniformly along the arc angle.
        """
        if self.dim == 2:
            return self._to_polyline_2d(n_points)
        else:
            return self._to_polyline_3d(n_points)

    def _to_polyline_2d(self, n_points: int) -> Polyline:
        c = self.center
        theta_start = math.atan2(self.start[1] - c[1], self.start[0] - c[0])
        theta_end = math.atan2(self.end[1] - c[1], self.end[0] - c[0])
        theta_mid = math.atan2(self.midpoint[1] - c[1], self.midpoint[0] - c[0])

        # Determine arc direction
        def _normalize_angle(a, ref):
            while a - ref > math.pi:
                a -= 2 * math.pi
            while a - ref < -math.pi:
                a += 2 * math.pi
            return a

        theta_end = _normalize_angle(theta_end, theta_start)
        theta_mid = _normalize_angle(theta_mid, theta_start)

        # Check if midpoint is in the arc from start to end
        if theta_start < theta_end:
            if not (theta_start <= theta_mid <= theta_end):
                theta_end -= 2 * math.pi
        else:
            if not (theta_end <= theta_mid <= theta_start):
                theta_end += 2 * math.pi

        thetas = np.linspace(theta_start, theta_end, n_points)
        pts = np.column_stack((
            c[0] + self.radius * np.cos(thetas),
            c[1] + self.radius * np.sin(thetas),
        ))
        return Polyline(pts, self.start_vertex, self.end_vertex)

    def _to_polyline_3d(self, n_points: int) -> Polyline:
        """Convert 3D arc to polyline using rotation in the arc plane."""
        c = self.center
        v_start = self.start - c
        v_end = self.end - c

        # Arc plane normal
        normal = np.cross(v_start, v_end)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-15:
            # Degenerate — straight line
            t = np.linspace(0, 1, n_points)
            pts = self.start[None, :] + t[:, None] * (self.end - self.start)[None, :]
            return Polyline(pts, self.start_vertex, self.end_vertex)

        normal = normal / n_len

        # Angle between start and end vectors
        cos_angle = np.clip(
            np.dot(v_start, v_end) / (np.linalg.norm(v_start) * np.linalg.norm(v_end)),
            -1, 1,
        )
        total_angle = math.acos(cos_angle)

        # Check direction via midpoint
        v_mid = self.midpoint - c
        cross_check = np.dot(np.cross(v_start, v_mid), normal)
        if cross_check < 0:
            total_angle = 2 * math.pi - total_angle

        # Generate points by rotating v_start around normal
        pts = np.empty((n_points, 3), dtype=np.float64)
        for i, angle in enumerate(np.linspace(0, total_angle, n_points)):
            pts[i] = c + self._rodrigues_rotate(v_start, normal, angle)

        return Polyline(pts, self.start_vertex, self.end_vertex)

    @staticmethod
    def _rodrigues_rotate(
        v: NDArray, k: NDArray, theta: float
    ) -> NDArray[np.float64]:
        """Rotate vector v around axis k by angle theta (Rodrigues formula)."""
        return (
            v * math.cos(theta)
            + np.cross(k, v) * math.sin(theta)
            + k * np.dot(k, v) * (1 - math.cos(theta))
        )

    def total_length(self) -> float:
        """Approximate arc length."""
        return self.to_polyline(200).total_length()

    def __repr__(self) -> str:
        return (
            f"Arc(r={self.radius:.6f}, "
            f"v{self.start_vertex}→v{self.end_vertex})"
        )


@dataclass
class BlockEdge:
    """An edge of a structured block, defined by either a Polyline or Arc.

    Attributes:
        curve: The geometric curve (Polyline or Arc).
        edge_type: 'polyline', 'arc', or 'line' (straight).
        block_idx: Index of the owning block.
        local_edge_idx: Local edge index within the block (0-11 for hex).
    """

    curve: Polyline | Arc
    edge_type: Literal["polyline", "arc", "line"] = "polyline"
    block_idx: int = -1
    local_edge_idx: int = -1

    def to_polyline(self, n_points: int = 50) -> Polyline:
        """Convert to Polyline regardless of underlying type."""
        if isinstance(self.curve, Arc):
            return self.curve.to_polyline(n_points)
        return self.curve

    def resample_graded(self, n_points: int, grading: float = 1.0) -> Polyline:
        """Resample with grading applied."""
        pl = self.to_polyline(max(n_points * 5, 200))
        return pl.resample_graded(n_points, grading)
