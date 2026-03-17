"""Vertex extraction and block topology builder.

Parses blade profile data and block vertex arrays to automatically
build the edge/face topology needed for structured mesh generation.

This fills the adaptation requirement:
  "Extract block vertex co-ordinates and rearrange the vertices
   to form edges and faces of block."
  "Extract Polyline data and the corresponding co-ordinates of
   all points between the block vertices."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .polyline import Polyline, Arc, BlockEdge


@dataclass
class BlockVertex:
    """A vertex of a structured block."""

    index: int
    coords: NDArray[np.float64]  # (2,) or (3,)

    def __repr__(self) -> str:
        c = ", ".join(f"{x:.6f}" for x in self.coords)
        return f"V{self.index}({c})"


@dataclass
class BlockFace:
    """A face of a structured block (quad face defined by 4 vertex indices)."""

    vertex_indices: list[int]  # 4 indices forming the quad
    name: str = ""
    face_type: str = ""  # 'inlet', 'outlet', 'blade', 'hub', 'shroud', 'periodic'

    def __repr__(self) -> str:
        return f"Face({self.name}: {self.vertex_indices})"


@dataclass
class BlockTopology:
    """Complete topology for a single structured block.

    A 2D block has 4 vertices, 4 edges, and 1 face (the block itself).
    A 3D block has 8 vertices, 12 edges, and 6 faces.
    """

    vertices: list[BlockVertex] = field(default_factory=list)
    edges: list[BlockEdge] = field(default_factory=list)
    faces: list[BlockFace] = field(default_factory=list)
    block_index: int = -1
    cell_counts: list[int] = field(default_factory=lambda: [10, 10])
    gradings: list[float] = field(default_factory=lambda: [1.0, 1.0])


def extract_vertices_from_profile(
    profile: NDArray[np.float64],
    le_index: int | None = None,
    te_index: int = 0,
) -> dict[str, BlockVertex]:
    """Extract key vertices (LE, TE, suction peak, pressure peak) from a profile.

    Args:
        profile: (N, 2) closed airfoil profile points.
        le_index: Index of leading edge point. None = auto-detect (min x).
        te_index: Index of trailing edge point (default 0 = start of array).

    Returns:
        Dict with named vertices: 'te', 'le', 'ss_peak', 'ps_peak'.
    """
    n = len(profile)
    if le_index is None:
        le_index = int(np.argmin(profile[:, 0]))

    # TE is at start/end of closed contour
    te_pt = profile[te_index]
    le_pt = profile[le_index]

    # Suction side peak (max y on upper surface)
    upper = profile[:le_index + 1]
    ss_peak_idx = int(np.argmax(upper[:, 1]))
    ss_peak = upper[ss_peak_idx]

    # Pressure side peak (min y on lower surface)
    lower = profile[le_index:]
    ps_peak_idx = int(np.argmin(lower[:, 1]))
    ps_peak = lower[ps_peak_idx]

    return {
        "te": BlockVertex(0, te_pt),
        "le": BlockVertex(1, le_pt),
        "ss_peak": BlockVertex(2, ss_peak),
        "ps_peak": BlockVertex(3, ps_peak),
    }


def extract_polylines_from_profile(
    profile: NDArray[np.float64],
    le_index: int | None = None,
) -> dict[str, Polyline]:
    """Extract polyline segments from a closed blade profile.

    Splits the profile at the leading edge into:
      - suction_side: TE → LE along the suction (upper) surface
      - pressure_side: LE → TE along the pressure (lower) surface

    Also extracts the full profile as a single closed polyline.

    Args:
        profile: (N, 2) closed airfoil profile array.
        le_index: Index of the LE. None = auto-detect.

    Returns:
        Dict with 'suction', 'pressure', 'full' polylines.
    """
    n = len(profile)
    if le_index is None:
        le_index = int(np.argmin(profile[:, 0]))

    # Suction side: from index 0 (TE) to le_index (LE)
    suction_pts = profile[: le_index + 1]
    # Pressure side: from le_index (LE) to end (TE)
    pressure_pts = profile[le_index:]

    return {
        "suction": Polyline(suction_pts, start_vertex=0, end_vertex=1),
        "pressure": Polyline(pressure_pts, start_vertex=1, end_vertex=0),
        "full": Polyline(profile, start_vertex=0, end_vertex=0),
    }


def build_passage_vertices(
    profile: NDArray[np.float64],
    pitch: float,
    inlet_offset: float,
    outlet_offset: float,
    le_index: int | None = None,
) -> list[BlockVertex]:
    """Build all block vertices for a blade passage domain.

    Creates vertices at the blade LE/TE, passage inlet/outlet,
    and periodic boundaries (upper/lower pitch lines).

    Vertex numbering:
        0: TE (blade)
        1: LE (blade)
        2: Inlet, lower periodic
        3: Inlet, upper periodic
        4: Outlet, lower periodic
        5: Outlet, upper periodic
        6: Inlet, blade (LE - inlet_offset)
        7: Outlet, blade (TE + outlet_offset)

    Args:
        profile: (N, 2) closed profile.
        pitch: Blade pitch (passage width).
        inlet_offset: Distance upstream of LE to inlet boundary.
        outlet_offset: Distance downstream of TE to outlet boundary.
        le_index: LE point index (None = auto).

    Returns:
        List of BlockVertex objects.
    """
    if le_index is None:
        le_index = int(np.argmin(profile[:, 0]))

    te = profile[0]
    le = profile[le_index]
    y_mid = (te[1] + le[1]) / 2.0

    x_inlet = le[0] - inlet_offset
    x_outlet = te[0] + outlet_offset
    y_lower = y_mid - pitch / 2.0
    y_upper = y_mid + pitch / 2.0

    vertices = [
        BlockVertex(0, te.copy()),                          # TE
        BlockVertex(1, le.copy()),                          # LE
        BlockVertex(2, np.array([x_inlet, y_lower])),       # Inlet lower
        BlockVertex(3, np.array([x_inlet, y_upper])),       # Inlet upper
        BlockVertex(4, np.array([x_outlet, y_lower])),      # Outlet lower
        BlockVertex(5, np.array([x_outlet, y_upper])),      # Outlet upper
        BlockVertex(6, np.array([x_inlet, le[1]])),         # Inlet at blade height
        BlockVertex(7, np.array([x_outlet, te[1]])),        # Outlet at blade height
    ]
    return vertices


def build_passage_edges(
    vertices: list[BlockVertex],
    profile: NDArray[np.float64],
    le_index: int | None = None,
) -> list[BlockEdge]:
    """Build polyline edges connecting passage vertices.

    Creates edges for:
      - Blade surface (suction and pressure)
      - Inlet boundary
      - Outlet boundary
      - Periodic boundaries (upper/lower)
      - Internal connections

    Args:
        vertices: List of BlockVertex from build_passage_vertices().
        profile: (N, 2) closed profile.
        le_index: LE index.

    Returns:
        List of BlockEdge objects.
    """
    if le_index is None:
        le_index = int(np.argmin(profile[:, 0]))

    v = {vtx.index: vtx.coords for vtx in vertices}

    # Blade polylines
    suction_pts = profile[: le_index + 1]
    pressure_pts = profile[le_index:]

    edges = []

    # Edge 0: Suction side (TE → LE)
    edges.append(BlockEdge(
        curve=Polyline(suction_pts, start_vertex=0, end_vertex=1),
        edge_type="polyline", block_idx=0, local_edge_idx=0,
    ))

    # Edge 1: Pressure side (LE → TE)
    edges.append(BlockEdge(
        curve=Polyline(pressure_pts, start_vertex=1, end_vertex=0),
        edge_type="polyline", block_idx=0, local_edge_idx=1,
    ))

    # Edge 2: Inlet boundary (lower → upper)
    inlet_pts = np.array([v[2], v[6], v[3]])
    edges.append(BlockEdge(
        curve=Polyline(inlet_pts, start_vertex=2, end_vertex=3),
        edge_type="line", block_idx=-1, local_edge_idx=2,
    ))

    # Edge 3: Outlet boundary (lower → upper)
    outlet_pts = np.array([v[4], v[7], v[5]])
    edges.append(BlockEdge(
        curve=Polyline(outlet_pts, start_vertex=4, end_vertex=5),
        edge_type="line", block_idx=-1, local_edge_idx=3,
    ))

    # Edge 4: Lower periodic (inlet → outlet)
    lower_pts = np.array([v[2], v[4]])
    edges.append(BlockEdge(
        curve=Polyline(lower_pts, start_vertex=2, end_vertex=4),
        edge_type="line", block_idx=-1, local_edge_idx=4,
    ))

    # Edge 5: Upper periodic (inlet → outlet)
    upper_pts = np.array([v[3], v[5]])
    edges.append(BlockEdge(
        curve=Polyline(upper_pts, start_vertex=3, end_vertex=5),
        edge_type="line", block_idx=-1, local_edge_idx=5,
    ))

    return edges


def build_passage_faces(vertices: list[BlockVertex]) -> list[BlockFace]:
    """Build boundary faces for the blade passage domain.

    Returns:
        List of BlockFace with typed boundaries.
    """
    return [
        BlockFace([2, 6, 3], name="inlet", face_type="inlet"),
        BlockFace([4, 7, 5], name="outlet", face_type="outlet"),
        BlockFace([0, 1], name="blade", face_type="wall"),
        BlockFace([2, 4], name="periodic_lower", face_type="cyclic"),
        BlockFace([3, 5], name="periodic_upper", face_type="cyclic"),
    ]


def build_block_topology_from_profile(
    profile: NDArray[np.float64],
    pitch: float,
    inlet_offset: float | None = None,
    outlet_offset: float | None = None,
) -> BlockTopology:
    """Build complete block topology from a blade profile.

    This is the main entry point that combines vertex extraction,
    edge building, and face assignment into a single topology object.

    Args:
        profile: (N, 2) closed blade profile.
        pitch: Blade pitch.
        inlet_offset: Upstream distance (None = 50% chord).
        outlet_offset: Downstream distance (None = 50% chord).

    Returns:
        BlockTopology ready for mesh generation.
    """
    le_index = int(np.argmin(profile[:, 0]))
    x_min = profile[:, 0].min()
    x_max = profile[:, 0].max()
    chord = x_max - x_min

    if inlet_offset is None:
        inlet_offset = 0.5 * chord
    if outlet_offset is None:
        outlet_offset = 0.5 * chord

    vertices = build_passage_vertices(
        profile, pitch, inlet_offset, outlet_offset, le_index
    )
    edges = build_passage_edges(vertices, profile, le_index)
    faces = build_passage_faces(vertices)

    return BlockTopology(
        vertices=vertices,
        edges=edges,
        faces=faces,
    )


def import_profile_points_from_xml(xml_path: str) -> dict[str, NDArray[np.float64]]:
    """Import airfoil profile point data from an XML file.

    Supports legacy XML format and generic point-list XML.

    Expected XML structure:
        <profile>
            <points>
                <point x="0.0" y="0.0" />
                <point x="0.1" y="0.02" />
                ...
            </points>
        </profile>

    Or legacy XML format with camber/thickness arrays.

    Args:
        xml_path: Path to XML file.

    Returns:
        Dict with 'profile', 'camber', 'thickness' arrays where available.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}

    # Try generic point format
    for points_elem in root.iter("points"):
        pts = []
        for pt in points_elem.findall("point"):
            x = float(pt.get("x", pt.get("X", 0)))
            y = float(pt.get("y", pt.get("Y", 0)))
            pts.append([x, y])
        if pts:
            result["profile"] = np.array(pts, dtype=np.float64)

    # Try legacy XML format — arrays stored as text
    for tag in ["camberLine", "camber", "thicknessDistribution", "thickness",
                "profilePoints", "profile", "upperSurface", "lowerSurface"]:
        for elem in root.iter(tag):
            text = elem.text
            if text and text.strip():
                try:
                    values = [float(v) for v in text.strip().split()]
                    n = len(values) // 2
                    arr = np.array(values).reshape(n, 2)
                    result[tag] = arr
                except (ValueError, IndexError):
                    pass

            # Also check for nested x/y elements
            x_elem = elem.find("x")
            y_elem = elem.find("y")
            if x_elem is not None and y_elem is not None:
                try:
                    x_vals = [float(v) for v in x_elem.text.strip().split()]
                    y_vals = [float(v) for v in y_elem.text.strip().split()]
                    result[tag] = np.column_stack((x_vals, y_vals))
                except (ValueError, AttributeError):
                    pass

    return result
