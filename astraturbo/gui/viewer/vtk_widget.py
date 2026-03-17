"""VTK-based 3D viewer widget for blade geometry and mesh visualization."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel

import numpy as np
from numpy.typing import NDArray

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


class VTKViewerWidget(QWidget):
    """3D viewer using VTK for blade and mesh visualization."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_VTK:
            self._vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self._vtk_widget)

            self._renderer = vtk.vtkRenderer()
            self._renderer.SetBackground(0.95, 0.95, 0.95)
            self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

            self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
            style = vtk.vtkInteractorStyleTrackballCamera()
            self._interactor.SetInteractorStyle(style)
        else:
            layout.addWidget(QLabel(
                "VTK not installed. Install with:\n  pip install vtk"
            ))
            self._vtk_widget = None
            self._renderer = None

    def add_surface_from_points(
        self,
        points: NDArray[np.float64],
        ni: int,
        nj: int,
        color: tuple[float, float, float] = (0.6, 0.6, 0.9),
        opacity: float = 1.0,
    ) -> None:
        """Add a structured surface to the 3D view.

        Args:
            points: (ni*nj, 3) array of surface points.
            ni, nj: Grid dimensions.
            color: RGB tuple (0-1).
            opacity: Surface opacity (0-1).
        """
        if not HAS_VTK or self._renderer is None:
            return

        vtk_points = vtk.vtkPoints()
        for pt in points:
            vtk_points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))

        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nj, ni, 1)
        grid.SetPoints(vtk_points)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(grid)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)

        self._renderer.AddActor(actor)

    def add_point_cloud(
        self,
        points: NDArray[np.float64],
        color: tuple[float, float, float] = (1, 0, 0),
        point_size: float = 3.0,
    ) -> None:
        """Add a point cloud to the 3D view."""
        if not HAS_VTK or self._renderer is None:
            return

        vtk_points = vtk.vtkPoints()
        for pt in points:
            vtk_points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        vtk_cells = vtk.vtkCellArray()
        for i in range(len(points)):
            vtk_cells.InsertNextCell(1)
            vtk_cells.InsertCellPoint(i)
        polydata.SetVerts(vtk_cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetPointSize(point_size)

        self._renderer.AddActor(actor)

    def clear(self) -> None:
        """Remove all actors from the scene."""
        if self._renderer is not None:
            self._renderer.RemoveAllViewProps()

    def reset_camera(self) -> None:
        """Reset camera to fit all objects."""
        if self._renderer is not None:
            self._renderer.ResetCamera()

    def render(self) -> None:
        """Trigger a render update."""
        if self._vtk_widget is not None:
            self._vtk_widget.GetRenderWindow().Render()

    def initialize(self) -> None:
        """Initialize the VTK interactor (call after widget is shown)."""
        if self._interactor is not None:
            self._interactor.Initialize()

    # ── Enhanced 3D rendering methods ──────────────────────────

    def add_blade_surface(
        self,
        profiles_3d: NDArray[np.float64],
        ni: int,
        nj: int,
        color: tuple[float, float, float] = (0.7, 0.75, 0.85),
        opacity: float = 1.0,
        smooth_normals: bool = True,
    ) -> None:
        """Render a smooth NURBS blade surface using vtkStructuredGrid.

        Creates a high-quality surface rendering from a grid of 3D profile
        points. The surface is shaded with smooth normals for realistic
        blade visualization.

        Args:
            profiles_3d: (ni*nj, 3) array of surface points ordered as
                [profile_0_pt_0, profile_0_pt_1, ..., profile_1_pt_0, ...].
                ni = number of spanwise sections (profiles).
                nj = number of points per profile.
            ni: Number of spanwise sections.
            nj: Number of chordwise points per section.
            color: RGB surface color tuple (0-1).
            opacity: Surface opacity (0-1).
            smooth_normals: If True, compute smooth normals for better
                surface rendering quality.
        """
        if not HAS_VTK or self._renderer is None:
            return

        # Build VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(ni * nj)
        for idx in range(ni * nj):
            vtk_points.SetPoint(
                idx,
                float(profiles_3d[idx, 0]),
                float(profiles_3d[idx, 1]),
                float(profiles_3d[idx, 2]),
            )

        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nj, ni, 1)
        grid.SetPoints(vtk_points)

        if smooth_normals:
            # Extract surface and compute smooth normals
            surface_filter = vtk.vtkStructuredGridGeometryFilter()
            surface_filter.SetInputData(grid)
            surface_filter.Update()

            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(surface_filter.GetOutputPort())
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOn()
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
        else:
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(grid)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20.0)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetAmbient(0.1)

        self._renderer.AddActor(actor)

    def add_mesh_wireframe(
        self,
        block_points: NDArray[np.float64],
        ni: int,
        nj: int,
        color: tuple[float, float, float] = (0.1, 0.1, 0.1),
        line_width: float = 1.0,
        opacity: float = 0.6,
    ) -> None:
        """Render a structured mesh as a wireframe overlay.

        Displays mesh lines in both i and j directions, useful for
        visualizing mesh quality and topology over a blade surface.

        Args:
            block_points: (ni*nj, 3) array of mesh points, or (ni, nj, 3) array.
            ni: Points in i-direction (e.g., streamwise).
            nj: Points in j-direction (e.g., spanwise or pitchwise).
            color: RGB wireframe color.
            line_width: Width of mesh lines in pixels.
            opacity: Line opacity (0-1).
        """
        if not HAS_VTK or self._renderer is None:
            return

        # Reshape if needed
        if block_points.ndim == 3:
            pts = block_points.reshape(-1, 3)
        else:
            pts = block_points

        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(ni * nj)
        for idx in range(ni * nj):
            vtk_points.SetPoint(
                idx,
                float(pts[idx, 0]),
                float(pts[idx, 1]),
                float(pts[idx, 2]),
            )

        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nj, ni, 1)
        grid.SetPoints(vtk_points)

        # Extract edges as wireframe
        edges = vtk.vtkExtractEdges()
        edges.SetInputData(grid)
        edges.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(edges.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetRepresentationToWireframe()

        self._renderer.AddActor(actor)

    def add_hub_shroud_surface(
        self,
        contour_points: NDArray[np.float64],
        n_blades: int,
        surface_type: str = "full",
        color: tuple[float, float, float] = (0.85, 0.85, 0.85),
        opacity: float = 0.5,
        n_circumferential: int = 72,
    ) -> None:
        """Render hub or shroud as a surface of revolution.

        Takes a 2D meridional contour (axial, radial) and revolves it
        around the machine axis to create a 3D hub/shroud surface.

        Args:
            contour_points: (N, 2) array of (axial, radial) points defining
                the meridional contour of the hub or shroud.
            n_blades: Number of blades (used for 'passage' surface_type).
            surface_type: Rendering mode:
                'full' - complete 360-degree revolution
                'passage' - single blade passage (360/n_blades degrees)
                'half' - 180-degree revolution (cross-section view)
            color: RGB surface color.
            opacity: Surface opacity.
            n_circumferential: Number of points in circumferential direction.
        """
        if not HAS_VTK or self._renderer is None:
            return

        n_meridional = len(contour_points)
        if n_meridional < 2:
            return

        # Determine angular extent
        if surface_type == "passage":
            theta_max = 2.0 * np.pi / max(n_blades, 1)
        elif surface_type == "half":
            theta_max = np.pi
        else:
            theta_max = 2.0 * np.pi

        # Reduce circumferential points for partial surfaces
        if surface_type == "passage":
            n_circ = max(int(n_circumferential * theta_max / (2 * np.pi)), 8)
        else:
            n_circ = n_circumferential

        theta_vals = np.linspace(0, theta_max, n_circ)

        # Build surface points by revolving the contour
        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(n_meridional * n_circ)

        for i in range(n_meridional):
            x_axial = float(contour_points[i, 0])
            r = float(contour_points[i, 1])

            for j in range(n_circ):
                theta = theta_vals[j]
                x = x_axial
                y = r * np.cos(theta)
                z = r * np.sin(theta)
                vtk_points.SetPoint(i * n_circ + j, x, y, z)

        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(n_circ, n_meridional, 1)
        grid.SetPoints(vtk_points)

        # Extract surface with normals
        surface_filter = vtk.vtkStructuredGridGeometryFilter()
        surface_filter.SetInputData(grid)
        surface_filter.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(surface_filter.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToPhong()

        self._renderer.AddActor(actor)

    def add_contour_map(
        self,
        surface_points: NDArray[np.float64],
        scalar_field: NDArray[np.float64],
        ni: int,
        nj: int,
        colormap: str = "jet",
        scalar_range: tuple[float, float] | None = None,
        show_colorbar: bool = True,
        title: str = "",
    ) -> None:
        """Map a scalar field (pressure, temperature, etc.) onto a surface.

        Creates a color-mapped visualization of a scalar field on a
        structured surface grid. Supports standard VTK colormaps.

        Args:
            surface_points: (ni*nj, 3) array of surface point coordinates.
            scalar_field: (ni*nj,) array of scalar values at each point.
            ni, nj: Grid dimensions.
            colormap: Colormap name. Supported: 'jet', 'rainbow', 'coolwarm',
                'viridis', 'hot', 'diverging'.
            scalar_range: (min, max) range for color mapping. If None,
                uses the data range.
            show_colorbar: If True, add a scalar bar (legend) to the view.
            title: Title for the scalar bar.
        """
        if not HAS_VTK or self._renderer is None:
            return

        n_pts = ni * nj
        if len(surface_points) < n_pts or len(scalar_field) < n_pts:
            return

        # Build VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(n_pts)
        for idx in range(n_pts):
            vtk_points.SetPoint(
                idx,
                float(surface_points[idx, 0]),
                float(surface_points[idx, 1]),
                float(surface_points[idx, 2]),
            )

        # Build scalar array
        scalars = vtk.vtkFloatArray()
        scalars.SetName(title or "Scalar")
        scalars.SetNumberOfTuples(n_pts)
        for idx in range(n_pts):
            scalars.SetValue(idx, float(scalar_field[idx]))

        # Create structured grid with scalars
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nj, ni, 1)
        grid.SetPoints(vtk_points)
        grid.GetPointData().SetScalars(scalars)

        # Extract surface
        surface_filter = vtk.vtkStructuredGridGeometryFilter()
        surface_filter.SetInputData(grid)
        surface_filter.Update()

        # Create lookup table (colormap)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)

        if colormap == "coolwarm":
            lut.SetHueRange(0.667, 0.0)
            lut.SetSaturationRange(0.8, 0.8)
        elif colormap == "hot":
            lut.SetHueRange(0.0, 0.15)
            lut.SetSaturationRange(1.0, 0.5)
        elif colormap == "viridis":
            # Approximate viridis with VTK LUT
            lut.SetHueRange(0.75, 0.15)
            lut.SetSaturationRange(0.7, 0.9)
            lut.SetValueRange(0.3, 1.0)
        elif colormap == "rainbow":
            lut.SetHueRange(0.0, 0.667)
        elif colormap == "diverging":
            lut.SetHueRange(0.667, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
        else:
            # Default: jet-like
            lut.SetHueRange(0.667, 0.0)

        lut.Build()

        # Set scalar range
        if scalar_range is not None:
            s_min, s_max = scalar_range
        else:
            s_min = float(np.min(scalar_field[:n_pts]))
            s_max = float(np.max(scalar_field[:n_pts]))
            if abs(s_max - s_min) < 1e-15:
                s_max = s_min + 1.0
        lut.SetTableRange(s_min, s_max)

        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface_filter.GetOutputPort())
        mapper.SetScalarRange(s_min, s_max)
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToPhong()

        self._renderer.AddActor(actor)

        # Add scalar bar (colorbar)
        if show_colorbar:
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(lut)
            scalar_bar.SetTitle(title or "Value")
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetWidth(0.08)
            scalar_bar.SetHeight(0.5)
            scalar_bar.SetPosition(0.9, 0.25)
            scalar_bar.GetTitleTextProperty().SetFontSize(12)
            scalar_bar.GetLabelTextProperty().SetFontSize(10)

            self._renderer.AddActor2D(scalar_bar)
