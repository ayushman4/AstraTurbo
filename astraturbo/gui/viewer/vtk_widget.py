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
