"""
AstraTurbo — Open-source integrated turbomachinery design and simulation platform.

Modules:
    foundation   - Property system, signals, undo/redo, serialization
    baseclass    - Base objects, tree nodes, drawable mixin
    camberline   - Camber line generators (circular arc, polynomial, NACA, NURBS)
    thickness    - Thickness distributions (polynomial, NACA, elliptic, DCA)
    distribution - Point sampling distributions (Chebyshev, linear, weighted)
    profile      - 2D airfoil profile construction (superposition)
    blade        - 3D blade geometry (stacking, surfaces, hub/shroud)
    nurbs        - NURBS curve/surface utilities (via geomdl)
    machine      - Turbomachine container and project management
    mesh         - Mesh generation (SCM, O-grid, transfinite interpolation)
    export       - Export writers (CGNS, OpenFOAM, STEP, IGES, VTK)
    cfd          - CFD solver interface (OpenFOAM, SU2)
    optimization - Design optimization (pymoo integration)
    gui          - PySide6 graphical interface
    cli          - Command-line interface
"""

__version__ = "0.1.0"
__project__ = "AstraTurbo"
