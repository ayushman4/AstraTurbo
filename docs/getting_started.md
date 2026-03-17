# Getting Started with AstraTurbo

## Installation

### Prerequisites

- Python 3.9 or later
- pip

### Install from source

```bash
git clone https://github.com/ayushman4/AstraTurbo.git
cd AstraTurbo

# Core only (no GUI)
pip install -e .

# With GUI support
pip install -e ".[gui]"

# Everything including dev tools
pip install -e ".[all]"
```

### Verify installation

```bash
python -m astraturbo --version
# astraturbo 0.1.0

python -m pytest tests/ -q
# 123 passed
```

---

## Quick Start: CLI (fastest way to try)

You don't need the GUI to use AstraTurbo. The CLI does everything:

```bash
# Generate a blade profile
python -m astraturbo profile --camber naca65 --thickness naca4digit -o blade.csv

# Generate a structured mesh
python -m astraturbo mesh --profile blade.csv --pitch 0.05 -o mesh.cgns

# Inspect the result
python -m astraturbo info mesh.cgns

# Set up an OpenFOAM case
python -m astraturbo cfd --solver openfoam --velocity 100 -o my_case
```

---

## Quick Start: GUI

```bash
python -m astraturbo gui
```

When the window opens:

1. The **2D Profile** tab shows a default NACA 65-1-10 airfoil. Change the camber and thickness dropdowns to see different profiles.
2. Click **Compute > Compute Blade Geometry** to build the 3D blade from the 3 default span profiles.
3. Click **Compute > Generate Multi-Block Mesh** to create a structured mesh. The quality report appears in the bottom panel.
4. Click **File > Export > CGNS Mesh** to save the mesh. Open it in ParaView to verify.

---

## Tutorial 1: Create a NACA 65-Series Compressor Profile

This is the most common profile type for axial compressor blades.

```python
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA65Series
from astraturbo.profile import Superposition

# NACA 65-series with design lift coefficient 1.0
camber = NACA65(cl0=1.0)

# 10% maximum thickness
thickness = NACA65Series(max_thickness=0.10)

# Combine into a closed airfoil
profile = Superposition(camber, thickness)

# Get the coordinates
coords = profile.as_array()          # (399, 2) closed contour
upper = profile.upper_surface()       # (200, 2) suction side
lower = profile.lower_surface()       # (200, 2) pressure side

print(f"Profile has {len(coords)} points")
print(f"Centroid at: {profile.centroid}")
```

### Plot the profile

```python
import matplotlib.pyplot as plt

coords = profile.as_array()
camber_pts = camber.as_array()

plt.figure(figsize=(10, 4))
plt.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2, label='Profile')
plt.plot(camber_pts[:, 0], camber_pts[:, 1], 'r--', label='Camber line')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('NACA 65-1-10 Compressor Profile')
plt.xlabel('x/c')
plt.ylabel('y/c')
plt.show()
```

Or via CLI: `python -m astraturbo profile --camber naca65 --thickness naca65 --plot`

---

## Tutorial 2: Build a 3D Axial Compressor Blade

```python
import numpy as np
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA65Series
from astraturbo.profile import Superposition
from astraturbo.blade import BladeRow

# Profiles at hub, mid-span, and tip
profiles = [
    Superposition(NACA65(cl0=0.8), NACA65Series(max_thickness=0.08)),
    Superposition(NACA65(cl0=1.0), NACA65Series(max_thickness=0.10)),
    Superposition(NACA65(cl0=1.2), NACA65Series(max_thickness=0.12)),
]

# Define the flow channel
hub = np.array([[0.0, 0.10], [0.05, 0.10], [0.10, 0.10]])
shroud = np.array([[0.0, 0.20], [0.05, 0.20], [0.10, 0.20]])

# Build the blade row
row = BladeRow(hub_points=hub, shroud_points=shroud, stacking_mode=0)
row.number_blades = 24

for p in profiles:
    row.add_profile(p)

# Compute 3D geometry (varying stagger and chord from hub to tip)
row.compute(
    stagger_angles=np.deg2rad([30, 35, 40]),
    chord_lengths=np.array([0.04, 0.05, 0.06]),
)

print(f"Leading edge: {row.leading_edge.shape}")
print(f"Trailing edge: {row.trailing_edge.shape}")
print(f"3D profiles: {len(row.profiles_3d)} sections")
```

---

## Tutorial 3: Generate a Multi-Block Mesh and Export CGNS

```python
from astraturbo.mesh.multiblock import generate_blade_passage_mesh
from astraturbo.mesh import mesh_quality_report

# Use the mid-span profile from above
profile_2d = profiles[1].as_array()

mesh = generate_blade_passage_mesh(
    profile=profile_2d, pitch=0.05,
    n_blade=40, n_ogrid=10, n_inlet=15, n_outlet=15, n_passage=20,
    ogrid_thickness=0.005, grading_ogrid=1.3,
)

print(f"Mesh: {mesh.n_blocks} blocks, {mesh.total_cells} cells")

# Check quality
report = mesh_quality_report(mesh.blocks[0].points)
print(f"Max aspect ratio: {report['aspect_ratio_max']:.1f}")
print(f"Max skewness: {report['skewness_max']:.3f}")

# Export
mesh.export_cgns("compressor_mesh.cgns")
```

Open `compressor_mesh.cgns` in ParaView to visualize.

---

## Tutorial 4: Set Up an OpenFOAM CFD Case

```python
from astraturbo.cfd import create_openfoam_case

case = create_openfoam_case(
    case_dir="compressor_cfd",
    solver="simpleFoam",
    turbulence_model="kOmegaSST",
    inlet_velocity=100.0,
    viscosity=1.5e-5,
)
print(f"Case created at: {case}")
```

Or via CLI:

```bash
python -m astraturbo cfd --solver openfoam --velocity 100 -o compressor_cfd
```

Then run in terminal:
```bash
cd compressor_cfd
blockMesh          # Generate mesh
simpleFoam         # Run solver
paraFoam           # Visualize
```

---

## Tutorial 5: Multi-Stage Rotor+Stator

```python
from astraturbo.mesh.multistage import MultistageGenerator, RowMeshConfig

gen = MultistageGenerator()
gen.add_row("rotor", RowMeshConfig(
    profile=profiles[1].as_array(), pitch=0.05, is_rotor=True,
    n_blade=30, n_ogrid=8, n_inlet=10, n_outlet=10, n_passage=15,
))
gen.add_row("stator", RowMeshConfig(
    profile=profiles[1].as_array(), pitch=0.06, is_rotor=False,
    n_blade=30, n_ogrid=8, n_inlet=10, n_outlet=10, n_passage=15,
))
result = gen.generate()
print(f"Stage: {result.n_rows} rows, {result.total_cells} cells")

# Export combined or per-row
result.export_cgns("stage.cgns")
result.export_cgns_per_row("per_row/")
```

---

## Tutorial 6: Read an Existing OpenFOAM Mesh

```python
from astraturbo.export import read_openfoam_points, openfoam_points_to_cloud

points = read_openfoam_points("/path/to/constant/polyMesh/points")
stats = openfoam_points_to_cloud(points)

print(f"Points: {stats['n_points']:,}")
print(f"Chord:  {stats['x_range']*1000:.1f} mm")
print(f"Span:   {stats['z_range']*1000:.1f} mm")
```

Or via CLI: `python -m astraturbo info /path/to/points`

---

## Tutorial 7: Estimate y+ for Mesh Design

Before meshing, calculate the required first cell height:

```python
from astraturbo.mesh import first_cell_height_for_yplus, estimate_yplus

# What cell height for y+ = 1?
dy = first_cell_height_for_yplus(
    target_yplus=1.0, density=1.225,
    velocity=100.0, dynamic_viscosity=1.8e-5, chord=0.10,
)
print(f"First cell height for y+=1: {dy*1000:.4f} mm")

# What y+ will I get with a given cell height?
yp = estimate_yplus(dy, 1.225, 100.0, 1.8e-5, 0.10)
print(f"Estimated y+: {yp:.1f}")
```

---

## Module Reference

| Module | Import | Purpose |
|---|---|---|
| `astraturbo.camberline` | `from astraturbo.camberline import NACA65` | Camber line generators |
| `astraturbo.thickness` | `from astraturbo.thickness import NACA4Digit` | Thickness distributions |
| `astraturbo.profile` | `from astraturbo.profile import Superposition` | 2D profile construction |
| `astraturbo.blade` | `from astraturbo.blade import BladeRow` | 3D blade geometry |
| `astraturbo.nurbs` | `from astraturbo.nurbs import interpolate_3d` | NURBS utilities |
| `astraturbo.machine` | `from astraturbo.machine import TurboMachine` | Machine container |
| `astraturbo.mesh` | `from astraturbo.mesh import SCMMesher, OGridGenerator` | Mesh generation |
| `astraturbo.mesh.multiblock` | `from astraturbo.mesh.multiblock import generate_blade_passage_mesh` | Multi-block mesher |
| `astraturbo.mesh.multistage` | `from astraturbo.mesh.multistage import MultistageGenerator` | Multi-row stages |
| `astraturbo.export` | `from astraturbo.export import write_cgns_structured` | File export |
| `astraturbo.export` | `from astraturbo.export import read_openfoam_points` | OpenFOAM import |
| `astraturbo.cfd` | `from astraturbo.cfd import create_openfoam_case` | CFD case setup |
| `astraturbo.fea` | `from astraturbo.fea import FEAWorkflow, get_material` | Structural analysis |
| `astraturbo.optimization` | `from astraturbo.optimization import Optimizer` | Design optimization |
| `astraturbo.ai` | `from astraturbo.ai import create_assistant` | AI assistant |

---

## CLI Command Reference

| Command | Description |
|---|---|
| `python -m astraturbo gui` | Launch graphical interface |
| `python -m astraturbo ai` | AI design assistant (interactive chat) |
| `python -m astraturbo ai "prompt"` | AI single request |
| `python -m astraturbo profile [options]` | Generate a 2D blade profile |
| `python -m astraturbo mesh [options]` | Generate a mesh from a profile CSV |
| `python -m astraturbo meanline [options]` | Meanline compressor design |
| `python -m astraturbo cfd [options]` | Set up CFD case (OpenFOAM/Fluent/CFX/SU2) |
| `python -m astraturbo fea [options]` | Set up FEA structural analysis |
| `python -m astraturbo yplus [options]` | y+ / cell height calculator |
| `python -m astraturbo info <file>` | Inspect a mesh/points/CSV file |
| `python -m astraturbo formats` | List 30 supported file formats |
| `python -m astraturbo optimize [options]` | Run blade optimization |
| `python -m astraturbo multistage [options]` | Multi-row stage mesh |
| `python -m astraturbo run <case>` | Execute CFD/FEA solver |
| `python -m astraturbo --help` | Show all commands |
