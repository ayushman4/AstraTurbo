# AstraTurbo Tutorials

Step-by-step guides for using AstraTurbo. Each tutorial builds on the previous one.

---

## Tutorial 1: Create a Camber Line

A camber line is the mean line of a blade airfoil. It defines the blade's curvature.

### Imports

```python
import matplotlib.pyplot as plt
from astraturbo.camberline import NACA65
from astraturbo.distribution import Chebyshev
```

AstraTurbo provides 8 camber line types. This tutorial uses NACA 65-series,
the standard for axial compressor blades.

### Create the camber line

```python
camber = NACA65(cl0=1.0)            # Design lift coefficient = 1.0
camber.distribution = Chebyshev()    # Cluster points near LE and TE
camber.sample_rate = 200             # 200 points along the chord
```

The `cl0` parameter controls how much the camber line curves.
Higher values = more turning = higher pressure rise per stage.

| cl0 | Use case |
|-----|----------|
| 0.4 | Lightly loaded, high efficiency |
| 1.0 | Moderate loading (typical) |
| 1.8 | Heavily loaded, fewer stages needed |

### Get the coordinates

```python
points = camber.as_array()       # (200, 2) array of [x, y]
slopes = camber.get_derivations() # (200,) array of dy/dx

print(f"Points: {points.shape}")
print(f"Chord: x = {points[0, 0]:.3f} to {points[-1, 0]:.3f}")
print(f"Max camber: {points[:, 1].max():.4f}")
```

### Plot it

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='NACA 65 (CL0=1.0)')
ax.set_xlabel('x/c')
ax.set_ylabel('y/c')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('NACA 65-Series Camber Line')
plt.tight_layout()
plt.show()
```

### Try different camber lines

```python
from astraturbo.camberline import CircularArc, CubicPolynomial, Joukowski

cambers = {
    'NACA 65 (CL0=1.0)': NACA65(cl0=1.0),
    'Circular Arc (100°)': CircularArc(angle_of_inflow=100),
    'Cubic Polynomial': CubicPolynomial(angle_of_inflow=100, angle_of_outflow=90),
    'Joukowski (m=0.12)': Joukowski(max_camber=0.12),
}

fig, ax = plt.subplots(figsize=(10, 4))
for name, cl in cambers.items():
    pts = cl.as_array()
    ax.plot(pts[:, 0], pts[:, 1], label=name)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Available Camber Line Types')
plt.tight_layout()
plt.show()
```

Or use the CLI: `python -m astraturbo profile --camber naca65 --plot`

---

## Tutorial 2: Create a Thickness Distribution

A thickness distribution defines the airfoil's half-thickness at each
chordwise position.

### Imports

```python
import matplotlib.pyplot as plt
from astraturbo.thickness import NACA4Digit, NACA65Series, Elliptic
from astraturbo.distribution import Chebyshev
```

### Create a thickness distribution

```python
thickness = NACA4Digit(max_thickness=0.12)   # 12% maximum thickness
thickness.distribution = Chebyshev()
thickness.sample_rate = 200
```

`max_thickness` is the ratio of max thickness to chord:
- 0.06 = 6% (thin, low drag)
- 0.10 = 10% (typical compressor blade)
- 0.15 = 15% (structural, fan blade)

### Get coordinates and plot

```python
points = thickness.as_array()  # (200, 2) array of [x, half_thickness]

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)
ax.plot(points[:, 0], -points[:, 1], 'r-', linewidth=2)
ax.fill_between(points[:, 0], points[:, 1], -points[:, 1], alpha=0.2, color='red')
ax.set_xlabel('x/c')
ax.set_ylabel('t/c')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('NACA 4-Digit Thickness (12%)')
plt.tight_layout()
plt.show()
```

### Compare thickness distributions

```python
dists = {
    'NACA 4-Digit': NACA4Digit(max_thickness=0.10),
    'NACA 65-Series': NACA65Series(max_thickness=0.10),
    'Elliptic': Elliptic(max_thickness=0.10),
}

fig, ax = plt.subplots(figsize=(10, 3))
for name, td in dists.items():
    pts = td.as_array()
    ax.plot(pts[:, 0], pts[:, 1], label=name)
ax.set_xlabel('x/c')
ax.set_ylabel('Half-thickness / chord')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Thickness Distributions (10% max)')
plt.tight_layout()
plt.show()
```

---

## Tutorial 3: Design a Blade Profile

A blade profile combines a camber line with a thickness distribution
using the superposition method.

### Imports

```python
import matplotlib.pyplot as plt
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA4Digit
from astraturbo.profile import Superposition
```

### Create the profile

```python
camber = NACA65(cl0=1.0)
thickness = NACA4Digit(max_thickness=0.10)
profile = Superposition(camber, thickness)
```

The superposition method offsets thickness perpendicular to the camber line:
- Upper surface: `(x - t·sin(θ), y_c + t·cos(θ))`
- Lower surface: `(x + t·sin(θ), y_c - t·cos(θ))`

where θ is the camber slope angle.

### Get profile data

```python
coords = profile.as_array()          # (399, 2) closed contour
upper = profile.upper_surface()       # (200, 2) suction side
lower = profile.lower_surface()       # (200, 2) pressure side
centroid = profile.centroid           # [x, y] centroid

print(f"Profile: {len(coords)} points (closed contour)")
print(f"Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
```

### Plot the complete profile

```python
fig, ax = plt.subplots(figsize=(10, 4))

# Profile outline
ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2, label='Profile')

# Camber line
camber_pts = camber.as_array()
ax.plot(camber_pts[:, 0], camber_pts[:, 1], 'r--', linewidth=1, label='Camber line')

# Centroid
ax.plot(centroid[0], centroid[1], 'ko', markersize=5, label='Centroid')

ax.set_xlabel('x/c')
ax.set_ylabel('y/c')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('NACA 65-1-10 Compressor Profile')
plt.tight_layout()
plt.show()
```

### Save to CSV

```python
import numpy as np
np.savetxt('blade_profile.csv', coords, delimiter=',', header='x,y', comments='')
print("Saved to blade_profile.csv")
```

Or use the CLI:
```bash
python -m astraturbo profile --camber naca65 --thickness naca4digit --max-thickness 0.10 -o blade.csv --plot
```

---

## Tutorial 4: Build a 3D Blade Row

A blade row consists of multiple 2D profiles stacked along the span
(hub to tip) to form a 3D blade.

### Imports

```python
import numpy as np
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA65Series
from astraturbo.profile import Superposition
from astraturbo.blade import BladeRow
```

### Create profiles at different span positions

Real blades have different profiles at hub, mid-span, and tip because
flow conditions vary along the span.

```python
# Hub (inner): lower loading, thinner
hub = Superposition(NACA65(cl0=0.8), NACA65Series(max_thickness=0.08))

# Mid-span: moderate
mid = Superposition(NACA65(cl0=1.0), NACA65Series(max_thickness=0.10))

# Tip (outer): higher loading, thicker
tip = Superposition(NACA65(cl0=1.2), NACA65Series(max_thickness=0.12))
```

### Define the flow channel

The hub and shroud contours define the annular passage in the meridional
(z, r) plane:

```python
hub_contour = np.array([
    [0.00, 0.10],   # [z, r] at inlet
    [0.05, 0.10],   # at mid-chord
    [0.10, 0.10],   # at outlet
])
shroud_contour = np.array([
    [0.00, 0.20],
    [0.05, 0.20],
    [0.10, 0.20],
])
```

### Build the blade row

```python
row = BladeRow(
    hub_points=hub_contour,
    shroud_points=shroud_contour,
    stacking_mode=0,   # 0=axial, 1=radial, 2=cascade
)
row.number_blades = 24

# Add profiles from hub to tip
row.add_profile(hub)
row.add_profile(mid)
row.add_profile(tip)
```

### Compute 3D geometry

```python
row.compute(
    stagger_angles=np.deg2rad([30, 35, 40]),   # Twist from hub to tip
    chord_lengths=np.array([0.04, 0.05, 0.06]), # Chord varies with span
)

print(f"Blade surface: {row.blade_surface}")
print(f"Leading edge: {row.leading_edge.shape}")
print(f"Trailing edge: {row.trailing_edge.shape}")
print(f"3D profiles: {len(row.profiles_3d)} sections")
```

### Generate full annular array

```python
from astraturbo.blade import generate_blade_array_flat

all_blades = generate_blade_array_flat(row.profiles_3d, row.number_blades)
print(f"Full annulus: {len(all_blades)} points ({row.number_blades} blades)")
```

---

## Tutorial 5: Generate a Mesh

A structured mesh divides the blade passage into cells for CFD simulation.

### From the command line (quickest)

```bash
python -m astraturbo profile --camber naca65 --thickness naca4digit -o blade.csv
python -m astraturbo mesh --profile blade.csv --pitch 0.05 -o mesh.cgns
python -m astraturbo info mesh.cgns
```

### From Python

```python
from astraturbo.mesh.multiblock import generate_blade_passage_mesh
from astraturbo.mesh import mesh_quality_report

# Use the mid-span profile from Tutorial 4
profile_2d = mid.as_array()

mesh = generate_blade_passage_mesh(
    profile=profile_2d,
    pitch=0.05,           # Blade-to-blade spacing
    n_blade=40,           # Cells around blade
    n_ogrid=10,           # O-grid wall-normal cells
    n_inlet=15,           # Inlet block cells
    n_outlet=15,          # Outlet block cells
    n_passage=20,         # Passage pitchwise cells
    ogrid_thickness=0.005, # O-grid layer thickness
)

print(f"Mesh: {mesh.n_blocks} blocks, {mesh.total_cells} cells")

# Check quality
report = mesh_quality_report(mesh.blocks[0].points)
print(f"Max aspect ratio: {report['aspect_ratio_max']:.1f}")
print(f"Max skewness: {report['skewness_max']:.3f}")
```

### Export to CGNS

```python
mesh.export_cgns("compressor_mesh.cgns")
# Open in ParaView: paraview compressor_mesh.cgns
```

### Estimate y+ before meshing

```python
from astraturbo.mesh import first_cell_height_for_yplus

dy = first_cell_height_for_yplus(
    target_yplus=1.0,
    density=1.225,
    velocity=100.0,
    dynamic_viscosity=1.8e-5,
    chord=0.05,
)
print(f"First cell height for y+=1: {dy*1e6:.1f} μm")
```

Or CLI: `python -m astraturbo yplus --velocity 100 --chord 0.05`

---

## Tutorial 6: Design from Requirements (Meanline)

Instead of manually choosing blade angles, let the meanline analysis
calculate them from top-level requirements.

### From the command line

```bash
python -m astraturbo meanline --pr 4.0 --mass-flow 20 --rpm 12000 --r-hub 0.15 --r-tip 0.30
```

### From Python

```python
from astraturbo.design import meanline_compressor, meanline_to_blade_parameters

result = meanline_compressor(
    overall_pressure_ratio=4.0,
    mass_flow=20.0,      # kg/s
    rpm=12000,
    r_hub=0.15,          # meters
    r_tip=0.30,          # meters
)

print(result.summary())
```

Output:
```
Meanline Analysis: 5 stages
  Overall PR:   4.000
  Stage 1: PR=1.35, phi=0.48, psi=0.38, R=0.50
    Rotor beta: -52.1 → -38.4 deg
    De Haller: 0.78
  ...
```

### Convert to blade geometry parameters

```python
params = meanline_to_blade_parameters(result)
for p in params:
    print(f"Stage {p['stage']}:")
    print(f"  Rotor stagger: {p['rotor_stagger_deg']:.1f}°")
    print(f"  Stator stagger: {p['stator_stagger_deg']:.1f}°")
    print(f"  De Haller: {p['de_haller']:.3f}")
```

These parameters feed directly into the profile and blade modules.

---

## Tutorial 7: Set Up a CFD Case

### OpenFOAM

```bash
python -m astraturbo cfd --solver openfoam --velocity 100 --turbulence kOmegaSST -o my_case
cd my_case && bash Allrun
```

### From Python (with rotating frame)

```python
from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig

wf = CFDWorkflow(CFDWorkflowConfig(
    solver="openfoam",
    inlet_velocity=100.0,
    turbulence_model="kOmegaSST",
    is_rotating=True,
    omega=1200.0,
))
wf.set_mesh("mesh.cgns")
wf.setup_case("cfd_case/")
```

### ANSYS Fluent

```python
wf = CFDWorkflow(CFDWorkflowConfig(solver="fluent", inlet_velocity=80))
wf.setup_case("fluent_case/")
# Creates run.jou — run with: fluent 3ddp -i run.jou
```

### ANSYS CFX

```python
wf = CFDWorkflow(CFDWorkflowConfig(solver="cfx", is_rotating=True, omega=1500))
wf.setup_case("cfx_case/")
# Creates setup.ccl
```

---

## Tutorial 8: Structural Analysis (FEA)

### List available materials

```bash
python -m astraturbo fea --list-materials
```

### Set up a CalculiX analysis

```python
from astraturbo.fea import FEAWorkflow, FEAWorkflowConfig, get_material

fea = FEAWorkflow(FEAWorkflowConfig(
    material=get_material("inconel_718"),
    omega=1200.0,
    analysis_type="static",
))
fea.set_blade_surface(surface_points, ni, nj)
fea.setup("fea_case/")

# Quick stress estimate (no solver needed)
estimate = fea.estimate_stress_analytical()
print(f"Centrifugal stress: {estimate['centrifugal_stress_MPa']:.1f} MPa")
print(f"Safety factor: {estimate['safety_factor']:.2f}")
```

---

## Tutorial 9: AI Assistant

### Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Interactive chat

```bash
python -m astraturbo ai
```

### From Python

```python
from astraturbo.ai import create_assistant

assistant = create_assistant()
response = assistant.chat(
    "Design a 5-stage axial compressor with PR=8, "
    "mass flow 25 kg/s at 15000 RPM, hub=0.15m, tip=0.25m"
)
print(response)
```

The AI calls AstraTurbo tools automatically — meanline design, profile
generation, mesh generation, CFD setup, y+ calculation, and more.

### In the GUI

Launch `python -m astraturbo gui`, click the **AI Assistant** tab,
and type your request.

---

## Tutorial 10: Design Optimization

### Parametric sweep

```python
from astraturbo.foundation.design_chain import DesignChain

chain = DesignChain()
results = chain.sweep("cl0", start=0.6, end=1.4, steps=9)

for r in results:
    cl0 = r.parameters["cl0"]
    mesh_data = r.stages[-2].data if len(r.stages) > 2 else {}
    print(f"CL0={cl0:.1f}: {r.total_time:.3f}s")
```

### Multi-fidelity optimization

```python
from astraturbo.optimization import create_blade_design_space
from astraturbo.optimization.multifidelity import MultiFidelityOptimizer

ds = create_blade_design_space(n_profiles=3)
optimizer = MultiFidelityOptimizer.create_default_turbomachinery(
    ds, n_meanline=5000, n_throughflow=200, n_cfd=10,
)
result = optimizer.run()
print(f"Best design: {result.best_design()}")
```

---

## Tutorial 11: Import Existing Data

### OpenFOAM mesh

```bash
python -m astraturbo info /path/to/constant/polyMesh/points
```

```python
from astraturbo.export import read_openfoam_points, openfoam_points_to_cloud

points = read_openfoam_points("/path/to/points")
stats = openfoam_points_to_cloud(points)
print(f"{stats['n_points']:,} points, chord {stats['x_range']*1000:.1f} mm")
```

### Any mesh format

```python
from astraturbo.export import read_mesh

data = read_mesh("mesh.vtk")       # VTK
data = read_mesh("grid.cgns")      # CGNS
data = read_mesh("case.plt")       # Tecplot
data = read_mesh("mesh.su2")       # SU2
```

### List all supported formats

```bash
python -m astraturbo formats
```

---

## Complete End-to-End Example

Design a compressor from requirements to CFD-ready mesh in one script:

```python
import numpy as np
from astraturbo.design import meanline_compressor, meanline_to_blade_parameters
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA65Series
from astraturbo.profile import Superposition
from astraturbo.blade import BladeRow
from astraturbo.mesh.multiblock import generate_blade_passage_mesh
from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig

# 1. Meanline design
result = meanline_compressor(
    overall_pressure_ratio=4.0, mass_flow=20.0,
    rpm=12000, r_hub=0.15, r_tip=0.30,
)
params = meanline_to_blade_parameters(result)
print(f"Designed {result.n_stages} stages")

# 2. Generate profiles for first stage
stage = params[0]
profiles = [
    Superposition(NACA65(cl0=0.8), NACA65Series(max_thickness=0.08)),
    Superposition(NACA65(cl0=1.0), NACA65Series(max_thickness=0.10)),
    Superposition(NACA65(cl0=1.2), NACA65Series(max_thickness=0.12)),
]

# 3. Build 3D blade
row = BladeRow(
    hub_points=np.array([[0.0, 0.15], [0.10, 0.15]]),
    shroud_points=np.array([[0.0, 0.30], [0.10, 0.30]]),
)
for p in profiles:
    row.add_profile(p)
row.compute(
    stagger_angles=np.deg2rad([stage['rotor_inlet_beta_deg'],
                                stage['rotor_stagger_deg'],
                                stage['rotor_outlet_beta_deg']]),
    chord_lengths=np.array([0.04, 0.05, 0.06]),
)

# 4. Generate mesh
mesh = generate_blade_passage_mesh(
    profiles[1].as_array(), pitch=0.05,
    n_blade=40, n_ogrid=10, n_inlet=15, n_outlet=15, n_passage=20,
)
mesh.export_cgns("stage1_rotor.cgns")
print(f"Mesh: {mesh.n_blocks} blocks, {mesh.total_cells} cells")

# 5. Set up CFD
wf = CFDWorkflow(CFDWorkflowConfig(
    solver="openfoam", inlet_velocity=100.0, turbulence_model="kOmegaSST",
))
wf.set_mesh("stage1_rotor.cgns")
wf.setup_case("stage1_cfd/")
print("CFD case ready at stage1_cfd/")
```

Run it:
```bash
python design_compressor.py
cd stage1_cfd && bash Allrun
```
