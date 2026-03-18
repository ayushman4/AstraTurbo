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
# 453+ passed
```

---

## Quick Start: Design a Compressor Stage in 30 Minutes

This is the core workflow: compressor requirements in, solver-ready CFD case out.

### Step 1: Meanline design (2 minutes)

```bash
python -m astraturbo meanline \
  --pr 1.5 --mass-flow 20 --rpm 15000 \
  --r-hub 0.15 --r-tip 0.25 --radial-stations 5
```

This outputs:
- Velocity triangles and blade angles at mean radius
- **Auto-computed cl0** for profile generation (Lieblein correlation)
- **Radial blade angle distribution** (free-vortex, hub to tip)
- De Haller ratio and loading/flow coefficients

### Step 2: Generate blade profile (1 minute)

```bash
python -m astraturbo profile --camber naca65 --cl0 1.1 --thickness naca65 \
  --max-thickness 0.10 -o blade.csv
```

Use the cl0 value from the meanline output. This creates a NACA 65-series profile.

### Step 3: Generate mesh with boundary conditions (2 minutes)

```bash
# 2D mesh
python -m astraturbo mesh --profile blade.csv --pitch 0.05 \
  --with-bcs --format cgns -o mesh.cgns

# Or 3D mesh (stacked at multiple span stations)
python -m astraturbo mesh --profile blade.csv --pitch 0.05 \
  --3d --n-span 3 --span 0.04 --with-bcs --format cgns -o mesh_3d.cgns
```

The `--with-bcs` flag writes CGNS boundary conditions (inlet, outlet, blade wall, periodic) so the mesh is solver-ready.

### Step 4: Set up compressible CFD case (1 minute)

```bash
# Incompressible (low-speed cascades)
python -m astraturbo cfd --solver openfoam --velocity 100 -o cfd_case

# Compressible (real compressor conditions)
python -m astraturbo cfd --solver openfoam --compressible \
  --total-pressure 101325 --total-temperature 288.15 \
  -o cfd_case
```

The `--compressible` flag generates a `rhoSimpleFoam` case with:
- `thermophysicalProperties` (ideal gas, Sutherland viscosity)
- Total pressure/temperature inlet BCs
- Temperature field (`0/T`)

### Step 5: Run (or submit to HPC)

```bash
# Local
cd cfd_case && bash Allrun

# Or submit to SLURM cluster
python -m astraturbo hpc submit ./cfd_case --backend slurm
```

---

## Quick Start: CLI

```bash
# Generate a blade profile
python -m astraturbo profile --camber naca65 --thickness naca4digit -o blade.csv

# Generate a structured mesh with BCs
python -m astraturbo mesh --profile blade.csv --pitch 0.05 --with-bcs -o mesh.cgns

# Inspect the result
python -m astraturbo info mesh.cgns

# Set up an OpenFOAM case (incompressible)
python -m astraturbo cfd --solver openfoam --velocity 100 -o my_case

# Set up a compressible OpenFOAM case
python -m astraturbo cfd --solver openfoam --compressible \
  --total-pressure 150000 --total-temperature 350 -o comp_case
```

---

## Quick Start: GUI

```bash
python -m astraturbo gui
```

When the window opens:

1. The **2D Profile** tab shows a default NACA 65-1-10 airfoil.
2. Click **Compute > Meanline Design** — enter pressure ratio, mass flow, RPM, hub/tip radii, and radial stations. The result shows auto-computed cl0, stagger, and radial blade angle distribution.
3. Click **Compute > Compute Blade Geometry** to build the 3D blade. Any surface validation warnings (thin sections, self-intersection) are shown.
4. Click **Compute > Generate Multi-Block Mesh** for a 2D mesh, or **Generate 3D Mesh (Span Stacking)** for a full 3D mesh.
5. Click **File > Export > CGNS Mesh** to save. Boundary conditions are automatically included.
6. Click **Compute > CFD Case Setup > OpenFOAM** — tick "Compressible" for rhoSimpleFoam, set total pressure/temperature.

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

## Tutorial 2: Meanline Design with Radial Blade Angles

```python
from astraturbo.design.meanline import (
    meanline_compressor,
    meanline_to_blade_parameters,
    blade_angle_to_cl0,
)
import math

# Design a single-stage compressor
result = meanline_compressor(
    overall_pressure_ratio=1.5,
    mass_flow=20.0,
    rpm=15000,
    r_hub=0.15,
    r_tip=0.25,
    radial_stations=5,   # hub, 25%, mid, 75%, tip
)

# Overall performance
print(result.summary())

# Auto-compute cl0 for profile generation
params = meanline_to_blade_parameters(result)
stage = result.stages[0]
cl0 = blade_angle_to_cl0(
    stage.rotor_inlet_beta, stage.rotor_outlet_beta,
    params[0]["rotor_solidity"],
)
print(f"\nAuto-computed cl0: {cl0:.4f}")
print(f"Stagger: {params[0]['rotor_stagger_deg']:.1f} deg")

# Radial blade angle distribution (free vortex)
print("\nRadial distribution:")
for a in stage.radial_blade_angles:
    print(f"  r={a['r']:.4f} m  beta_in={math.degrees(a['beta_in']):+.1f}  "
          f"beta_out={math.degrees(a['beta_out']):+.1f} deg")
```

Or via CLI:

```bash
python -m astraturbo meanline --pr 1.5 --mass-flow 20 --rpm 15000 \
  --r-hub 0.15 --r-tip 0.25 --radial-stations 5
```

---

## Tutorial 3: Build a 3D Axial Compressor Blade

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
# Blade surface validation runs automatically — warnings logged for:
#   - Minimum thickness < 0.3mm
#   - Self-intersection (normal direction reversals)
row.compute(
    stagger_angles=np.deg2rad([30, 35, 40]),
    chord_lengths=np.array([0.04, 0.05, 0.06]),
)

print(f"Leading edge: {row.leading_edge.shape}")
print(f"Trailing edge: {row.trailing_edge.shape}")
print(f"3D profiles: {len(row.profiles_3d)} sections")
```

---

## Tutorial 4: Generate Mesh and Export CGNS with Boundary Conditions

```python
from astraturbo.mesh.multiblock import generate_blade_passage_mesh
from astraturbo.mesh import mesh_quality_report
from astraturbo.export.cgns_writer import write_cgns_structured

# Use the mid-span profile from above
profile_2d = profiles[1].as_array()

mesh = generate_blade_passage_mesh(
    profile=profile_2d, pitch=0.05,
    n_blade=40, n_ogrid=10, n_inlet=15, n_outlet=15, n_passage=20,
    ogrid_thickness=0.005, grading_ogrid=1.3,
)

print(f"Mesh: {mesh.n_blocks} blocks, {mesh.total_cells} cells")

# Check quality (warnings logged automatically for AR>100 or skewness>0.95)
report = mesh_quality_report(mesh.blocks[0].points)
print(f"Max aspect ratio: {report['aspect_ratio_max']:.1f}")
print(f"Max skewness: {report['skewness_max']:.3f}")

# Export CGNS with boundary conditions
block_arrays = [b.points for b in mesh.blocks]
block_names = [b.name for b in mesh.blocks]
patches = {b.name: b.patches for b in mesh.blocks if b.patches}

write_cgns_structured(
    "compressor_mesh.cgns", block_arrays, block_names,
    patches=patches,  # Writes inlet/outlet/blade/periodic BCs
)
```

### 3D mesh (span stacking)

```python
from astraturbo.mesh.multiblock import generate_blade_passage_mesh_3d

profiles_2d = [p.as_array() for p in profiles]
mesh_3d = generate_blade_passage_mesh_3d(
    profiles=profiles_2d,
    span_positions=[0.0, 0.025, 0.05],
    pitch=0.05,
    n_blade=30, n_ogrid=8, n_inlet=10, n_outlet=10, n_passage=15,
)
print(f"3D mesh: {mesh_3d.n_blocks} blocks, 3 span stations")
```

Or via CLI:

```bash
# 2D mesh with BCs
python -m astraturbo mesh --profile blade.csv --pitch 0.05 \
  --with-bcs --format cgns -o mesh.cgns

# 3D mesh with BCs
python -m astraturbo mesh --profile blade.csv --pitch 0.05 \
  --3d --n-span 3 --span 0.04 --with-bcs --format cgns -o mesh_3d.cgns
```

---

## Tutorial 5: Set Up Compressible CFD Case

```python
from astraturbo.cfd import create_openfoam_case

# Incompressible (simpleFoam)
case = create_openfoam_case(
    case_dir="compressor_cfd",
    solver="simpleFoam",
    turbulence_model="kOmegaSST",
    inlet_velocity=100.0,
)

# Compressible (rhoSimpleFoam) — for real compressor conditions
case = create_openfoam_case(
    case_dir="compressor_cfd_compressible",
    solver="rhoSimpleFoam",
    compressible=True,
    total_pressure=150000.0,     # Pa, inlet total pressure
    total_temperature=350.0,     # K, inlet total temperature
    inlet_velocity=100.0,
)
# Creates: thermophysicalProperties, 0/T, totalPressure inlet BC
```

Or via CLI:

```bash
# Incompressible
python -m astraturbo cfd --solver openfoam --velocity 100 -o cfd_case

# Compressible
python -m astraturbo cfd --solver openfoam --compressible \
  --total-pressure 150000 --total-temperature 350 -o comp_case
```

Then run:
```bash
cd comp_case && bash Allrun
```

---

## Tutorial 6: End-to-End Pipeline with DesignChain

The `DesignChain` runs the full pipeline automatically: meanline auto-computes cl0, stagger, and chord, then propagates them to profile, blade, mesh, export, and CFD stages.

```python
from astraturbo.foundation.design_chain import DesignChain

chain = DesignChain()
result = chain.set_parameters({
    "pressure_ratio": 1.5,
    "mass_flow": 20.0,
    "rpm": 15000.0,
    "cfd_output": "./my_cfd_case",       # generates CFD case
    "cfd_compressible": True,             # use rhoSimpleFoam
    "cfd_total_pressure": 101325.0,
    "cfd_total_temperature": 288.15,
})

# All stages run automatically:
# meanline → profile → blade → mesh → export → cfd
for stage in result.stages:
    status = "OK" if stage.success else f"FAIL: {stage.error}"
    print(f"  {stage.stage_name:12s}  {stage.elapsed_time:.3f}s  {status}")

# Access computed cl0
meanline_data = next(s.data for s in result.stages if s.stage_name == "meanline")
print(f"\nAuto-computed cl0: {meanline_data['cl0']:.4f}")
```

---

## Tutorial 7: Multi-Stage Rotor+Stator

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

## Tutorial 8: Read an Existing OpenFOAM Mesh

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

## Tutorial 9: Estimate y+ for Mesh Design

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

## Tutorial 10: Validate Against NASA Rotor 37

AstraTurbo includes reference data for NASA Rotor 37 (Reid & Moore, NASA TP-1138) — the standard turbomachinery validation case.

```python
import json
from pathlib import Path
from astraturbo.design.meanline import meanline_compressor, meanline_to_blade_parameters

# Load published data
ref = json.loads(Path("tests/test_validation/reference_data/nasa_rotor37.json").read_text())

# Run meanline at Rotor 37 conditions
dp = ref["design_point"]
geo = ref["geometry"]
result = meanline_compressor(
    overall_pressure_ratio=dp["total_pressure_ratio"],  # 2.106
    mass_flow=dp["mass_flow_kg_s"],                     # 20.19 kg/s
    rpm=dp["rpm"],                                       # 17188.7 RPM
    r_hub=geo["hub_radius_inlet_m"],                    # 0.178 m
    r_tip=geo["tip_radius_inlet_m"],                    # 0.252 m
    n_stages=1,
    radial_stations=5,
)

print(f"Published PR: {dp['total_pressure_ratio']}")
print(f"Meanline PR:  {result.overall_pressure_ratio:.3f}")
print(f"Published eta: {dp['adiabatic_efficiency']}")
print(f"Meanline eta:  {result.overall_efficiency:.3f} (polytropic)")
```

Run the full validation suite: `python -m pytest tests/test_validation/test_nasa_rotor37.py -v`

**Limitations**: Rotor 37 is transonic (Mach 1.48 at tip). Our meanline analysis captures bulk thermodynamics but cannot predict shock losses, tip clearance effects, or detailed radial profiles. These require CFD.

---

## Tutorial 11: Off-Design Analysis & Compressor Maps

Once you have a design-point solution, evaluate how the compressor behaves at different RPM and mass flow conditions.

### Off-design at a single operating point

```python
from astraturbo.design import meanline_compressor, off_design_compressor

# Design the compressor
design = meanline_compressor(
    overall_pressure_ratio=2.1, mass_flow=20.0,
    rpm=17189, r_hub=0.178, r_tip=0.252,
)

# Evaluate at reduced mass flow
od = off_design_compressor(design, mass_flow=16.0, rpm=17189)
print(f"PR = {od.overall_pr:.3f}")
print(f"Efficiency = {od.overall_efficiency:.4f}")
print(f"Stalled = {od.is_stalled}")

# Per-stage details
for s in od.stages:
    print(f"  Stage {s['stage']}: DF={s['DF']:.3f}, incidence={s['incidence_deg']:.1f} deg")
```

### Generate a compressor map

```python
from astraturbo.design import generate_compressor_map

cmap = generate_compressor_map(
    design,
    rpm_fractions=[0.7, 0.85, 1.0, 1.05],
    n_points=15,
)
print(cmap.summary())
# Prints speed lines with mass flow, PR, eta, stall/choke flags
# Plus surge line connecting stall points across speeds
```

### From the CLI

```bash
# Single off-design point
python -m astraturbo meanline --pr 2.1 --mass-flow 20 --rpm 17189 \
  --r-hub 0.178 --r-tip 0.252 --off-design

# Full compressor map
python -m astraturbo meanline --pr 2.1 --mass-flow 20 --rpm 17189 \
  --r-hub 0.178 --r-tip 0.252 --map

# Custom speed lines
python -m astraturbo meanline --pr 2.1 --mass-flow 20 --rpm 17189 \
  --r-hub 0.178 --r-tip 0.252 --map --rpm-fractions "0.7,0.85,1.0,1.05"
```

### From the GUI

1. **Compute > Meanline Design** — enter your requirements
2. A dialog asks "Generate Compressor Map" — check the box
3. The result includes speed line tables, surge points, and surge margin

---

## Module Reference

| Module | Import | Purpose |
|---|---|---|
| `astraturbo.camberline` | `from astraturbo.camberline import NACA65` | Camber line generators |
| `astraturbo.thickness` | `from astraturbo.thickness import NACA4Digit` | Thickness distributions |
| `astraturbo.profile` | `from astraturbo.profile import Superposition` | 2D profile construction |
| `astraturbo.blade` | `from astraturbo.blade import BladeRow` | 3D blade geometry |
| `astraturbo.design.meanline` | `from astraturbo.design.meanline import meanline_compressor` | Meanline analysis with radial output |
| `astraturbo.design.meanline` | `from astraturbo.design.meanline import blade_angle_to_cl0` | Lieblein cl0 correlation |
| `astraturbo.design.off_design` | `from astraturbo.design import off_design_compressor` | Off-design meanline analysis |
| `astraturbo.design.compressor_map` | `from astraturbo.design import generate_compressor_map` | Compressor map generation |
| `astraturbo.design.centrifugal` | `from astraturbo.design import centrifugal_compressor` | Centrifugal compressor design |
| `astraturbo.design.turbine` | `from astraturbo.design import meanline_turbine` | Axial turbine meanline design |
| `astraturbo.reports` | `from astraturbo.reports import generate_report` | HTML report generation |
| `astraturbo.nurbs` | `from astraturbo.nurbs import interpolate_3d` | NURBS utilities |
| `astraturbo.machine` | `from astraturbo.machine import TurboMachine` | Machine container |
| `astraturbo.mesh` | `from astraturbo.mesh import SCMMesher, OGridGenerator` | Mesh generation |
| `astraturbo.mesh.multiblock` | `from astraturbo.mesh.multiblock import generate_blade_passage_mesh` | Multi-block mesher (2D) |
| `astraturbo.mesh.multiblock` | `from astraturbo.mesh.multiblock import generate_blade_passage_mesh_3d` | Multi-block mesher (3D span stacking) |
| `astraturbo.mesh.multistage` | `from astraturbo.mesh.multistage import MultistageGenerator` | Multi-row stages |
| `astraturbo.mesh.smoothing` | `from astraturbo.mesh.smoothing import laplacian_smooth` | Mesh quality improvement |
| `astraturbo.mesh.tip_clearance` | `from astraturbo.mesh.tip_clearance import generate_tip_clearance_mesh` | Tip gap mesh |
| `astraturbo.solver` | `from astraturbo.solver.throughflow import ThroughflowSolver` | S2m throughflow solver |
| `astraturbo.export` | `from astraturbo.export import write_cgns_structured` | CGNS export (with BCs) |
| `astraturbo.export.cgns_writer` | `from astraturbo.export.cgns_writer import write_cgns_boundary_conditions` | CGNS BC writer |
| `astraturbo.export` | `from astraturbo.export import read_openfoam_points` | OpenFOAM import |
| `astraturbo.cfd` | `from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig` | CFD workflow (compressible + incompressible) |
| `astraturbo.cfd` | `from astraturbo.cfd import create_openfoam_case` | OpenFOAM case (rhoSimpleFoam/simpleFoam) |
| `astraturbo.fea` | `from astraturbo.fea import FEAWorkflow, get_material` | Structural analysis |
| `astraturbo.fea.material` | `mat.youngs_modulus_at(T)`, `mat.properties_at(T)` | Temperature-dependent properties |
| `astraturbo.optimization` | `from astraturbo.optimization import Optimizer` | Design optimization |
| `astraturbo.foundation` | `from astraturbo.foundation.design_chain import DesignChain` | Auto-propagating design pipeline |
| `astraturbo.database` | `from astraturbo.database import DesignDatabase` | Design database (SQLite) |
| `astraturbo.hpc` | `from astraturbo.hpc import HPCJobManager, HPCConfig` | HPC job management |
| `astraturbo.ai` | `from astraturbo.ai import create_assistant` | AI assistant |
| `astraturbo.ai.surrogate` | `from astraturbo.ai.surrogate import SurrogateTrainer` | Surrogate modeling (GPR/MLP) |

---

## CLI Command Reference

| Command | Description |
|---|---|
| `astraturbo gui` | Launch graphical interface |
| `astraturbo ai` | AI design assistant (interactive chat) |
| `astraturbo profile [options]` | Generate a 2D blade profile |
| `astraturbo mesh [options]` | Generate a mesh (`--3d` for 3D, `--with-bcs` for CGNS BCs) |
| `astraturbo meanline [options]` | Meanline design (`--off-design`, `--map`, `--radial-stations N`) |
| `astraturbo blade [options]` | Build 3D blade from hub-to-tip profiles (`-o mesh.cgns`) |
| `astraturbo pipeline [options]` | Full design pipeline (`--compressible`, `--cfd-output`) |
| `astraturbo centrifugal [options]` | Centrifugal compressor design (`--report report.html`) |
| `astraturbo turbine [options]` | Axial turbine meanline design (`--expansion-ratio`, `--inlet-temp`) |
| `astraturbo cfd [options]` | CFD case setup (`--compressible` for rhoSimpleFoam) |
| `astraturbo fea [options]` | FEA structural analysis |
| `astraturbo yplus [options]` | y+ / cell height calculator |
| `astraturbo info <file>` | Inspect a mesh/points/CSV file |
| `astraturbo formats` | List 30 supported file formats |
| `astraturbo optimize [options]` | Run blade optimization |
| `astraturbo multistage [options]` | Multi-row stage mesh |
| `astraturbo run <case>` | Execute CFD/FEA solver |
| `astraturbo throughflow [options]` | Run S2m throughflow solver |
| `astraturbo smooth [options]` | Apply Laplacian mesh smoothing |
| `astraturbo sweep [options]` | Run parametric sweep |
| `astraturbo database {list,save,export}` | Design database management |
| `astraturbo hpc submit <case>` | Submit job (local/SLURM/PBS/AWS) |
| `astraturbo hpc status <job-id>` | Check job status |
| `astraturbo hpc cancel <job-id>` | Cancel a running job |
| `astraturbo hpc download <job-id>` | Download job results |
| `astraturbo hpc setup-aws` | Provision AWS Batch infrastructure |
| `astraturbo hpc teardown-aws` | Delete AWS resources |
| `astraturbo --help` | Show all commands |

---

## Validation

AstraTurbo includes validation against published data:

| Test | Reference | What's validated |
|---|---|---|
| NACA 65-1-10 profile | NASA TN-3916 | Profile coordinates within 0.1% chord |
| Velocity triangles | Textbook examples | Euler equation, angle consistency |
| Thermodynamics | First principles | Isentropic relations, energy conservation |
| Mesh quality | Engineering bounds | Aspect ratio, skewness metrics |
| **NASA Rotor 37** | **NASA TP-1138** | **Overall PR, temperature ratio, radial trends, end-to-end pipeline** |
| **Off-design & maps** | **Physical consistency** | **Incidence, loss trends, stall detection, speed line ordering, surge margin** |
| **Temperature materials** | **MMPDS / ASM data** | **E(T), yield(T), k(T) interpolation for Inconel 718, CMSX-4, Ti-6Al-4V, +3 more** |

Run all validation tests: `python -m pytest tests/test_validation/ -v`
