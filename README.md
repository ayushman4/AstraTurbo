# AstraTurbo

**Open-source integrated turbomachinery design and simulation platform.**

AstraTurbo covers the full turbomachinery engineering pipeline:

```
Requirements → Meanline Design → Blade Geometry → Mesh → CFD → FEA → Optimization
                              ↑ AI Assistant (Claude) can drive the entire pipeline ↑
```

AstraTurbo is an open-source turbomachinery platform. Built with Python 3.10+, cross-platform dependencies, and a modular architecture. Runs natively on **Windows, Linux, and macOS**.

---

## Installation

```bash
git clone https://github.com/ayushman4/AstraTurbo.git
cd AstraTurbo

# Core only (design + mesh + export, no GUI)
pip install -e .

# With GUI (adds PySide6, pyqtgraph, VTK)
pip install -e ".[gui]"

# With AI assistant (adds Claude API integration)
pip install -e ".[ai]"

# Everything (adds optimization, AI, dev tools)
pip install -e ".[all]"

# With AWS Batch support (adds boto3)
pip install -e ".[aws]"
```

Verify:

```bash
python -m astraturbo --version
# astraturbo 0.1.0

python -m pytest tests/ -q
# 641 passed
```

---

## Three Ways to Use AstraTurbo

AstraTurbo can be used through a **GUI**, the **command line**, or the **Python API**.

---

## 1. GUI (Graphical Interface)

### Launch

```bash
python -m astraturbo gui
```

### Window Layout

```
┌─────────────┬──────────────────────────────────────────────┬──────────────┐
│  Machine    │  2D Profile / 3D Blade / 3D Viewer / AI Chat │  Properties  │
│  Structure  │  (center, tabbed)                            │  (editable   │
│  (tree)     │                                              │   fields)    │
├─────────────┴──────────────────────────────────────────────┴──────────────┤
│  Mesh Panel (cell counts, grading, quality report)                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Workflow

**Step 1 — Design a 2D airfoil profile:**
1. In the **2D Profile** tab, select a **Camber** type (NACA 65, Circular Arc, Polynomial, etc.)
2. Select a **Thickness** type (NACA 4-Digit, NACA 65-Series, Joukowski, Elliptic)
3. The profile plot updates live

**Step 2 — Configure the 3D blade:**
1. Switch to the **3D Blade** tab
2. Set number of blades, angular velocity, stacking mode
3. To add more span profiles: **Edit > Add Profile to Row**

**Step 3 — Compute 3D blade geometry:**
- **Compute > Compute Blade Geometry** (or toolbar: "Compute Blade")

**Step 4 — Generate a mesh:**
1. In the **Mesh** panel (bottom), set axial cells, radial cells, grading
2. **Compute > Generate Multi-Block Mesh**
3. Quality report appears: total cells, aspect ratio, skewness

**Step 5 — Export:**
- **File > Export > CGNS Mesh** — for ParaView, CFX, Fluent
- **File > Export > OpenFOAM blockMeshDict** — for OpenFOAM
- **File > Export > VTK Mesh** — for generic visualization

**Step 6 — Import existing meshes:**
- **File > Import OpenFOAM Points** — loads and visualizes in the **3D Viewer** tab
- **File > Import Legacy XML Project** — legacy projects

**Step 7 — AI Assistant (optional):**
1. Click the **AI Assistant** tab
2. Type a natural language request, e.g. "Design a 5-stage compressor with PR=8"
3. The AI calls AstraTurbo tools automatically (API mode) or generates commands (CLI fallback)
4. Requires: `pip install anthropic` + `export ANTHROPIC_API_KEY=sk-ant-...`

### Screenshots

**AI Assistant — Meanline Design**: Ask Claude to design a compressor. It calls `meanline_compressor` automatically and returns a full engineering breakdown with overall performance:

![AI Design Request](docs/images/Figure1.png)

**AI Assistant — Stage Analysis**: Stage-by-stage velocity triangles, blade angles, and De Haller ratio warnings flagged automatically:

![Stage Analysis](docs/images/Figure2.png)

**AI Assistant — Engineering Judgment**: The AI identifies that 5 stages is too few for PR=8 (loading too aggressive), explains the root cause, and recommends corrective actions:

![Root Cause Analysis](docs/images/Figure3.png)

**AI Assistant — Next Steps**: Offers to re-run with 7 stages, generate blade profiles, create CFD mesh, or run structural analysis — all from the same conversation:

![Next Steps](docs/images/Figure4.png)

**Keyboard shortcuts:** Cmd+N (New), Cmd+O (Open), Cmd+S (Save), Cmd+Q (Quit)

---

## 2. Command Line (CLI)

```bash
python -m astraturbo --help
```

### Generate a 2D blade profile

```bash
python -m astraturbo profile --camber naca65 --thickness naca4digit -o blade.csv
python -m astraturbo profile --camber circular_arc --thickness elliptic --plot
```

### Generate a mesh from a profile

```bash
python -m astraturbo mesh --profile blade.csv --pitch 0.05 -o mesh.cgns
python -m astraturbo mesh --profile blade.csv --pitch 0.05 --format vtk -o mesh.vtk
```

### Inspect any file

```bash
python -m astraturbo info /path/to/points          # OpenFOAM
python -m astraturbo info mesh.cgns                 # CGNS
python -m astraturbo info blade.csv                 # CSV
```

### Set up a CFD case

```bash
python -m astraturbo cfd --solver openfoam --velocity 100 -o my_case
python -m astraturbo cfd --solver fluent --velocity 120 -o fluent_case
python -m astraturbo cfd --solver cfx --rotating --omega 1500 -o cfx_case
python -m astraturbo cfd --solver su2 -o my_su2_case
```

### AI Assistant

```bash
# Interactive chat (requires ANTHROPIC_API_KEY)
python -m astraturbo ai

# Single request
python -m astraturbo ai "Design a 5-stage compressor with PR=8, mass flow 25 kg/s"
```

### Meanline design

```bash
# Design-point analysis
python -m astraturbo meanline --pr 4.0 --mass-flow 20 --rpm 12000 --r-hub 0.15 --r-tip 0.30

# Off-design analysis at given conditions
python -m astraturbo meanline --pr 1.5 --mass-flow 20 --rpm 15000 --r-hub 0.15 --r-tip 0.25 --off-design

# Generate full compressor map (speed lines + surge line)
python -m astraturbo meanline --pr 2.1 --mass-flow 20 --rpm 17189 --r-hub 0.178 --r-tip 0.252 --map

# Custom RPM fractions for map
python -m astraturbo meanline --pr 2.1 --mass-flow 20 --rpm 17189 --r-hub 0.178 --r-tip 0.252 \
  --map --rpm-fractions "0.7,0.85,1.0,1.05"
```

### y+ calculator

```bash
python -m astraturbo yplus --velocity 100 --chord 0.1
python -m astraturbo yplus --velocity 100 --chord 0.1 --cell-height 0.00001
```

### Centrifugal compressor design

```bash
# eVTOL / drone / turbocharger centrifugal compressor
python -m astraturbo centrifugal --pr 3.0 --mass-flow 1.0 --rpm 60000

# With HTML report
python -m astraturbo centrifugal --pr 2.5 --mass-flow 0.5 --rpm 120000 --report report.html
```

### Axial turbine design

```bash
# HP turbine (Kaveri-class): ER=2.5, hot gas at 1500 K
python -m astraturbo turbine --expansion-ratio 2.5 --mass-flow 20 --rpm 17189 \
  --r-hub 0.25 --r-tip 0.35 --inlet-temp 1500

# LP turbine with 2 stages
python -m astraturbo turbine --expansion-ratio 3.0 --mass-flow 20 --rpm 12000 \
  --r-hub 0.30 --r-tip 0.45 --inlet-temp 1000 --n-stages 2

# With HTML report
python -m astraturbo turbine --expansion-ratio 2.5 --mass-flow 20 --rpm 17189 \
  --r-hub 0.25 --r-tip 0.35 --inlet-temp 1500 --report turbine_report.html

# Off-design analysis (same geometry, different operating point)
python -m astraturbo turbine --expansion-ratio 2.5 --mass-flow 20 --rpm 17189 \
  --r-hub 0.25 --r-tip 0.35 --inlet-temp 1500 --off-design

# Generate turbine performance map
python -m astraturbo turbine --expansion-ratio 2.5 --mass-flow 20 --rpm 17189 \
  --r-hub 0.25 --r-tip 0.35 --inlet-temp 1500 --map
```

### Engine cycle analysis

```bash
# Simple turbojet at sea level
python -m astraturbo engine-cycle --opr 8 --tit 1400 --mass-flow 20 --rpm 15000 \
  --r-hub 0.15 --r-tip 0.30

# Kaveri-class turbojet at altitude
python -m astraturbo engine-cycle --engine-type turbojet --opr 20 --tit 1700 \
  --mass-flow 20 --altitude 10000 --mach 0.8

# Turboshaft for helicopter
python -m astraturbo engine-cycle --engine-type turboshaft --opr 8 --tit 1400 \
  --mass-flow 10 --rpm 30000 --r-hub 0.05 --r-tip 0.10 --compressor-type centrifugal

# Twin-spool turbojet (Kaveri-class)
python -m astraturbo engine-cycle --opr 20 --tit 1700 --mass-flow 20 --rpm 10000 \
  --n-spools 2 --hp-pr 4.5 --hp-rpm 15000
```

### Electric motor sizing

```bash
# Size an electric motor for eVTOL / hybrid-electric propulsion
python -m astraturbo electric-motor --power 50000 --rpm 8000 --voltage 400
```

### Propeller design

```bash
# Design a propeller for UAV / eVTOL / general aviation
python -m astraturbo propeller --thrust 50 --n-blades 3 --diameter 0.5 --rpm 8000
```

### Rocket turbopump design

```bash
# Standalone pump (LOX or fuel side)
python -m astraturbo pump --head 500 --flow-rate 0.1 --rpm 30000 --fluid LOX

# Integrated turbopump (turbine-driven pump assembly)
python -m astraturbo turbopump --pump-head 500 --pump-flow 0.1 --fluid LOX \
  --turbine-temp 900 --turbine-pressure 5000000 --rpm 30000
```

### Cooling system analysis

```bash
# Turbine blade / combustor cooling analysis
python -m astraturbo cooling --t-gas 1700 --t-coolant 600 --cooling-type film
```

### Engine cycle with afterburner and nozzle

```bash
# Afterburning turbojet with convergent-divergent nozzle
python -m astraturbo engine-cycle --opr 8 --tit 1400 --mass-flow 20 --rpm 15000 \
  --afterburner --nozzle-type convergent_divergent
```

### Design reports

```bash
# Compressor report with blade profile + loading images
python -m astraturbo meanline --pr 2.1 --mass-flow 20 --rpm 17189 \
  --r-hub 0.178 --r-tip 0.252 --map --report design_report.html

# Engine cycle report with station P/T chart and T-s diagram
python -m astraturbo engine-cycle --opr 20 --tit 1700 --mass-flow 20 --rpm 15000 \
  --report engine_report.html

# CFD report with pressure/velocity fields and residual convergence
python -m astraturbo cfd --solver openfoam --velocity 150 -o cfd_case --report cfd_report.html

# Full pipeline report (meanline + profile + mesh + CFD)
python -m astraturbo pipeline --pr 2.0 --mass-flow 20 --rpm 15000 \
  --cfd-output ./cfd_case --report pipeline_report.html
```

### Military jet engine pipeline (Kaveri, F414, M88)

```bash
# Demo pipeline: 3 engines × 9 stages (cycle → compressor → turbine → profile → 3D blade → mesh → OpenFOAM → report)
python examples/pipeline/run_engines.py

# Production pipeline: converged CFD with 3360-cell O-grid meshes, k-ω SST, 11-image HTML reports
python examples/pipeline/run_production.py

# All 3 engines converge:
#   Kaveri GTX-35VS  — 96.8 kN thrust, 352 iterations
#   GE F414          — 101.3 kN thrust, 88 iterations
#   Safran M88       — 79.1 kN thrust, 1000 iterations
#
# Reports include: engine station chart, T-s diagram, velocity triangles,
# compressor map, blade profile, blade loading, mesh wireframe,
# CFD pressure field, CFD velocity field, residual convergence
```

### 3D blade building

```bash
# Build 3D blade with hub-to-tip variation
python -m astraturbo blade --r-hub 0.15 --r-tip 0.25 \
  --cl0-hub 0.8 --cl0-mid 1.0 --cl0-tip 1.2 \
  --stagger-hub 30 --stagger-mid 35 --stagger-tip 40 \
  -o blade_mesh.cgns
```

### Full design pipeline (one command)

```bash
# Run entire pipeline: meanline → profile → blade → mesh → export → CFD
python -m astraturbo pipeline --pr 1.5 --mass-flow 20 --rpm 15000
python -m astraturbo pipeline --pr 2.1 --mass-flow 20 --rpm 17189 \
  --compressible --cfd-output ./cfd_case

# With HTML report
python -m astraturbo pipeline --pr 2.0 --mass-flow 20 --rpm 15000 \
  --report pipeline_report.html
```

### FEA setup

```bash
python -m astraturbo fea --list-materials
python -m astraturbo fea --material inconel_718 --omega 1200 --surface blade.csv -o fea_case

# Temperature-dependent analysis (hot-section blade at 973K)
python -m astraturbo fea --material inconel_718 --temperature 973 \
  --omega 1200 --surface blade.csv -o fea_case
# Shows: E at 973K = 140 GPa (vs 200 GPa room), safety factor at temp vs room
```

### Other commands

```bash
python -m astraturbo formats                    # List 30 supported formats
python -m astraturbo optimize --profile blade.csv --generations 50
python -m astraturbo multistage --profiles r.csv s.csv --pitches 0.05 0.06 -o stage.cgns
python -m astraturbo run cfd_case --solver openfoam
python -m astraturbo smooth --input mesh.cgns --iterations 20 -o smooth.cgns
python -m astraturbo throughflow --pr 1.5 --mass-flow 20 --rpm 15000
python -m astraturbo sweep --parameter cl0 --start 0.3 --end 1.2 --steps 10
python -m astraturbo blade --r-hub 0.15 --r-tip 0.25 -o blade.cgns
python -m astraturbo pipeline --pr 1.5 --mass-flow 20 --rpm 15000
python -m astraturbo database list
python -m astraturbo database save --name "rotor_v1" --params '{"chord": 0.05}'
```

### HPC / Cloud job management

```bash
# Run locally (default)
python -m astraturbo hpc submit ./cfd_case --backend local --solver openfoam

# Run on SLURM cluster
python -m astraturbo hpc submit ./cfd_case --backend slurm \
  --host cluster.example.com --user engineer --nprocs 64

# Run on AWS Batch (one-time setup, then submit)
python -m astraturbo hpc setup-aws --region us-east-1 --platform EC2
python -m astraturbo hpc submit ./cfd_case --backend aws \
  --aws-s3-bucket astraturbo-batch-123456789012-us-east-1 \
  --aws-job-queue astraturbo-queue --solver openfoam --nprocs 8

# Monitor and retrieve
python -m astraturbo hpc status <job-id>
python -m astraturbo hpc download <job-id> --output-dir ./results
python -m astraturbo hpc cancel <job-id>

# Tear down AWS resources when done
python -m astraturbo hpc teardown-aws --region us-east-1
```

Supported backends: **Local** (subprocess), **SLURM** (SSH), **PBS/Torque** (SSH), **AWS Batch** (boto3).

### End-to-end (no GUI)

```bash
python -m astraturbo profile --camber naca65 --thickness naca4digit -o blade.csv
python -m astraturbo mesh --profile blade.csv --pitch 0.05 -o mesh.cgns
python -m astraturbo info mesh.cgns
python -m astraturbo cfd --solver openfoam --velocity 100 -o cfd_case
```

---

## 3. Python API

### Meanline design: requirements → blade angles

```python
from astraturbo.design import meanline_compressor, meanline_to_blade_parameters

# Input: top-level requirements
result = meanline_compressor(
    overall_pressure_ratio=4.0,
    mass_flow=20.0,        # kg/s
    rpm=12000,
    r_hub=0.15,            # m
    r_tip=0.30,            # m
)

print(result.summary())
# Meanline Analysis: 5 stages
#   Overall PR:   4.000
#   Stage 1: PR=1.35, phi=0.48, psi=0.38, R=0.50
#   Rotor beta: -52.1 → -38.4 deg
#   ...

# Convert to blade geometry parameters
blade_params = meanline_to_blade_parameters(result)
# [{stage: 1, rotor_stagger_deg: -45.2, rotor_camber_deg: 13.7, ...}, ...]
```

### Off-design analysis & compressor maps

```python
from astraturbo.design import (
    meanline_compressor, off_design_compressor, generate_compressor_map,
)

# Design the compressor
design = meanline_compressor(
    overall_pressure_ratio=2.1, mass_flow=20.0,
    rpm=17189, r_hub=0.178, r_tip=0.252,
)

# Off-design at reduced mass flow
od = off_design_compressor(design, mass_flow=16.0, rpm=17189)
print(f"PR={od.overall_pr:.3f}, eta={od.overall_efficiency:.4f}, stalled={od.is_stalled}")

# Generate full compressor map
cmap = generate_compressor_map(design, rpm_fractions=[0.7, 0.85, 1.0, 1.05])
print(cmap.summary())
# Speed lines with mass flow, PR, efficiency, stall/choke flags
# Surge line connecting stall points across speed lines
```

### Centrifugal compressor design

```python
from astraturbo.design import centrifugal_compressor

# Design a drone/eVTOL centrifugal compressor
result = centrifugal_compressor(
    pressure_ratio=3.0, mass_flow=0.5, rpm=80000,
    r1_tip=0.03, beta2_blade_deg=-30, n_blades=17,
)
print(result.summary())
# PR, efficiency, power, tip speed, impeller + diffuser geometry

# Generate an HTML report
from astraturbo.reports import generate_report, ReportConfig
generate_report(
    config=ReportConfig(title="Drone Compressor", output_path="report.html"),
    centrifugal_result=result,
)
```

### Axial turbine design

```python
from astraturbo.design import meanline_turbine, meanline_to_turbine_blade_parameters

# Design an HP turbine (hot gas from combustor)
result = meanline_turbine(
    overall_expansion_ratio=2.5,  # P_in / P_out
    mass_flow=20.0,               # kg/s
    rpm=17189,
    r_hub=0.25,                   # m
    r_tip=0.35,                   # m
    T_inlet=1500.0,               # K (turbine inlet temperature)
)

print(result.summary())
# Turbine Meanline Analysis: 1 stage(s)
#   Overall ER:   2.500
#   Overall TR:   0.7901
#   Total work:   316450 J/kg
#   Stage 1: ER=2.500, phi=0.835, psi=1.085, R=0.500
#   Zweifel=0.813, Nozzle M=1.021

# Get NGV and rotor blade parameters
params = meanline_to_turbine_blade_parameters(result)
# [{stage: 1, ngv_stagger_deg: 25.7, rotor_stagger_deg: -24.2, zweifel: 0.813, ...}]

# Generate HTML report
from astraturbo.reports import generate_report, ReportConfig
generate_report(
    config=ReportConfig(title="HP Turbine", output_path="turbine_report.html"),
    turbine_result=result,
)
```

### Engine cycle (turbojet / turboshaft)

```python
from astraturbo.design import engine_cycle

# Turbojet at sea level
result = engine_cycle(
    engine_type="turbojet",
    overall_pressure_ratio=20.0,
    turbine_inlet_temp=1700.0,
    mass_flow=20.0,
    rpm=15000, r_hub=0.15, r_tip=0.30,
    altitude=0.0, mach_flight=0.0,
)
print(result.summary())
print(f"Thrust: {result.net_thrust/1000:.1f} kN, SFC: {result.specific_fuel_consumption*3600:.3f} kg/(N·h)")

# Turboshaft
shaft = engine_cycle(
    engine_type="turboshaft",
    overall_pressure_ratio=8.0,
    turbine_inlet_temp=1400.0,
    mass_flow=10.0,
    rpm=30000, r_hub=0.05, r_tip=0.10,
    compressor_type="centrifugal",
)
print(f"Shaft power: {shaft.shaft_power/1000:.1f} kW")
```

### Electric motor sizing

```python
from astraturbo.design import electric_motor

result = electric_motor(
    shaft_power=50000,   # W
    rpm=8000,
    voltage=400,         # V
)
print(result.summary())
# Motor type, torque, current, efficiency, weight estimate
```

### Propeller design

```python
from astraturbo.design import propeller_design

result = propeller_design(
    thrust_required=50.0,  # N
    n_blades=3,
    diameter=0.5,          # m
    rpm=8000,
)
print(result.summary())
# Thrust, power, FM (hover), advance ratio, CT/CP, tip Mach
```

### Pump design (rocket turbopumps)

```python
from astraturbo.design import centrifugal_pump, turbopump

# Standalone pump
pump_result = centrifugal_pump(
    head=500.0,          # m
    flow_rate=0.1,       # m³/s
    rpm=30000,
    fluid_name="LOX",
)
print(pump_result.summary())

# Integrated turbopump
tp = turbopump(
    pump_head=500.0,
    pump_flow_rate=0.1,
    fluid_density=1141.0,
    fluid_name="LOX",
    turbine_inlet_temp=900.0,
    turbine_inlet_pressure=5e6,
    rpm=30000,
)
print(tp.summary())
```

### Cooling system analysis

```python
from astraturbo.design import cooling_flow

result = cooling_flow(
    T_gas=1700.0,        # K (hot gas temperature)
    T_coolant=600.0,     # K (coolant temperature)
    cooling_type="film",
)
print(result.summary())
# Per-row effectiveness, coolant fraction, total coolant mass flow
```

### Generate a profile

```python
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA4Digit
from astraturbo.profile import Superposition

profile = Superposition(NACA65(cl0=1.0), NACA4Digit(max_thickness=0.10))
coords = profile.as_array()  # (399, 2) array
```

### Full pipeline: profile → 3D blade → mesh → CGNS

```python
import numpy as np
from astraturbo.camberline import NACA65
from astraturbo.thickness import NACA65Series
from astraturbo.profile import Superposition
from astraturbo.blade import BladeRow
from astraturbo.mesh.multiblock import generate_blade_passage_mesh

profiles = [
    Superposition(NACA65(cl0=0.8), NACA65Series(max_thickness=0.08)),
    Superposition(NACA65(cl0=1.0), NACA65Series(max_thickness=0.10)),
    Superposition(NACA65(cl0=1.2), NACA65Series(max_thickness=0.12)),
]

row = BladeRow(
    hub_points=np.array([[0.0, 0.10], [0.10, 0.10]]),
    shroud_points=np.array([[0.0, 0.20], [0.10, 0.20]]),
)
for p in profiles:
    row.add_profile(p)
row.compute(
    stagger_angles=np.deg2rad([30, 35, 40]),
    chord_lengths=np.array([0.04, 0.05, 0.06]),
)

mesh = generate_blade_passage_mesh(
    profile=profiles[1].as_array(), pitch=0.05,
    n_blade=40, n_ogrid=10, n_inlet=15, n_outlet=15, n_passage=20,
)
mesh.export_cgns("compressor.cgns")
```

### CFD workflow (OpenFOAM, Fluent, CFX, SU2)

```python
from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig

# OpenFOAM with rotating frame
wf = CFDWorkflow(CFDWorkflowConfig(
    solver="openfoam",
    inlet_velocity=100.0,
    turbulence_model="kOmegaSST",
    is_rotating=True,
    omega=1200.0,
    n_procs=4,
))
wf.set_mesh("mesh.cgns")
wf.setup_case("cfd_case/")
# Creates: Allrun, blockMeshDict or cgnsToFoam, MRFProperties, BCs

# ANSYS Fluent journal
wf = CFDWorkflow(CFDWorkflowConfig(solver="fluent", inlet_velocity=80))
wf.set_mesh("blade.msh")
wf.setup_case("fluent_case/")
# Creates: run.jou with k-omega SST, BCs, iteration control

# ANSYS CFX definition
wf = CFDWorkflow(CFDWorkflowConfig(solver="cfx", is_rotating=True, omega=1500))
wf.setup_case("cfx_case/")
# Creates: setup.ccl with domain, turbulence, boundaries, solver control

# SU2
wf = CFDWorkflow(CFDWorkflowConfig(solver="su2"))
wf.setup_case("su2_case/")
# Creates: astraturbo.cfg, run_su2.sh
```

### FEA structural analysis

```python
from astraturbo.fea import (
    FEAWorkflow, FEAWorkflowConfig,
    get_material, list_materials,
)

# See available materials
print(list_materials())
# ['al_7075', 'cmsx_4', 'inconel_625', 'inconel_718', 'steel_17_4ph', 'ti_6al_4v']

# Set up structural analysis
fea = FEAWorkflow(FEAWorkflowConfig(
    material=get_material("inconel_718"),
    omega=1200.0,           # Centrifugal load
    blade_thickness=0.002,
    analysis_type="static", # Or "frequency" for modal analysis
))
fea.set_blade_surface(surface_points, ni, nj)
fea.set_cfd_pressure(cfd_points, cfd_pressure)  # Map CFD loads to FEA
fea.setup("fea_case/")
# Creates: blade.inp (CalculiX/Abaqus format) with:
#   - Solid hex mesh extruded from blade surface
#   - Inconel 718 material properties
#   - Centrifugal load (CENTRIF)
#   - CFD pressure mapped to surface
#   - Fixed root boundary condition
#   - Stress/displacement output requests

# Quick analytical stress estimate (no solver needed)
estimate = fea.estimate_stress_analytical()
print(f"Centrifugal stress: {estimate['centrifugal_stress_MPa']:.1f} MPa")
print(f"Safety factor: {estimate['safety_factor']:.2f}")
```

### Multi-stage rotor + stator

```python
from astraturbo.mesh.multistage import MultistageGenerator, RowMeshConfig

gen = MultistageGenerator()
gen.add_row("rotor", RowMeshConfig(profile=rotor_profile, pitch=0.05, is_rotor=True))
gen.add_row("stator", RowMeshConfig(profile=stator_profile, pitch=0.06))
result = gen.generate()
result.export_cgns("stage.cgns")
```

### Read any mesh format

```python
from astraturbo.export import read_mesh, write_mesh, read_openfoam_points

# Unified API — auto-detects format
data = read_mesh("mesh.vtk")
data = read_mesh("grid.cgns")
data = read_mesh("case.plt")       # Tecplot
data = read_mesh("mesh.ugrid")     # NASA UGRID

# Write to any format
write_mesh("output.vtu", points, cells)

# OpenFOAM points with validation
points = read_openfoam_points("/path/to/points")
```

### AI Assistant (natural language → AstraTurbo)

```python
from astraturbo.ai import create_assistant

# Requires ANTHROPIC_API_KEY environment variable
assistant = create_assistant()

# Single request — AI calls tools automatically
response = assistant.chat(
    "Design a 5-stage axial compressor with PR=8, mass flow 25 kg/s at 15000 RPM. "
    "Generate NACA 65 profiles and set up an OpenFOAM case."
)
print(response)

# Multi-turn conversation
response2 = assistant.chat("Now check what y+ I need for the first stage at Mach 0.6")
print(response2)

# Reset conversation
assistant.reset()
```

Claude calls 30 AstraTurbo tools directly —
meanline design (axial compressor + centrifugal + turbine), engine cycle analysis
(turbojet/turboshaft), off-design analysis,
compressor maps, profile generation, 3D blade building, mesh generation and export,
CFD setup, solver execution, FEA setup, y+ calculator, full design pipeline, design
database, HTML report generation (with CFD field plots, mesh images, blade profiles),
file inspection, material database,
electric motor sizing, propeller design, turbopump analysis, rocket pump design,
cooling system analysis.

Setup:
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

---

## Architecture

```
astraturbo/
├── ai/              Claude-powered AI assistant (30 tools, NL interface)
├── design/          Velocity triangles, meanline (axial compressor + centrifugal + turbine), engine cycle, off-design, compressor maps, electric motor, propeller, pump, turbopump, cooling
├── foundation/      Property system, signals, undo/redo, serialization
├── baseclass/       ATObject, Node tree, Drawable mixin
├── camberline/      8 camber line types
├── thickness/       4 thickness distributions
├── distribution/    Point sampling (Chebyshev, Linear)
├── profile/         2D airfoil construction (superposition)
├── blade/           3D blade geometry (stacking, NURBS lofting)
├── nurbs/           NURBS curves & surfaces (via geomdl)
├── machine/         TurboMachine container, project management
├── mesh/            Mesh generation:
│   ├── transfinite    TFI with grading
│   ├── scm_mesher     S2m meridional plane mesh
│   ├── s1_mesher      Blade-to-blade mesh
│   ├── ogrid/         O10H topology O-grid around blades
│   ├── polyline       Polyline/Arc edge geometry
│   ├── grading        Edge grading, boundary layer clustering
│   ├── vertex_extraction  Block topology from profiles
│   ├── multiblock     Multi-block structured mesher (GridZ replacement)
│   ├── multistage     Rotor+stator multi-row orchestration
│   ├── tip_clearance  Tip clearance mesh generation
│   ├── smoothing      Laplacian + orthogonality smoothing
│   └── quality        Aspect ratio, skewness, y+ estimation
├── export/          30 formats: CGNS, OpenFOAM, Tecplot, VTK, Fluent, etc.
├── cfd/             4 solvers: OpenFOAM, Fluent, CFX, SU2 + post-processing (field reader, residual parser)
├── fea/             Structural analysis: CalculiX/Abaqus
│   ├── material       32 turbomachinery materials database
│   ├── calculix       Input file generation
│   ├── mesh_export    Surface-to-solid mesh, CFD pressure mapping
│   └── workflow       Coupled CFD-FEA pipeline
├── optimization/    pymoo-based multi-objective + multi-fidelity optimization
├── solver/          Throughflow (S2m) solver with loss models
├── database/        SQLite design database (save/search/compare/export)
├── reports/         HTML report generator with 15 matplotlib visualizations (engine stations, T-s diagram, velocity triangles, compressor map, blade profile, blade loading, mesh, CFD pressure/velocity fields, residual convergence)
├── hpc/             HPC backends: Local, SLURM, PBS, AWS Batch + auto-provisioner
├── gui/             PySide6 GUI with 3D viewer + AI chat panel
└── cli/             30+ commands (profile, mesh, blade, pipeline, meanline, cfd, fea, hpc, electric-motor, propeller, pump, turbopump, cooling, ...)
```

### Design pipeline

```
┌────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Meanline  │───▶│ Geometry │───▶│   Mesh   │───▶│   CFD    │───▶│   FEA    │───▶│ Optimize │
│  Design    │    │          │    │          │    │          │    │          │    │          │
│            │    │ Stacking │    │  O-Grid  │    │ OpenFOAM │    │ CalculiX │    │  pymoo   │
│ Vel. tri.  │    │  NURBS   │    │   TFI    │    │ Fluent   │    │ Abaqus   │    │  NSGA-II │
│ Euler eqn  │    │Hub/Shroud│    │Multi-blk │    │ CFX      │    │Materials │    │  DOE     │
│ Off-design │    │          │    │  CGNS    │    │ SU2      │    │Stress/   │    │          │
│ Comp. maps │    │          │    │          │    │          │    │ modal    │    │          │
└────────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
      │                                                                                │
      └────────────────────────── Optimization Loop ───────────────────────────────────┘
```

---

## Supported File Formats (30)

### Geometry & CAD

| Format | Extensions | Read | Write | Method |
|---|---|---|---|---|
| STEP | .step, .stp | - | Yes | cadquery (optional) |
| IGES | .iges, .igs | - | Yes | cadquery (optional) |
| STL | .stl | Yes | Yes | Native + meshio |
| OBJ | .obj | Yes | Yes | meshio |
| PLY | .ply | Yes | Yes | meshio |

### Mesh Formats

| Format | Extensions | Read | Write | Method |
|---|---|---|---|---|
| CGNS | .cgns | Yes | Yes | Native (h5py) |
| OpenFOAM points | points | Yes | - | Native |
| OpenFOAM blockMeshDict | blockMeshDict | - | Yes | Native |
| PLOT3D | .xyz, .p3d, .q | Yes | Yes | Native |
| Tecplot | .plt, .dat, .tec | Yes | Yes | Native |
| Gmsh | .msh | Yes | Yes | meshio |
| UNV (I-DEAS) | .unv | Yes | Yes | meshio |
| Nastran | .nas, .bdf | Yes | Yes | meshio |
| ANSYS Fluent | .cas, .msh | Yes | Yes | meshio |
| SU2 | .su2 | Yes | Yes | meshio |
| UGRID (NASA) | .ugrid | Yes | - | Native |
| Abaqus / CalculiX | .inp | Yes | Yes | meshio + Native |

### Visualization & Solver I/O

| Format | Extensions | Read | Write | Method |
|---|---|---|---|---|
| VTK / VTU / PVTU | .vtk, .vtu | Yes | Yes | meshio |
| EnSight Gold | .case | Yes | - | Native + meshio |
| XDMF + HDF5 | .xdmf | Yes | Yes | meshio |
| HDF5 (generic) | .h5, .hdf5 | Yes | - | Native (h5py) |
| Exodus II | .exo | Yes | Yes | meshio |

### Unified API

```python
from astraturbo.export import read_mesh, write_mesh
data = read_mesh("any_file.vtk")  # Auto-detect format
write_mesh("output.su2", points, cells)
```

---

## Material Database

32 aerospace-grade materials across 7 categories. **6 key alloys include temperature-dependent
property tables** (E, yield, thermal conductivity vs temperature) for hot-section analysis —
critical for turbine blade and combustor design.

### Nickel Superalloys (12) — Hot section

| Material | Density | E (GPa) | Yield (MPa) | Max Temp (K) | Use Case |
|---|---|---|---|---|---|
| Inconel 718 | 8190 | 200 | 1035 | 973 | Compressor disks, LP turbine |
| Inconel 625 | 8440 | 205 | 758 | 1073 | Combustor, exhaust |
| Inconel 713C | 7910 | 200 | 740 | 1143 | Small engine turbine blades |
| Rene 41 | 8250 | 219 | 760 | 1143 | Turbine components |
| Rene 80 | 8160 | 210 | 690 | 1255 | Cast turbine blades |
| Rene N5 | 8630 | 131 | 960 | 1383 | Single crystal (GE) |
| Hastelloy X | 8220 | 205 | 360 | 1473 | Combustors, afterburners |
| Waspaloy | 8190 | 213 | 795 | 1003 | Disks, shafts |
| Udimet 720 | 8080 | 222 | 1000 | 1023 | HP compressor disks |
| CMSX-4 | 8700 | 130 | 950 | 1373 | 1st gen single crystal |
| PWA 1484 | 8950 | 128 | 1000 | 1393 | 2nd gen single crystal (P&W) |
| MAR-M-247 | 8540 | 200 | 830 | 1253 | Cast turbine blades/vanes |

### Titanium Alloys (5) — Fan & compressor

| Material | Density | E (GPa) | Yield (MPa) | Max Temp (K) | Use Case |
|---|---|---|---|---|---|
| Ti-6Al-4V | 4430 | 114 | 880 | 673 | Fan, LP compressor |
| Ti-6-2-4-2 | 4540 | 120 | 990 | 813 | Compressor blades, disks |
| Ti-5553 | 4650 | 110 | 1200 | 623 | Disks, landing gear |
| IMI 834 | 4550 | 120 | 1000 | 873 | Compressor blades (Rolls-Royce) |
| Ti-6-2-4-6 | 4650 | 114 | 1100 | 723 | High-strength disks |

### Steels (5) — Shafts & structure

| Material | Density | E (GPa) | Yield (MPa) | Max Temp (K) | Use Case |
|---|---|---|---|---|---|
| 17-4PH | 7780 | 197 | 1170 | 623 | Structural components |
| 15-5PH | 7800 | 196 | 1000 | 623 | Aerospace components |
| AISI 4340 | 7850 | 205 | 1210 | 673 | Shafts, gears |
| Maraging 300 | 8000 | 190 | 2000 | 723 | Shafts, critical fasteners |
| Incoloy 909 | 8310 | 160 | 1000 | 923 | Low-CTE casings, rings |

### Aluminum (2), CMC/Ceramics (3), Coatings (2), Cobalt/Exotic (3)

| Material | Category | Density | Yield (MPa) | Max Temp (K) | Use Case |
|---|---|---|---|---|---|
| Al 7075-T6 | aluminum | 2810 | 503 | 473 | Structural, nacelle |
| Al 2024-T3 | aluminum | 2780 | 345 | 473 | Airframe, inlet |
| SiC/SiC CMC | cmc | 2350 | 300 | 1588 | Turbine shrouds (GE LEAP) |
| Oxide/Oxide CMC | cmc | 2800 | 170 | 1473 | Combustor liners |
| Si3N4 | cmc | 3200 | 700 | 1623 | Bearings, turbocharger |
| YSZ TBC | coating | 5600 | 50 | 1473 | Thermal insulation |
| MCrAlY | coating | 7300 | 350 | 1373 | Bond coat (under TBC) |
| Haynes 188 | cobalt | 8980 | 455 | 1363 | Combustor, transition ducts |
| Haynes 25/L-605 | cobalt | 9130 | 475 | 1253 | Turbine vanes, afterburner |
| C-103 Niobium | exotic | 8860 | 350 | 1643 | Rocket nozzles, hypersonics |

```python
from astraturbo.fea import get_material, list_materials
mat = get_material("inconel_718")
print(mat.to_calculix_format())  # Ready for FEA input

# Temperature-dependent properties for hot-section analysis
props = mat.properties_at(973)  # At 973 K (max service temp)
print(f"E at 973K: {props['youngs_modulus_GPa']:.1f} GPa (room: 200 GPa)")
print(f"Yield at 973K: {props['yield_strength_MPa']:.0f} MPa (room: 1035 MPa)")
# E at 973K: 140.0 GPa, Yield at 973K: 580 MPa — 30-44% reduction!
```

---

## Dependencies

| Dependency | Purpose | Required? |
|---|---|---|
| numpy, scipy | Numerics | Yes |
| geomdl | NURBS curves/surfaces | Yes |
| h5py | CGNS + HDF5 read/write | Yes |
| meshio | 20+ mesh formats | Yes |
| pyyaml | Project file format | Yes |
| blinker | Signal/event system | Yes |
| PySide6 | GUI framework | Optional (`[gui]`) |
| pyqtgraph | 2D/3D plotting in GUI | Optional (`[gui]`) |
| vtk | 3D visualization | Optional (`[gui]`) |
| pymoo | Multi-objective optimization | Optional (`[optimization]`) |
| anthropic | Claude AI assistant | Optional (`[ai]`) |
| boto3 | AWS Batch HPC backend | Optional (`[aws]`) |
| cadquery | STEP/IGES CAD export | Optional (`[cad]`) |
| matplotlib | Report plots & CLI profile visualization | Yes |

All cross-platform (Windows, Linux, macOS).

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
# 641 tests pass (unit, integration, validation, GUI, CLI)
```

### Test coverage

- **Unit tests**: Foundation, camberline, thickness, profile, blade, NURBS, mesh, export, design, FEA, CFD, electric motor, propeller, pump, turbopump, cooling, report plots
- **Integration tests**: CLI commands (38 tests), GUI components (29 tests), AI tools (9 tests), report image embedding (7 tests), CFD field plots, end-to-end pipeline
- **Validation tests**: Velocity triangles, meanline thermodynamics, NACA 65 profiles, mesh quality bounds, off-design compressor maps, NASA Rotor 37
- **Security tests**: XXE prevention, deserialization whitelisting, command injection prevention

### Security

- SSH commands use `shlex.quote()` — no shell injection
- SQL queries use parameterized `?` placeholders — no SQL injection
- XML parser rejects `DOCTYPE`/`ENTITY` — no XXE attacks
- YAML deserialization whitelists `astraturbo.*` modules only
- AWS credentials validated at init (STS `get-caller-identity`)
- No `eval()`, `exec()`, or `pickle.load()` of untrusted data

---

## License

Apache-2.0
