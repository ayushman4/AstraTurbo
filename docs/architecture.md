# AstraTurbo Architecture

## Overview

AstraTurbo is structured as a layered platform where each module handles
one stage of the turbomachinery engineering pipeline.

```
┌─────────────────────────────────────────────────────────────┐
│              AI Assistant (Claude API / CLI fallback)         │
├─────────────────────────────────────────────────────────────┤
│                         GUI / CLI                           │
├─────────────────────────────────────────────────────────────┤
│                      Optimization (pymoo)                   │
├──────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│Design│   CFD    │   FEA    │  Export  │   Mesh   │  Blade   │
│      │          │          │          │          │          │
│Meanln│ OpenFOAM │ CalculiX │  CGNS    │   TFI    │ BladeRow │
│Vel.  │ Fluent   │ Abaqus   │ blockMsh │   SCM    │ Stacking │
│Tri.  │ CFX      │ Material │ Tecplot  │  O-Grid  │  NURBS   │
│OffDes│ SU2      │ CFD→FEA  │ VTK/30+  │ MultBlk  │Hub/Shrd │
├──────┴──────────┴──────────┴──────────┴──────────┴──────────┤
│                     NURBS Engine (geomdl)                    │
├─────────────────────────────────────────────────────────────┤
│           Foundation (Properties, Signals, Undo)             │
├─────────────────────────────────────────────────────────────┤
│              Base Classes (ATObject, Node, Drawable)          │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
foundation ← baseclass ← distribution
                       ← camberline ← profile
                       ← thickness  ←┘
                                      ← blade ← machine
              nurbs ←─────────────────┘
design (standalone — outputs feed into blade/)
                                      ← mesh ← export
                                               ← cfd ← cfd/workflow
                                               ← fea ← fea/workflow
                                      ← optimization
                                               ← gui/cli
ai (top layer — calls all modules via tool use)
```

## Modules

### ai/
Claude-powered AI assistant with 9 tools. Uses the `anthropic` SDK with tool use —
Claude calls AstraTurbo functions directly (meanline, profile, mesh, CFD, FEA, y+,
materials, formats, file inspect). Requires `ANTHROPIC_API_KEY`.
Accessible via: GUI (AI Assistant tab), CLI (`astraturbo ai`), Python (`create_assistant()`).

### design/
Velocity triangle calculations, meanline stage-by-stage analysis (axial and centrifugal),
off-design solver, and compressor map generation.
Input: pressure ratio, mass flow, RPM, radii.
Output: blade angles, loading coefficients, De Haller ratios, speed lines, surge margin,
impeller/diffuser geometry (centrifugal).
Connects to blade/ by auto-generating stagger, camber, and solidity parameters.

### cfd/
Unified `CFDWorkflow` class generates complete case files for:
- **OpenFOAM**: Allrun, controlDict, fvSchemes, fvSolution, BCs, MRF for rotors
- **Fluent**: Journal (.jou) with kw-SST, BCs, convergence monitors
- **CFX**: CCL definition with domain, turbulence, boundaries, solver control
- **SU2**: Config (.cfg) with RANS/SST setup

### fea/
Structural analysis integration:
- **material.py**: 6 materials (Inconel 718/625, Ti-6Al-4V, CMSX-4, Steel, Al)
- **mesh_export.py**: Extrude blade surface to hex solid, map CFD pressure to FEA
- **calculix.py**: Write CalculiX/Abaqus .inp with centrifugal + aero loads
- **workflow.py**: Full CFD-FEA coupled pipeline

### export/ (30 formats)
Native readers/writers: CGNS, OpenFOAM, PLOT3D, Tecplot, EnSight, UGRID, STL, HDF5.
Via meshio: VTK, Gmsh, Nastran, UNV, Fluent, SU2, XDMF, Exodus, Abaqus, +10 more.
Via cadquery (optional): STEP, IGES.

## Design Principles

1. **Property Descriptors** — automatic validation, change notification, undo/redo
2. **Observer Pattern** — tree nodes notify parents on change
3. **Factory Functions** — `create_camberline("naca65")`, `get_material("inconel_718")`
4. **Unified I/O** — `read_mesh()` / `write_mesh()` auto-detect format from extension
5. **Workflow Classes** — `CFDWorkflow`, `FEAWorkflow`, `AstraTurboAssistant` orchestrate pipelines
6. **AI-First Option** — natural language interface to the entire platform via Claude
7. **Cross-Platform** — every dependency is pip-installable on Win/Linux/Mac
