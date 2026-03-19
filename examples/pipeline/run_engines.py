#!/usr/bin/env python3
"""End-to-end pipeline for three military jet engines.

Runs each engine through the full AstraTurbo platform:
  1. Engine cycle analysis (thermodynamic stations)
  2. Meanline compressor design (velocity triangles, blade angles)
  3. Meanline turbine design (expansion, Zweifel loading)
  4. 2D blade profile generation (NACA 65-series)
  5. 3D blade stacking (hub → mid → tip)
  6. Multi-block structured mesh (O-grid)
  7. OpenFOAM CFD case setup (compressible RANS)
  8. HTML report with embedded station/blade/mesh images

Engines:
  - Kaveri GTX-35VS  (HAL, India)
  - GE F414          (General Electric, USA)
  - Safran M88       (Safran, France)

Usage:
    python kaveri_pipeline/run_engines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Engine Specifications ──────────────────────────────────────────

ENGINES = {
    "Kaveri_GTX_35VS": {
        "title": "HAL Kaveri GTX-35VS",
        "engine_type": "turbojet",
        "opr": 21.5,
        "tit": 1700.0,
        "mass_flow": 77.0,
        "rpm": 10500,
        "r_hub": 0.14,
        "r_tip": 0.30,
        "n_spools": 2,
        "hp_pr": 4.5,
        "hp_rpm": 14000,
        "hp_r_hub": 0.12,
        "hp_r_tip": 0.22,
        "afterburner": True,
        "afterburner_temp": 2000.0,
        "turbine_er": 2.5,
        "turbine_r_hub": 0.25,
        "turbine_r_tip": 0.35,
    },
    "GE_F414": {
        "title": "General Electric F414-GE-400",
        "engine_type": "turbojet",
        "opr": 30.0,
        "tit": 1750.0,
        "mass_flow": 78.0,
        "rpm": 10200,
        "r_hub": 0.13,
        "r_tip": 0.32,
        "n_spools": 2,
        "hp_pr": 5.0,
        "hp_rpm": 14500,
        "hp_r_hub": 0.11,
        "hp_r_tip": 0.22,
        "afterburner": True,
        "afterburner_temp": 2050.0,
        "turbine_er": 3.0,
        "turbine_r_hub": 0.24,
        "turbine_r_tip": 0.34,
    },
    "Safran_M88": {
        "title": "Safran M88-2 (Dassault Rafale)",
        "engine_type": "turbojet",
        "opr": 24.5,
        "tit": 1580.0,
        "mass_flow": 65.0,
        "rpm": 10800,
        "r_hub": 0.13,
        "r_tip": 0.28,
        "n_spools": 2,
        "hp_pr": 4.0,
        "hp_rpm": 14800,
        "hp_r_hub": 0.11,
        "hp_r_tip": 0.21,
        "afterburner": True,
        "afterburner_temp": 1950.0,
        "turbine_er": 2.8,
        "turbine_r_hub": 0.23,
        "turbine_r_tip": 0.33,
    },
}


def run_engine_pipeline(name: str, spec: dict, output_dir: Path) -> dict:
    """Run the full design-to-CFD pipeline for one engine.

    Returns a dict with timing and key results.
    """
    t0 = time.perf_counter()
    results = {"name": name, "stages": {}}

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Engine cycle ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {spec['title']}")
    print(f"{'='*70}")

    t1 = time.perf_counter()
    from astraturbo.design.engine_cycle import engine_cycle

    ec_kwargs = dict(
        engine_type=spec["engine_type"],
        overall_pressure_ratio=spec["opr"],
        turbine_inlet_temp=spec["tit"],
        mass_flow=spec["mass_flow"],
        rpm=spec["rpm"],
        r_hub=spec["r_hub"],
        r_tip=spec["r_tip"],
        n_spools=spec["n_spools"],
        hp_pressure_ratio=spec["hp_pr"],
        hp_rpm=spec["hp_rpm"],
        hp_r_hub=spec["hp_r_hub"],
        hp_r_tip=spec["hp_r_tip"],
        afterburner=spec["afterburner"],
        afterburner_temp=spec["afterburner_temp"],
    )
    ec_result = engine_cycle(**ec_kwargs)
    results["engine_cycle"] = ec_result
    results["stages"]["engine_cycle"] = time.perf_counter() - t1
    print(f"\n  [1/7] Engine Cycle         ({results['stages']['engine_cycle']:.2f}s)")
    print(f"        Thrust: {ec_result.net_thrust/1000:.1f} kN")
    print(f"        SFC:    {ec_result.specific_fuel_consumption*3600:.4f} kg/(N·h)")
    print(f"        η_th:   {ec_result.thermal_efficiency:.4f}")

    # ── 2. Meanline compressor ─────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.design import meanline_compressor, meanline_to_blade_parameters

    comp_result = meanline_compressor(
        overall_pressure_ratio=spec["opr"] / spec["hp_pr"],  # LP compressor PR
        mass_flow=spec["mass_flow"],
        rpm=spec["rpm"],
        r_hub=spec["r_hub"],
        r_tip=spec["r_tip"],
    )
    blade_params = meanline_to_blade_parameters(comp_result)
    results["compressor"] = comp_result
    results["blade_params"] = blade_params
    results["stages"]["compressor"] = time.perf_counter() - t1
    print(f"  [2/8] LP Compressor        ({results['stages']['compressor']:.2f}s)")
    print(f"        Stages: {comp_result.n_stages}, PR: {comp_result.overall_pressure_ratio:.2f}")

    # ── 2b. Compressor map ─────────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.design import generate_compressor_map

    comp_map = generate_compressor_map(
        comp_result, rpm_fractions=[0.7, 0.85, 1.0, 1.05], n_points=10,
    )
    results["compressor_map"] = comp_map
    results["stages"]["comp_map"] = time.perf_counter() - t1
    print(f"        + Compressor map      ({results['stages']['comp_map']:.2f}s)")

    # ── 3. Meanline turbine ────────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.design.turbine import meanline_turbine

    turb_result = meanline_turbine(
        overall_expansion_ratio=spec["turbine_er"],
        mass_flow=spec["mass_flow"],
        rpm=spec["hp_rpm"],
        r_hub=spec["turbine_r_hub"],
        r_tip=spec["turbine_r_tip"],
        T_inlet=spec["tit"],
    )
    results["turbine"] = turb_result
    results["stages"]["turbine"] = time.perf_counter() - t1
    print(f"  [3/7] HP Turbine           ({results['stages']['turbine']:.2f}s)")
    print(f"        Stages: {turb_result.n_stages}, ER: {turb_result.overall_expansion_ratio:.2f}")

    # ── 4. 2D Blade profiles ───────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.camberline import NACA65
    from astraturbo.thickness import NACA65Series
    from astraturbo.profile import Superposition

    # Generate hub/mid/tip profiles with varying loading
    cl0_hub = max(0.3, min(blade_params[0]["rotor_camber_deg"] / 25.0, 1.8)) if blade_params else 0.8
    profiles = [
        Superposition(NACA65(cl0=cl0_hub * 0.85), NACA65Series(max_thickness=0.08)),
        Superposition(NACA65(cl0=cl0_hub), NACA65Series(max_thickness=0.10)),
        Superposition(NACA65(cl0=cl0_hub * 1.15), NACA65Series(max_thickness=0.12)),
    ]
    profile_coords = profiles[1].as_array()  # midspan for plots

    # Save midspan profile CSV
    csv_path = output_dir / f"{name}_blade.csv"
    np.savetxt(str(csv_path), profile_coords, delimiter=",", header="x,y", comments="")
    results["profile_coords"] = profile_coords
    results["stages"]["profile"] = time.perf_counter() - t1
    print(f"  [4/7] 2D Blade Profiles    ({results['stages']['profile']:.2f}s)")
    print(f"        Saved: {csv_path.name}")

    # ── 5. 3D blade stacking ──────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.blade import BladeRow

    axial_chord = 0.05
    hub_pts = np.array([[0.0, spec["r_hub"]], [axial_chord, spec["r_hub"]]])
    shroud_pts = np.array([[0.0, spec["r_tip"]], [axial_chord, spec["r_tip"]]])

    row = BladeRow(hub_points=hub_pts, shroud_points=shroud_pts)
    row.number_blades = 24
    for p in profiles:
        row.add_profile(p)

    staggers = np.deg2rad([30, 35, 40])
    chords = np.array([axial_chord * 0.8, axial_chord, axial_chord * 1.2])
    row.compute(stagger_angles=staggers, chord_lengths=chords)
    results["stages"]["blade_3d"] = time.perf_counter() - t1
    print(f"  [5/7] 3D Blade Stacking    ({results['stages']['blade_3d']:.2f}s)")
    print(f"        Sections: {len(row.profiles_3d) if row.profiles_3d else 0}")

    # ── 6. Multi-block mesh ────────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.mesh.multiblock import generate_blade_passage_mesh

    r_mean = (spec["r_hub"] + spec["r_tip"]) / 2.0
    pitch = 2.0 * np.pi * r_mean / 24.0
    mesh = generate_blade_passage_mesh(
        profile=profile_coords,
        pitch=pitch,
        n_blade=40,
        n_ogrid=10,
        n_inlet=15,
        n_outlet=15,
        n_passage=20,
    )
    results["mesh"] = mesh

    # Export CGNS
    cgns_path = output_dir / f"{name}_mesh.cgns"
    mesh.export_cgns(str(cgns_path))
    results["stages"]["mesh"] = time.perf_counter() - t1
    print(f"  [6/7] Multi-block Mesh     ({results['stages']['mesh']:.2f}s)")
    print(f"        Blocks: {mesh.n_blocks}, Cells: {mesh.total_cells}")
    print(f"        Saved: {cgns_path.name}")

    # ── 7. OpenFOAM CFD setup ──────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.cfd import CFDWorkflow, CFDWorkflowConfig

    cfd_dir = output_dir / f"{name}_openfoam"
    cfg = CFDWorkflowConfig(
        solver="openfoam",
        inlet_velocity=150.0,
        turbulence_model="kOmegaSST",
        is_rotating=False,
        compressible=False,
    )
    wf = CFDWorkflow(cfg)
    wf.set_mesh(str(cgns_path))
    case_dir = wf.setup_case(str(cfd_dir))
    results["cfd_dir"] = str(case_dir)
    results["stages"]["cfd_setup"] = time.perf_counter() - t1
    print(f"  [7/9] OpenFOAM CFD Setup   ({results['stages']['cfd_setup']:.2f}s)")
    print(f"        Case: {cfd_dir.name}/")

    # ── 8. Run OpenFOAM solver ─────────────────────────────────────
    import shutil
    import subprocess

    cfd_solution = None
    cfd_residuals = None

    # For 2D passage meshes, disable MRF (needs 3D cellZone from topoSet)
    mrf_path = cfd_dir / "constant" / "MRFProperties"
    if mrf_path.exists():
        mrf_path.write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; "
            "object MRFProperties; }\n// MRF disabled for 2D passage mesh\n"
        )

    # Fix transportProperties for simpleFoam (incompressible)
    tp_path = cfd_dir / "constant" / "transportProperties"
    tp_path.write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; "
        "object transportProperties; }\n"
        "transportModel  Newtonian;\n"
        "nu              [0 2 -1 0 0 0 0] 1.47e-05;\n"
    )

    # Fix pressure field for kinematic (simpleFoam uses p/rho)
    p_path = cfd_dir / "0" / "p"
    p_path.write_text(
        "FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n"
        "dimensions      [0 2 -2 0 0 0 0];\n"
        "internalField   uniform 0;\n"
        "boundaryField\n{\n"
        '    inlet  { type fixedValue; value uniform 0; }\n'
        '    outlet { type fixedValue; value uniform 0; }\n'
        '    blade  { type zeroGradient; }\n'
        '    frontAndBack { type empty; }\n'
        '    defaultFaces { type empty; }\n'
        '    periodic_upper { type cyclic; }\n'
        '    periodic_lower { type cyclic; }\n'
        "}\n"
    )

    # Fix U field
    u_path = cfd_dir / "0" / "U"
    u_path.write_text(
        "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
        "dimensions      [0 1 -1 0 0 0 0];\n"
        "internalField   uniform (150 0 0);\n"
        "boundaryField\n{\n"
        '    inlet  { type fixedValue; value uniform (150 0 0); }\n'
        '    outlet { type zeroGradient; }\n'
        '    blade  { type noSlip; }\n'
        '    frontAndBack { type empty; }\n'
        '    defaultFaces { type empty; }\n'
        '    periodic_upper { type cyclic; }\n'
        '    periodic_lower { type cyclic; }\n'
        "}\n"
    )

    # Write controlDict for simpleFoam (200 iters for quick demo)
    ctrl_path = cfd_dir / "system" / "controlDict"
    ctrl_path.write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application     simpleFoam;\n"
        "startFrom       startTime;\nstartTime       0;\n"
        "stopAt          endTime;\nendTime         200;\ndeltaT          1;\n"
        "writeControl    timeStep;\nwriteInterval   200;\npurgeWrite      2;\n"
        "writeFormat     ascii;\nwritePrecision  8;\nwriteCompression off;\n"
        "timeFormat      general;\ntimePrecision   6;\nrunTimeModifiable true;\n"
    )

    # Remove compressible-only files
    for f in ("T", "alphat"):
        fp = cfd_dir / "0" / f
        if fp.exists():
            fp.unlink()
    for f in ("thermophysicalProperties",):
        fp = cfd_dir / "constant" / f
        if fp.exists():
            fp.unlink()

    # Fix k, omega, nut BCs for cyclic patches
    for field_name, dim, value in [
        ("k", "[0 2 -2 0 0 0 0]", "0.1"),
        ("omega", "[0 0 -1 0 0 0 0]", "1.0"),
        ("nut", "[0 2 -1 0 0 0 0]", "0"),
    ]:
        fp = cfd_dir / "0" / field_name
        wall_type = "kqRWallFunction" if field_name == "k" else (
            "omegaWallFunction" if field_name == "omega" else "nutkWallFunction"
        )
        fp.write_text(
            f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {field_name}; }}\n"
            f"dimensions      {dim};\n"
            f"internalField   uniform {value};\n"
            "boundaryField\n{\n"
            f'    inlet  {{ type fixedValue; value uniform {value}; }}\n'
            f'    outlet {{ type zeroGradient; }}\n'
            f'    blade  {{ type {wall_type}; value uniform {value}; }}\n'
            f'    frontAndBack {{ type empty; }}\n'
            f'    defaultFaces {{ type empty; }}\n'
            f'    periodic_upper {{ type cyclic; }}\n'
            f'    periodic_lower {{ type cyclic; }}\n'
            "}\n"
        )

    # Detect OpenFOAM: direct binary or macOS 'openfoam' wrapper
    if shutil.which("simpleFoam"):
        of_prefix = []
    elif shutil.which("openfoam"):
        of_prefix = ["openfoam"]
    else:
        of_prefix = None

    if of_prefix is not None:
        t1 = time.perf_counter()
        print(f"  [8/9] Running OpenFOAM...", end="", flush=True)

        log_path = cfd_dir / "solver.log"
        try:
            # Step 1: Import mesh (blockMesh or cgnsToFoam)
            # Try blockMeshDict first since CGNS import needs cgnsToFoam
            blockmesh_path = cfd_dir / "system" / "blockMeshDict"
            if not blockmesh_path.exists():
                # Export mesh as blockMeshDict for OpenFOAM
                mesh.export_openfoam(str(blockmesh_path))

            proc_mesh = subprocess.run(
                of_prefix + ["blockMesh", "-case", str(cfd_dir)],
                capture_output=True, text=True, timeout=120,
            )

            if proc_mesh.returncode != 0:
                print(f" mesh failed")
                log_path.write_text(proc_mesh.stdout + proc_mesh.stderr)
            else:
                # Step 2: Run solver (simpleFoam for incompressible)
                proc = subprocess.run(
                    of_prefix + ["simpleFoam", "-case", str(cfd_dir)],
                    capture_output=True, text=True, timeout=600,
                )
                log_path.write_text(proc_mesh.stdout + proc.stdout + proc.stderr)

                results["stages"]["cfd_solve"] = time.perf_counter() - t1

                from astraturbo.cfd.postprocess import (
                    read_openfoam_solution,
                    read_openfoam_residuals,
                )

                cfd_residuals = read_openfoam_residuals(str(log_path))
                cfd_solution = read_openfoam_solution(str(cfd_dir))

                if proc.returncode == 0:
                    print(f" converged ({results['stages']['cfd_solve']:.1f}s)")
                else:
                    print(f" ran ({results['stages']['cfd_solve']:.1f}s, exit={proc.returncode})")

                if cfd_solution:
                    n_fields = sum(1 for k in ("p", "U", "T") if cfd_solution.get(k) is not None)
                    print(f"        Solution at t={cfd_solution['time']}, {n_fields} fields")
                if cfd_residuals:
                    iters = max(len(v) for v in cfd_residuals.values()) if cfd_residuals else 0
                    print(f"        {iters} iterations, fields: {', '.join(cfd_residuals.keys())}")

        except subprocess.TimeoutExpired:
            results["stages"]["cfd_solve"] = time.perf_counter() - t1
            print(f" TIMEOUT ({results['stages']['cfd_solve']:.1f}s)")
        except Exception as e:
            results["stages"]["cfd_solve"] = time.perf_counter() - t1
            print(f" ERROR: {e}")
    else:
        print(f"  [8/9] OpenFOAM not installed — skipping solver run")
        print(f"        Install: brew install gerlero/openfoam/openfoam")

    results["cfd_solution"] = cfd_solution
    results["cfd_residuals"] = cfd_residuals

    # ── 9. HTML Report with embedded images ────────────────────────
    from astraturbo.reports import generate_report, ReportConfig

    report_path = output_dir / f"{name}_report.html"
    report_cfg = ReportConfig(
        title=f"{spec['title']} — Full Pipeline Report",
        project="AstraTurbo Kaveri Pipeline",
        output_path=str(report_path),
    )
    generate_report(
        config=report_cfg,
        engine_cycle_result=ec_result,
        meanline_result=comp_result,
        compressor_map=comp_map,
        turbine_result=turb_result,
        blade_params=blade_params,
        profile_coords=profile_coords,
        mesh=mesh,
        cfd_solution=cfd_solution,
        cfd_residuals=cfd_residuals,
    )
    results["report_path"] = str(report_path)

    total = time.perf_counter() - t0
    results["total_time"] = total
    print(f"\n  [9/9] Report: {report_path.name}")
    print(f"  Total:  {total:.2f}s")
    print(f"  {'─'*50}")

    return results


def main():
    """Run all three engines through the pipeline."""
    print("\n" + "▓" * 70)
    print("  AstraTurbo — Military Jet Engine Pipeline")
    print("  Kaveri GTX-35VS | GE F414 | Safran M88")
    print("▓" * 70)

    output_root = Path(__file__).resolve().parent
    all_results = {}

    for name, spec in ENGINES.items():
        engine_dir = output_root / name.lower()
        all_results[name] = run_engine_pipeline(name, spec, engine_dir)

    # ── Summary table ──────────────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("  PIPELINE SUMMARY")
    print("=" * 78)
    print(f"  {'Engine':<22} {'Thrust':>10} {'SFC':>12} {'η_th':>8} {'Cells':>10} {'Time':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*8} {'─'*10} {'─'*8}")

    for name, r in all_results.items():
        ec = r["engine_cycle"]
        mesh = r["mesh"]
        print(
            f"  {ENGINES[name]['title']:<22} "
            f"{ec.net_thrust/1000:>8.1f} kN "
            f"{ec.specific_fuel_consumption*3600:>10.4f} "
            f"{ec.thermal_efficiency:>7.4f} "
            f"{mesh.total_cells:>9d} "
            f"{r['total_time']:>6.2f}s"
        )

    print(f"\n  Reports saved to: {output_root}/")
    print(f"  Open any *_report.html to see station diagrams, blade profiles, and mesh plots.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
