#!/usr/bin/env python3
"""Production-quality CFD pipeline for three military jet engines.

Unlike run_engines.py (demo), this uses:
  - Higher mesh resolution (~5000 cells per passage)
  - O-grid wall refinement with grading
  - Conservative under-relaxation for solver stability
  - Self-consistent simpleFoam (incompressible) setup
  - 500-iteration convergence target

Engines: Kaveri GTX-35VS | GE F414 | Safran M88

Usage:
    python examples/pipeline/run_production.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from examples.pipeline.run_engines import ENGINES


def run_production(name: str, spec: dict, output_dir: Path) -> dict:
    """Run one engine through the production CFD pipeline."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {spec['title']}  [PRODUCTION]")
    print(f"{'='*70}")

    # ── 1. Engine cycle ────────────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.design.engine_cycle import engine_cycle

    ec_result = engine_cycle(
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
    dt = time.perf_counter() - t1
    print(f"  [1/8] Engine Cycle         ({dt:.2f}s)")
    print(f"        Thrust: {ec_result.net_thrust/1000:.1f} kN, SFC: {ec_result.specific_fuel_consumption*3600:.4f}")

    # ── 2. LP Compressor + map ─────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.design import meanline_compressor, meanline_to_blade_parameters
    from astraturbo.design import generate_compressor_map

    lp_pr = spec["opr"] / spec["hp_pr"]
    comp = meanline_compressor(
        overall_pressure_ratio=lp_pr,
        mass_flow=spec["mass_flow"],
        rpm=spec["rpm"],
        r_hub=spec["r_hub"],
        r_tip=spec["r_tip"],
    )
    blade_params = meanline_to_blade_parameters(comp)
    comp_map = generate_compressor_map(comp, rpm_fractions=[0.7, 0.85, 1.0, 1.05], n_points=10)
    dt = time.perf_counter() - t1
    print(f"  [2/8] LP Compressor        ({dt:.2f}s)  {comp.n_stages} stages, PR={comp.overall_pressure_ratio:.2f}")

    # ── 3. HP Turbine ──────────────────────────────────────────────
    t1 = time.perf_counter()
    from astraturbo.design.turbine import meanline_turbine

    turb = meanline_turbine(
        overall_expansion_ratio=spec["turbine_er"],
        mass_flow=spec["mass_flow"],
        rpm=spec["hp_rpm"],
        r_hub=spec["turbine_r_hub"],
        r_tip=spec["turbine_r_tip"],
        T_inlet=spec["tit"],
    )
    dt = time.perf_counter() - t1
    print(f"  [3/8] HP Turbine           ({dt:.2f}s)  {turb.n_stages} stages, ER={turb.overall_expansion_ratio:.2f}")

    # ── 4. 2D Blade profile ────────────────────────────────────────
    from astraturbo.camberline import NACA65
    from astraturbo.thickness import NACA65Series
    from astraturbo.profile import Superposition

    cl0 = max(0.3, min(blade_params[0]["rotor_camber_deg"] / 25.0, 1.8)) if blade_params else 0.8
    prof = Superposition(NACA65(cl0=cl0), NACA65Series(max_thickness=0.10))
    profile_coords = prof.as_array()
    np.savetxt(str(output_dir / f"{name}_blade.csv"), profile_coords, delimiter=",", header="x,y", comments="")
    print(f"  [4/8] Blade Profile        ({cl0:.2f} cl0, {len(profile_coords)} pts)")

    # ── 5. Production mesh (high resolution) ───────────────────────
    t1 = time.perf_counter()
    from astraturbo.mesh.multiblock import generate_blade_passage_mesh

    r_mean = (spec["r_hub"] + spec["r_tip"]) / 2.0
    pitch = 2.0 * np.pi * r_mean / 24.0

    mesh = generate_blade_passage_mesh(
        profile=profile_coords,
        pitch=pitch,
        n_blade=80,
        n_ogrid=12,
        n_inlet=30,
        n_outlet=30,
        n_passage=40,
        ogrid_thickness=0.010,
        grading_ogrid=1.1,
        grading_inlet=0.5,
        grading_outlet=2.0,
    )
    dt = time.perf_counter() - t1
    print(f"  [5/8] Production Mesh      ({dt:.2f}s)  {mesh.n_blocks} blocks, {mesh.total_cells} cells")

    # ── 6. OpenFOAM case (self-consistent simpleFoam) ──────────────
    t1 = time.perf_counter()
    from astraturbo.cfd.openfoam import write_simpleFoam_case

    cfd_dir = output_dir / f"{name}_cfd"
    write_simpleFoam_case(
        case_dir=cfd_dir,
        mesh=mesh,
        inlet_velocity=150.0,
        n_iterations=1000,
    )
    dt = time.perf_counter() - t1
    print(f"  [6/8] CFD Case Setup       ({dt:.2f}s)  simpleFoam, 1000 iters")

    # ── 7. Run OpenFOAM ───────────────────────────────────────────
    cfd_solution = None
    cfd_residuals = None

    if shutil.which("simpleFoam"):
        of_prefix = []
    elif shutil.which("openfoam"):
        of_prefix = ["openfoam"]
    else:
        of_prefix = None

    if of_prefix is not None:
        t1 = time.perf_counter()
        print(f"  [7/8] Running simpleFoam...", end="", flush=True)

        log_path = cfd_dir / "solver.log"
        try:
            # blockMesh
            proc_m = subprocess.run(
                of_prefix + ["blockMesh", "-case", str(cfd_dir)],
                capture_output=True, text=True, timeout=120,
            )
            if proc_m.returncode != 0:
                print(f" blockMesh failed")
                log_path.write_text(proc_m.stdout + proc_m.stderr)
            else:
                # simpleFoam
                proc = subprocess.run(
                    of_prefix + ["simpleFoam", "-case", str(cfd_dir)],
                    capture_output=True, text=True, timeout=600,
                )
                log_path.write_text(proc.stdout + proc.stderr)
                dt = time.perf_counter() - t1

                from astraturbo.cfd.postprocess import read_openfoam_solution, read_openfoam_residuals
                cfd_residuals = read_openfoam_residuals(str(log_path))
                cfd_solution = read_openfoam_solution(str(cfd_dir))

                iters = len(cfd_residuals.get("p", [])) if cfd_residuals else 0
                converged = proc.returncode == 0

                if converged:
                    print(f" CONVERGED ({dt:.1f}s, {iters} iters)")
                else:
                    print(f" {iters} iters ({dt:.1f}s, exit={proc.returncode})")

                if cfd_residuals:
                    final_p = cfd_residuals.get("p", [1.0])[-1] if "p" in cfd_residuals else "N/A"
                    print(f"        Final residuals: p={final_p:.2e}, fields={list(cfd_residuals.keys())}")
                if cfd_solution:
                    n_fields = sum(1 for k in ("p", "U") if cfd_solution.get(k) is not None)
                    pts = cfd_solution.get("points")
                    n_pts = len(pts) if pts is not None else 0
                    print(f"        Solution: {n_fields} fields, {n_pts} points")

        except subprocess.TimeoutExpired:
            print(f" TIMEOUT")
        except Exception as e:
            print(f" ERROR: {e}")
    else:
        print(f"  [7/8] OpenFOAM not installed — skipping")

    # ── 8. Report ──────────────────────────────────────────────────
    from astraturbo.reports import generate_report, ReportConfig

    report_path = output_dir / f"{name}_report.html"
    generate_report(
        config=ReportConfig(
            title=f"{spec['title']} — Production CFD Report",
            project="AstraTurbo Production Pipeline",
            output_path=str(report_path),
        ),
        engine_cycle_result=ec_result,
        meanline_result=comp,
        compressor_map=comp_map,
        turbine_result=turb,
        blade_params=blade_params,
        profile_coords=profile_coords,
        mesh=mesh,
        cfd_solution=cfd_solution,
        cfd_residuals=cfd_residuals,
    )

    total = time.perf_counter() - t0
    print(f"  [8/8] Report: {report_path.name}")
    print(f"  Total: {total:.1f}s")

    return {
        "name": name,
        "thrust_kN": ec_result.net_thrust / 1000,
        "cells": mesh.total_cells,
        "cfd_iters": len(cfd_residuals.get("p", [])) if cfd_residuals else 0,
        "total_time": total,
        "report": str(report_path),
    }


def main():
    print("\n" + "=" * 70)
    print("  AstraTurbo — Production CFD Pipeline")
    print("  Kaveri GTX-35VS | GE F414 | Safran M88")
    print("=" * 70)

    output_root = Path(__file__).resolve().parent
    results = []

    for name, spec in ENGINES.items():
        r = run_production(name, spec, output_root / f"{name.lower()}_production")
        results.append(r)

    print(f"\n\n{'='*70}")
    print("  PRODUCTION RESULTS")
    print(f"{'='*70}")
    print(f"  {'Engine':<24} {'Thrust':>8} {'Cells':>8} {'CFD Iters':>10} {'Time':>8}")
    print(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for r in results:
        print(f"  {r['name']:<24} {r['thrust_kN']:>7.1f}kN {r['cells']:>7d} {r['cfd_iters']:>9d} {r['total_time']:>6.1f}s")
    print(f"\n  Reports: {output_root}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
