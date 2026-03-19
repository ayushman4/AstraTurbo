"""Matplotlib-based plot generators for HTML reports.

Each function returns a base64-encoded PNG string suitable for embedding
as a data URI in HTML.  Returns ``""`` if matplotlib is not available.
"""

from __future__ import annotations

import base64
import io


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_engine_stations(result) -> str:
    """Dual-axis bar chart of station P (kPa) and T (K).

    Afterburner station highlighted in orange.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    names = list(result.stations.keys())
    pressures = [st.P_total / 1000.0 for st in result.stations.values()]
    temperatures = [st.T_total for st in result.stations.values()]

    x = np.arange(len(names))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax2 = ax1.twinx()

    # Colour afterburner station differently
    ab = getattr(result, "afterburner", None)
    p_colors = []
    t_colors = []
    for n in names:
        if ab is not None and "afterburner" in n.lower():
            p_colors.append("#e67e22")
            t_colors.append("#e74c3c")
        else:
            p_colors.append("#2980b9")
            t_colors.append("#c0392b")

    ax1.bar(x - width / 2, pressures, width, color=p_colors, label="P (kPa)")
    ax2.bar(x + width / 2, temperatures, width, color=t_colors, alpha=0.7, label="T (K)")

    ax1.set_xlabel("Station")
    ax1.set_ylabel("Total Pressure (kPa)", color="#2980b9")
    ax2.set_ylabel("Total Temperature (K)", color="#c0392b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Engine Station Conditions")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    return _fig_to_base64(fig)


def plot_blade_profile(coords) -> str:
    """Airfoil profile plot from (N,2) coordinate array.

    Blue filled shape with red dashed camber line.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return ""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill(coords[:, 0], coords[:, 1], alpha=0.25, color="#2980b9")
    ax.plot(coords[:, 0], coords[:, 1], color="#2980b9", linewidth=1.5)

    # Approximate camber as midpoint between upper and lower halves
    n = len(coords)
    half = n // 2
    if half > 2:
        upper = coords[:half]
        lower = coords[half:][::-1]
        min_len = min(len(upper), len(lower))
        camber_x = (upper[:min_len, 0] + lower[:min_len, 0]) / 2
        camber_y = (upper[:min_len, 1] + lower[:min_len, 1]) / 2
        ax.plot(camber_x, camber_y, "r--", linewidth=1.2, label="Camber")
        ax.legend(fontsize=8)

    ax.set_aspect("equal")
    ax.set_title("Blade Profile")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def plot_cooling_rows(result) -> str:
    """Grouped bars for effectiveness and coolant fraction per row, with T_blade line."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    rows = result.rows
    if not rows:
        return ""

    indices = [r.row_index for r in rows]
    effectiveness = [r.cooling_effectiveness for r in rows]
    coolant_frac = [r.coolant_fraction for r in rows]
    t_blade = [r.T_blade for r in rows]

    x = np.arange(len(indices))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(6, len(indices) * 1.5), 5))
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, effectiveness, width, color="#27ae60", label="Effectiveness")
    ax1.bar(x + width / 2, coolant_frac, width, color="#3498db", label="Coolant Fraction")
    ax2.plot(x, t_blade, "ro-", linewidth=2, markersize=6, label="T_blade (K)")

    ax1.set_xlabel("Cooling Row")
    ax1.set_ylabel("Fraction")
    ax2.set_ylabel("Blade Temperature (K)", color="red")
    ax1.set_xticks(x)
    ax1.set_xticklabels(indices)
    ax1.set_title("Blade Cooling: Effectiveness & Coolant per Row")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    return _fig_to_base64(fig)


def plot_turbopump_power(result) -> str:
    """Horizontal bar chart: pump power, turbine power, shaft power, margin."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    labels = ["Pump Power", "Turbine Power", "Shaft Power", "Power Margin"]
    values = [
        result.pump_power / 1000.0,
        result.turbine_power / 1000.0,
        result.shaft_power / 1000.0,
        result.power_margin * 100.0,
    ]
    colors = ["#3498db", "#e74c3c", "#9b59b6", "#2ecc71"]
    units = ["kW", "kW", "kW", "%"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors)
    for bar, val, unit in zip(bars, values, units):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} {unit}", va="center", fontsize=9)

    ax.set_title("Turbopump Power Budget")
    ax.set_xlabel("Value")
    return _fig_to_base64(fig)


def plot_propeller_summary(result) -> str:
    """Multi-metric bar chart: FM, tip Mach, disk loading (normalised), CT, CP."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    labels = ["Figure of Merit", "Tip Mach", "Disk Loading\n(kN/m²)", "CT (×10³)", "CP (×10³)"]
    values = [
        result.figure_of_merit,
        result.tip_mach,
        result.disk_loading / 1000.0,
        result.CT * 1000.0,
        result.CP * 1000.0,
    ]
    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Propeller Performance Summary")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    return _fig_to_base64(fig)


def plot_motor_summary(result) -> str:
    """Key metrics bar: efficiency, thermal margin, power density (normalised)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    labels = ["Efficiency", "Thermal Margin", "Power Density\n(kW/kg ÷10)"]
    values = [
        result.efficiency,
        result.thermal_margin,
        result.power_density / 10.0,
    ]
    colors = ["#27ae60", "#e67e22", "#2980b9"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"Electric Motor Summary — {result.motor_type}")
    ax.set_ylim(0, max(values) * 1.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    return _fig_to_base64(fig)


def plot_mesh_2d(mesh) -> str:
    """2D wireframe grid of all blocks in a MultiBlockMesh."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    if not mesh.blocks:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    block_colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12", "#1abc9c"]

    for idx, block in enumerate(mesh.blocks):
        pts = block.points
        color = block_colors[idx % len(block_colors)]
        ni, nj = pts.shape[0], pts.shape[1]

        # Draw i-lines
        for j in range(nj):
            ax.plot(pts[:, j, 0], pts[:, j, 1], color=color, linewidth=0.4, alpha=0.6)
        # Draw j-lines
        for i in range(ni):
            ax.plot(pts[i, :, 0], pts[i, :, 1], color=color, linewidth=0.4, alpha=0.6)

    ax.set_aspect("equal")
    ax.set_title(f"Mesh: {mesh.name} ({mesh.n_blocks} blocks, {mesh.total_cells} cells)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return _fig_to_base64(fig)


def plot_ts_diagram(result) -> str:
    """Temperature-entropy (T-s) diagram for an engine cycle.

    Computes entropy from station P/T data using ideal gas relations
    and plots the thermodynamic cycle.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    stations = result.stations
    if not stations:
        return ""

    cp = 1004.5  # J/(kg·K) for air
    gamma = 1.4
    R_gas = cp * (gamma - 1.0) / gamma  # ~287 J/(kg·K)

    # Compute entropy relative to first station: ds = cp*ln(T2/T1) - R*ln(P2/P1)
    names = list(stations.keys())
    T_vals = [st.T_total for st in stations.values()]
    P_vals = [st.P_total for st in stations.values()]

    T0, P0 = T_vals[0], P_vals[0]
    s_vals = []
    for T, P in zip(T_vals, P_vals):
        ds = cp * np.log(T / T0) - R_gas * np.log(P / P0)
        s_vals.append(ds)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot cycle path
    ax.plot(s_vals, T_vals, "b-o", linewidth=2.0, markersize=7, zorder=5)

    # Annotate each station
    for i, name in enumerate(names):
        short = name.replace("_exit", "").replace("_", " ").title()
        ax.annotate(
            short,
            (s_vals[i], T_vals[i]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=7,
            color="#2c3e50",
        )

    # Close the cycle visually (nozzle exit → ambient)
    ax.plot(
        [s_vals[-1], s_vals[0]], [T_vals[-1], T_vals[0]],
        "b--", linewidth=1.0, alpha=0.4,
    )

    # Shade compressor (work in) and turbine (work out) regions
    # Find compressor and turbine station indices
    comp_idx = []
    turb_idx = []
    for i, n in enumerate(names):
        if "compressor" in n.lower() or "inlet" in n.lower():
            comp_idx.append(i)
        if "turbine" in n.lower():
            turb_idx.append(i)

    ax.set_xlabel("Specific Entropy, s - s₀ (J/kg·K)", fontsize=10)
    ax.set_ylabel("Total Temperature (K)", fontsize=10)
    ax.set_title("Engine Thermodynamic Cycle (T-s Diagram)")
    ax.grid(True, alpha=0.3)

    # Add cycle info
    ab = getattr(result, "afterburner", None)
    info = f"OPR = {P_vals[max(comp_idx)] / P0:.1f}" if comp_idx else ""
    if ab is not None:
        info += "  |  Afterburner ON"
    if info:
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                va="top", color="#7f8c8d")

    return _fig_to_base64(fig)


def plot_velocity_triangles(result) -> str:
    """Velocity triangle diagram for compressor or turbine stages.

    Shows inlet and outlet velocity triangles for the first stage rotor
    as vector arrows within a clean, bounded diagram.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.lines import Line2D
    except ImportError:
        return ""

    if not hasattr(result, "stages") or not result.stages:
        return ""

    n_plot = min(len(result.stages), 2)
    fig, axes = plt.subplots(1, n_plot, figsize=(7 * n_plot, 5))
    if n_plot == 1:
        axes = [axes]

    for idx in range(n_plot):
        ax = axes[idx]
        stage = result.stages[idx]
        tri = stage.rotor_triangles

        U = tri.inlet.U
        Ca_in = tri.inlet.C_axial
        Ct_in = tri.inlet.C_theta
        Ca_out = tri.outlet.C_axial
        Ct_out = tri.outlet.C_theta

        # Normalise by blade speed
        scale = max(abs(U), 1.0)
        u = U / scale
        ca1 = Ca_in / scale
        ct1 = Ct_in / scale
        ca2 = Ca_out / scale
        ct2 = Ct_out / scale
        wt1 = ct1 - u
        wt2 = ct2 - u

        def _arrow(ax, x0, y0, dx, dy, color, ls="-"):
            ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                        linestyle=ls),
                        annotation_clip=True)

        # ── Inlet triangle (left half) ──
        ox = 0.0
        _arrow(ax, ox, 0, ca1, ct1, "#2980b9")          # C1
        _arrow(ax, ox, 0, ca1, wt1, "#e74c3c", ls="--") # W1
        _arrow(ax, ox + ca1, wt1, 0, u, "#27ae60")      # U

        # ── Outlet triangle (right half) ──
        gap = 0.3
        ox2 = ca1 + gap
        _arrow(ax, ox2, 0, ca2, ct2, "#2980b9")          # C2
        _arrow(ax, ox2, 0, ca2, wt2, "#e74c3c", ls="--") # W2
        _arrow(ax, ox2 + ca2, wt2, 0, u, "#27ae60")      # U

        # Text labels (placed relative to midpoint of each vector)
        fs = 9
        ax.text(ca1 * 0.4, ct1 * 0.6, "C₁", color="#2980b9", fontsize=fs, fontweight="bold")
        ax.text(ca1 * 0.4, wt1 * 0.6, "W₁", color="#e74c3c", fontsize=fs, fontweight="bold")
        ax.text(ox2 + ca2 * 0.4, ct2 * 0.6, "C₂", color="#2980b9", fontsize=fs, fontweight="bold")
        ax.text(ox2 + ca2 * 0.4, wt2 * 0.6, "W₂", color="#e74c3c", fontsize=fs, fontweight="bold")

        # Dashed zero line
        ax.axhline(0, color="gray", lw=0.5, ls=":")

        # Bracket labels
        ax.text(ca1 * 0.5, -0.12, "Inlet", ha="center", fontsize=8, color="gray")
        ax.text(ox2 + ca2 * 0.5, -0.12, "Outlet", ha="center", fontsize=8, color="gray")

        ax.set_title(f"Stage {stage.stage_number} Velocity Triangles", fontsize=11)
        ax.set_xlabel("Axial (normalised by U)")
        ax.set_ylabel("Tangential (normalised by U)")
        ax.grid(True, alpha=0.3)

        # Set explicit limits to contain all vectors
        all_x = [0, ca1, ox2, ox2 + ca2, ca1, ox2 + ca2]
        all_y = [0, ct1, wt1, ct2, wt2, wt1 + u, wt2 + u]
        margin = 0.2
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        legend_elements = [
            Line2D([0], [0], color="#2980b9", lw=2, label="C (absolute)"),
            Line2D([0], [0], color="#e74c3c", lw=2, ls="--", label="W (relative)"),
            Line2D([0], [0], color="#27ae60", lw=2, label="U (blade)"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=7)

        # Angle info box
        beta_in = np.degrees(tri.inlet.beta)
        beta_out = np.degrees(tri.outlet.beta)
        alpha_in = np.degrees(tri.inlet.alpha)
        alpha_out = np.degrees(tri.outlet.alpha)
        info = (f"α₁={alpha_in:.1f}° β₁={beta_in:.1f}°\n"
                f"α₂={alpha_out:.1f}° β₂={beta_out:.1f}°\n"
                f"De Haller={tri.de_haller_ratio:.3f}")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7,
                va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_compressor_map_chart(cmap) -> str:
    """Compressor map: PR vs corrected mass flow with speed lines and surge line."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    if not cmap.speed_lines:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cmap_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cmap.speed_lines)))

    # Left: PR vs mass flow
    for i, sl in enumerate(cmap.speed_lines):
        mf = sl.mass_flows
        pr = sl.pressure_ratios
        label = f"N/N_d = {sl.rpm_fraction:.2f}"
        ax1.plot(mf, pr, "-o", color=cmap_colors[i], markersize=4, linewidth=1.5, label=label)

        # Mark stall points
        for j in range(len(mf)):
            if sl.is_stalled[j]:
                ax1.plot(mf[j], pr[j], "rx", markersize=8, markeredgewidth=2)

    # Surge line
    if cmap.surge_line:
        surge_mf = [p[0] for p in cmap.surge_line]
        surge_pr = [p[1] for p in cmap.surge_line]
        ax1.plot(surge_mf, surge_pr, "r--", linewidth=2.5, label="Surge Line", zorder=10)

    # Design point
    if cmap.design_point:
        dp = cmap.design_point
        ax1.plot(dp.get("mass_flow", 0), dp.get("pr", 0), "k*",
                 markersize=15, zorder=15, label="Design Point")

    ax1.set_xlabel("Mass Flow (kg/s)")
    ax1.set_ylabel("Pressure Ratio")
    ax1.set_title("Compressor Map — PR vs Mass Flow")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right: Efficiency vs mass flow
    for i, sl in enumerate(cmap.speed_lines):
        mf = sl.mass_flows
        eff = sl.efficiencies
        label = f"N/N_d = {sl.rpm_fraction:.2f}"
        ax2.plot(mf, eff, "-s", color=cmap_colors[i], markersize=3, linewidth=1.5, label=label)

    if cmap.design_point:
        dp = cmap.design_point
        ax2.plot(dp.get("mass_flow", 0), dp.get("efficiency", 0), "k*",
                 markersize=15, zorder=15, label="Design Point")

    ax2.set_xlabel("Mass Flow (kg/s)")
    ax2.set_ylabel("Isentropic Efficiency")
    ax2.set_title("Compressor Map — Efficiency vs Mass Flow")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_blade_loading(profile_coords, stagger_deg: float = 0.0) -> str:
    """Blade surface velocity distribution (loading diagram).

    Approximates suction/pressure side V/V_inlet from the airfoil
    thickness distribution using thin-airfoil theory.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    coords = np.asarray(profile_coords)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) < 10:
        return ""

    n = len(coords)
    half = n // 2
    if half < 5:
        return ""

    # Split into upper and lower surfaces
    upper = coords[:half]
    lower = coords[half:][::-1]

    # Resample both to common x/c
    min_len = min(len(upper), len(lower))
    xc = np.linspace(0, 1, min_len)
    x_range_u = upper[-1, 0] - upper[0, 0]
    x_range_l = lower[-1, 0] - lower[0, 0]

    if abs(x_range_u) < 1e-12 or abs(x_range_l) < 1e-12:
        return ""

    xc_u = (upper[:, 0] - upper[0, 0]) / x_range_u
    xc_l = (lower[:, 0] - lower[0, 0]) / x_range_l
    y_u = np.interp(xc, xc_u, upper[:, 1])
    y_l = np.interp(xc, xc_l, lower[:, 1])

    # Thickness = gap between surfaces
    thickness = y_u - y_l
    max_thick = np.max(np.abs(thickness)) + 1e-12

    # Thin-airfoil: suction side accelerates, pressure side decelerates
    # V/V_inf ≈ 1 ± k * (thickness / max_thickness)
    k = 0.6  # scaling factor for realistic range
    v_suction = 1.0 + k * np.abs(thickness) / max_thick
    v_pressure = 1.0 - k * 0.5 * np.abs(thickness) / max_thick

    # Smooth LE/TE stagnation
    taper = np.ones_like(xc)
    le_zone = xc < 0.05
    te_zone = xc > 0.95
    taper[le_zone] = xc[le_zone] / 0.05
    taper[te_zone] = (1.0 - xc[te_zone]) / 0.05
    v_suction = 1.0 + (v_suction - 1.0) * taper
    v_pressure = 1.0 + (v_pressure - 1.0) * taper

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xc, v_suction, "b-", linewidth=2.0, label="Suction side")
    ax.plot(xc, v_pressure, "r-", linewidth=2.0, label="Pressure side")
    ax.fill_between(xc, v_suction, 1.0, alpha=0.15, color="blue")
    ax.fill_between(xc, v_pressure, 1.0, alpha=0.15, color="red")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Freestream")
    ax.set_xlabel("x / chord")
    ax.set_ylabel("V / V∞")
    ax.set_title("Blade Surface Velocity Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1.8)

    # Annotate max loading
    i_max = np.argmax(v_suction)
    ax.annotate(f"V/V∞={v_suction[i_max]:.2f}",
                xy=(xc[i_max], v_suction[i_max]),
                xytext=(xc[i_max] + 0.1, v_suction[i_max] + 0.08),
                arrowprops=dict(arrowstyle="->", color="blue"),
                fontsize=8, color="blue")

    return _fig_to_base64(fig)


# ── CFD Post-Processing Plots ──────────────────────────────────────


def _cfd_extract_2d(cfd_solution: dict, field_name: str):
    """Extract 2D points and field values, handling points/cells mismatch.

    OpenFOAM stores field values per cell but points are vertices (more points
    than cells). This function subsamples points to match the cell count.

    Returns (x, y, values) or (None, None, None) if not available.
    """
    import numpy as np

    points = cfd_solution.get("points")
    field = cfd_solution.get(field_name)
    if points is None or field is None:
        return None, None, None
    if len(field) <= 1:
        return None, None, None

    n_cells = len(field)
    n_points = len(points)

    # Subsample points to match cell count
    if n_points > n_cells:
        step = max(1, n_points // n_cells)
        pts = points[::step][:n_cells]
    else:
        pts = points[:n_cells]

    vals = field[:len(pts)]
    x, y = pts[:, 0], pts[:, 1]

    # Filter to one z-plane if 3D slab (e.g. OpenFOAM 2D with front/back)
    z = pts[:, 2]
    if np.ptp(z) > 1e-6:
        # Take points nearest the minimum z-plane (front face)
        z_min = z.min()
        tol = np.ptp(z) * 0.05 + 1e-6
        mask = np.abs(z - z_min) < tol
        if mask.sum() >= 3:
            x, y, vals = x[mask], y[mask], vals[mask]

    if len(x) < 3:
        return None, None, None

    return x, y, vals


def plot_cfd_pressure_field(cfd_solution: dict) -> str:
    """2D scatter plot of pressure field from OpenFOAM solution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    x, y, p = _cfd_extract_2d(cfd_solution, "p")
    if x is None:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(x, y, c=p, s=3, cmap="RdYlBu_r", edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Pressure (m²/s²)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CFD Pressure Field (simpleFoam)")
    ax.set_aspect("equal")
    return _fig_to_base64(fig)


def plot_cfd_velocity_field(cfd_solution: dict) -> str:
    """2D scatter plot of velocity magnitude from OpenFOAM solution."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    U = cfd_solution.get("U")
    if U is None or U.ndim == 1 or len(U) <= 1:
        return ""

    # Compute magnitude, then use _cfd_extract_2d with a synthetic scalar
    points = cfd_solution.get("points")
    if points is None:
        return ""

    U_mag = np.sqrt(U[:, 0]**2 + U[:, 1]**2 + U[:, 2]**2)
    fake_sol = {"points": points, "Umag": U_mag}
    x, y, U_plot = _cfd_extract_2d(fake_sol, "Umag")
    if x is None:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(x, y, c=U_plot, s=3, cmap="jet", edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Velocity Magnitude (m/s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CFD Velocity Field (simpleFoam)")
    ax.set_aspect("equal")
    return _fig_to_base64(fig)


def plot_cfd_temperature_field(cfd_solution: dict) -> str:
    """2D scatter plot of temperature field from OpenFOAM solution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    x, y, T_plot = _cfd_extract_2d(cfd_solution, "T")
    if x is None:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(x, y, c=T_plot, s=3, cmap="hot", edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Temperature (K)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CFD Temperature Field")
    ax.set_aspect("equal")
    return _fig_to_base64(fig)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CFD Temperature Field")
    ax.set_aspect("equal")
    return _fig_to_base64(fig)


def plot_cfd_residuals(residuals: dict) -> str:
    """Residual convergence history plot.

    Args:
        residuals: Dict from read_openfoam_residuals(), field_name → array.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    if not residuals:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12", "#1abc9c"]

    for i, (field, values) in enumerate(residuals.items()):
        if len(values) == 0:
            continue
        color = colors[i % len(colors)]
        iterations = np.arange(1, len(values) + 1)
        ax.semilogy(iterations, values, color=color, linewidth=1.2, label=field)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Initial Residual")
    ax.set_title("CFD Solver Convergence")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    return _fig_to_base64(fig)
