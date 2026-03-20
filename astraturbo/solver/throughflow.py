"""Streamline Curvature Method (SCM) throughflow solver.

Solves the axisymmetric throughflow equations on an S2m (meridional)
plane using the Streamline Curvature Method. The solver iterates on
streamline positions until convergence, computing radial equilibrium
at each computing station.

The radial equilibrium equation solved is:

    V_m * dV_m/dm = -(1/rho) * dP/dr * dr/dm
                    + V_theta^2/r * dr/dm
                    - V_m * dV_m/dr * sin(phi)
                    + F_blade

where m is the meridional coordinate, r is the radius, and phi is the
streamline slope angle.

References:
    Novak, R.A., "Streamline Curvature Computing Procedures for
    Fluid-Flow Problems", ASME J. Eng. Power, 1967.

    Cumpsty, N.A., "Compressor Aerodynamics", Longman, 1989.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .loss_models import (
    lieblein_diffusion_factor,
    lieblein_profile_loss,
    ainley_mathieson_secondary_loss,
    tip_clearance_loss,
    carter_deviation,
)


@dataclass
class ThroughflowConfig:
    """Configuration for the throughflow solver."""

    # Convergence
    max_iterations: int = 200
    convergence_tolerance: float = 1e-4
    relaxation_factor: float = 0.3

    # Fluid properties
    gamma: float = 1.4                    # Ratio of specific heats
    R_gas: float = 287.058                # Gas constant (J/kg/K)
    cp: float = 1004.5                    # Specific heat at constant pressure (J/kg/K)

    # Loss model selection
    profile_loss_model: Literal["lieblein", "none"] = "lieblein"
    secondary_loss_model: Literal["ainley_mathieson", "none"] = "ainley_mathieson"
    tip_clearance_model: Literal["simple", "none"] = "simple"
    deviation_model: Literal["carter", "none"] = "carter"

    # Blockage
    blockage_factor: float = 0.02         # Hub/tip endwall blockage fraction

    # Mesh
    n_streamlines: int = 11               # Number of streamlines (including hub/tip)
    n_stations: int = 20                  # Number of computing stations


@dataclass
class BladeRowSpec:
    """Specification for a single blade row in the throughflow solver."""

    row_type: Literal["rotor", "stator"] = "stator"
    n_blades: int = 40
    inlet_station: int = 0                # Index of upstream computing station
    outlet_station: int = 1               # Index of downstream computing station

    # Geometry at each streamline (arrays of length n_streamlines)
    chord: NDArray[np.float64] | None = None        # (m)
    stagger: NDArray[np.float64] | None = None      # (degrees)
    camber: NDArray[np.float64] | None = None        # (degrees)
    inlet_metal_angle: NDArray[np.float64] | None = None   # (degrees)
    outlet_metal_angle: NDArray[np.float64] | None = None  # (degrees)

    # Tip clearance
    tip_clearance: float = 0.0            # (m), 0 for stators / shrouded

    # Blade speed (only for rotors)
    omega: float = 0.0                    # Angular velocity (rad/s)


@dataclass
class ThroughflowResult:
    """Result from the throughflow solver."""

    converged: bool = False
    n_iterations: int = 0
    residual_history: list[float] = field(default_factory=list)

    # Field arrays: (n_stations, n_streamlines)
    radius: NDArray[np.float64] | None = None
    axial_coordinate: NDArray[np.float64] | None = None
    pressure: NDArray[np.float64] | None = None           # Static pressure (Pa)
    temperature: NDArray[np.float64] | None = None         # Static temperature (K)
    total_pressure: NDArray[np.float64] | None = None      # Total pressure (Pa)
    total_temperature: NDArray[np.float64] | None = None   # Total temperature (K)
    velocity_meridional: NDArray[np.float64] | None = None # V_m (m/s)
    velocity_tangential: NDArray[np.float64] | None = None # V_theta (m/s)
    velocity_absolute: NDArray[np.float64] | None = None   # V (m/s)
    flow_angle: NDArray[np.float64] | None = None          # alpha (degrees)
    mach_number: NDArray[np.float64] | None = None
    density: NDArray[np.float64] | None = None             # (kg/m^3)

    # Streamline positions
    streamline_r: NDArray[np.float64] | None = None        # (n_stations, n_streamlines)

    # Per-row results
    row_losses: list[dict] | None = None


class ThroughflowSolver:
    """Streamline Curvature Method throughflow solver.

    Solves axisymmetric flow through an axial compressor or turbine
    by iterating on streamline positions until radial equilibrium
    is satisfied at each computing station.

    Usage::

        config = ThroughflowConfig(n_streamlines=11, n_stations=20)
        solver = ThroughflowSolver(config)

        # Define annulus
        solver.set_annulus(hub_r, tip_r, axial_coords)

        # Add blade rows
        rotor = BladeRowSpec(row_type="rotor", n_blades=36, omega=1500.0, ...)
        solver.add_blade_row(rotor)

        # Set inlet conditions
        solver.set_inlet_conditions(Pt=101325, Tt=288.15, alpha=0.0)

        # Solve
        result = solver.solve()
    """

    def __init__(self, config: ThroughflowConfig | None = None) -> None:
        self.config = config or ThroughflowConfig()
        self._blade_rows: list[BladeRowSpec] = []
        self._hub_r: NDArray[np.float64] | None = None
        self._tip_r: NDArray[np.float64] | None = None
        self._axial: NDArray[np.float64] | None = None
        self._inlet_Pt: float = 101325.0
        self._inlet_Tt: float = 288.15
        self._inlet_alpha: float = 0.0
        self._mass_flow: float = 0.0

    def set_annulus(
        self,
        hub_radius: NDArray[np.float64],
        tip_radius: NDArray[np.float64],
        axial_coordinates: NDArray[np.float64],
    ) -> None:
        """Define the annulus geometry.

        Args:
            hub_radius: (n_stations,) hub radius at each axial station.
            tip_radius: (n_stations,) tip/casing radius at each station.
            axial_coordinates: (n_stations,) axial positions.
        """
        self._hub_r = np.asarray(hub_radius, dtype=np.float64)
        self._tip_r = np.asarray(tip_radius, dtype=np.float64)
        self._axial = np.asarray(axial_coordinates, dtype=np.float64)

    def add_blade_row(self, row: BladeRowSpec) -> None:
        """Add a blade row to the machine."""
        self._blade_rows.append(row)

    def set_inlet_conditions(
        self,
        total_pressure: float = 101325.0,
        total_temperature: float = 288.15,
        flow_angle: float = 0.0,
        mass_flow: float = 0.0,
    ) -> None:
        """Set inlet flow conditions.

        Args:
            total_pressure: Inlet total pressure (Pa).
            total_temperature: Inlet total temperature (K).
            flow_angle: Inlet absolute flow angle (degrees from axial).
            mass_flow: Mass flow rate (kg/s). Used to compute initial V_m.
        """
        self._inlet_Pt = total_pressure
        self._inlet_Tt = total_temperature
        self._inlet_alpha = flow_angle
        self._mass_flow = mass_flow
        self._inlet_alpha = flow_angle

    def solve(self) -> ThroughflowResult:
        """Run the throughflow solver.

        Iterates on streamline positions until convergence:
        1. Initialize streamlines as straight lines between hub and tip
        2. Compute flow at each station assuming the current streamline positions
        3. Apply loss/deviation models through blade rows
        4. Solve radial equilibrium to get new velocity distribution
        5. Update streamline positions based on mass flow balance
        6. Repeat until positions converge

        Returns:
            ThroughflowResult with all field variables.
        """
        cfg = self.config
        ns = cfg.n_stations
        nsl = cfg.n_streamlines

        result = ThroughflowResult()

        if self._hub_r is None or self._tip_r is None or self._axial is None:
            result.converged = False
            return result

        # Ensure annulus arrays match station count
        hub_r = np.interp(
            np.linspace(0, 1, ns),
            np.linspace(0, 1, len(self._hub_r)),
            self._hub_r,
        )
        tip_r = np.interp(
            np.linspace(0, 1, ns),
            np.linspace(0, 1, len(self._tip_r)),
            self._tip_r,
        )
        axial = np.interp(
            np.linspace(0, 1, ns),
            np.linspace(0, 1, len(self._axial)),
            self._axial,
        )

        # Initialize streamline radii (straight lines hub to tip)
        sl_r = np.zeros((ns, nsl), dtype=np.float64)
        for i in range(ns):
            sl_r[i] = np.linspace(hub_r[i], tip_r[i], nsl)

        # Initialize flow field arrays
        Pt = np.full((ns, nsl), self._inlet_Pt, dtype=np.float64)
        Tt = np.full((ns, nsl), self._inlet_Tt, dtype=np.float64)
        V_m = np.zeros((ns, nsl), dtype=np.float64)
        V_theta = np.zeros((ns, nsl), dtype=np.float64)
        alpha = np.full((ns, nsl), self._inlet_alpha, dtype=np.float64)
        rho = np.zeros((ns, nsl), dtype=np.float64)

        # Initial velocity estimate from mass flow and annulus area
        inlet_alpha_rad = np.radians(self._inlet_alpha)
        T_static_init = self._inlet_Tt * 2.0 / (cfg.gamma + 1.0)  # Choking limit as lower bound
        T_static_init = max(self._inlet_Tt * 0.95, T_static_init)
        rho_init = self._inlet_Pt / (cfg.R_gas * T_static_init)
        area_init = np.pi * (tip_r[0] ** 2 - hub_r[0] ** 2)

        # Compute V_m from continuity if mass flow is provided
        if hasattr(self, '_mass_flow') and self._mass_flow > 0 and area_init > 0:
            V_m_init = self._mass_flow / (rho_init * area_init * (1.0 - cfg.blockage_factor))
        else:
            V_m_init = 150.0  # Fallback

        V_m[:] = V_m_init
        V_theta[:] = V_m_init * np.tan(inlet_alpha_rad)

        row_losses = [{} for _ in self._blade_rows]
        residual_history = []

        # Main iteration loop
        for iteration in range(cfg.max_iterations):
            sl_r_old = sl_r.copy()

            # === Step 1: Compute thermodynamic state at each point ===
            for i in range(ns):
                for j in range(nsl):
                    V = np.sqrt(V_m[i, j] ** 2 + V_theta[i, j] ** 2)
                    T_static = Tt[i, j] - V**2 / (2.0 * cfg.cp)
                    T_static = max(T_static, 100.0)  # Floor for stability

                    # Isentropic relations
                    P_static = Pt[i, j] * (T_static / Tt[i, j]) ** (
                        cfg.gamma / (cfg.gamma - 1.0)
                    )
                    rho[i, j] = P_static / (cfg.R_gas * T_static)

            # === Step 2: Process blade rows (losses, deviation, work) ===
            for row_idx, row in enumerate(self._blade_rows):
                i_in = min(row.inlet_station, ns - 1)
                i_out = min(row.outlet_station, ns - 1)

                for j in range(nsl):
                    r = sl_r[i_out, j]
                    U = row.omega * r if row.row_type == "rotor" else 0.0

                    # Relative frame for rotors
                    if row.row_type == "rotor":
                        W_theta_in = V_theta[i_in, j] - U
                        W_m_in = V_m[i_in, j]
                        W_in = np.sqrt(W_m_in**2 + W_theta_in**2)
                        beta_in = np.degrees(np.arctan2(W_theta_in, W_m_in))
                    else:
                        W_theta_in = V_theta[i_in, j]
                        W_m_in = V_m[i_in, j]
                        W_in = np.sqrt(W_m_in**2 + W_theta_in**2)
                        beta_in = alpha[i_in, j]

                    # Get blade geometry at this streamline
                    camber_j = _get_blade_param(row.camber, j, nsl, 30.0)
                    stagger_j = _get_blade_param(row.stagger, j, nsl, 20.0)
                    chord_j = _get_blade_param(row.chord, j, nsl, 0.05)
                    metal_out = _get_blade_param(row.outlet_metal_angle, j, nsl, -10.0)
                    span = tip_r[i_out] - hub_r[i_out]
                    pitch = 2.0 * np.pi * r / max(row.n_blades, 1)
                    solidity = chord_j / pitch if pitch > 1e-10 else 1.0

                    # Apply deviation model
                    if cfg.deviation_model == "carter":
                        deviation = carter_deviation(camber_j, solidity, stagger_j)
                    else:
                        deviation = 0.0

                    # Exit flow angle = metal angle + deviation
                    beta_out = metal_out + deviation

                    # Exit velocity components
                    V_m_out = V_m[i_in, j]  # Continuity (approximate)
                    if row.row_type == "rotor":
                        W_theta_out = V_m_out * np.tan(np.radians(beta_out))
                        V_theta_out = W_theta_out + U

                        # Euler work equation: delta_h0 = U * delta_V_theta
                        delta_V_theta = V_theta_out - V_theta[i_in, j]
                        delta_h0 = U * delta_V_theta
                        Tt[i_out, j] = Tt[i_in, j] + delta_h0 / cfg.cp
                    else:
                        V_theta_out = V_m_out * np.tan(np.radians(beta_out))
                        Tt[i_out, j] = Tt[i_in, j]  # No work in stator

                    V_theta[i_out, j] = V_theta_out

                    # Exit absolute angle
                    alpha[i_out, j] = np.degrees(
                        np.arctan2(V_theta_out, V_m_out)
                    )

                    # Compute losses
                    W_out = np.sqrt(V_m_out**2 + V_theta_out**2)

                    df = lieblein_diffusion_factor(
                        W_in, W_out, W_theta_in,
                        V_theta_out - U if row.row_type == "rotor" else V_theta_out,
                        solidity,
                    )

                    Re_chord = rho[i_in, j] * W_in * chord_j / 1.8e-5
                    omega_p = lieblein_profile_loss(df, solidity, Re_chord)

                    omega_s = 0.0
                    if cfg.secondary_loss_model == "ainley_mathieson":
                        omega_s = ainley_mathieson_secondary_loss(
                            abs(beta_in), abs(beta_out), span, chord_j
                        )

                    omega_tc = 0.0
                    if cfg.tip_clearance_model == "simple" and row.tip_clearance > 0:
                        loading = abs(V_theta_out - V_theta[i_in, j]) / max(U, 1.0)
                        omega_tc = tip_clearance_loss(
                            row.tip_clearance, span, loading
                        )

                    omega_total = omega_p + omega_s + omega_tc

                    # Apply loss to total pressure
                    q_in = 0.5 * rho[i_in, j] * W_in**2
                    Pt[i_out, j] = Pt[i_in, j] - omega_total * q_in
                    if row.row_type == "rotor":
                        # Rothalpy is conserved, adjust Pt for work input
                        Pt[i_out, j] = Pt[i_in, j] * (
                            Tt[i_out, j] / Tt[i_in, j]
                        ) ** (cfg.gamma / (cfg.gamma - 1.0)) - omega_total * q_in

                    row_losses[row_idx] = {
                        "profile_loss_mean": float(omega_p),
                        "secondary_loss_mean": float(omega_s),
                        "tip_clearance_loss_mean": float(omega_tc),
                        "total_loss_mean": float(omega_total),
                        "diffusion_factor_mean": float(df),
                    }

                # Propagate conditions to downstream stations
                for i_mid in range(i_in + 1, i_out):
                    frac = (i_mid - i_in) / max(i_out - i_in, 1)
                    Pt[i_mid] = Pt[i_in] * (1 - frac) + Pt[i_out] * frac
                    Tt[i_mid] = Tt[i_in] * (1 - frac) + Tt[i_out] * frac
                    V_theta[i_mid] = V_theta[i_in] * (1 - frac) + V_theta[i_out] * frac

            # === Step 3: Radial equilibrium equation ===
            for i in range(ns):
                for j in range(1, nsl - 1):
                    r = sl_r[i, j]
                    if r < 1e-10:
                        continue

                    # Centripetal acceleration term: V_theta^2 / r
                    centripetal = V_theta[i, j] ** 2 / r

                    # Radial pressure gradient (finite difference)
                    dr = sl_r[i, j + 1] - sl_r[i, j - 1]
                    if abs(dr) < 1e-15:
                        continue

                    dPt_dr = (Pt[i, j + 1] - Pt[i, j - 1]) / dr
                    dTt_dr = (Tt[i, j + 1] - Tt[i, j - 1]) / dr

                    # Simple radial equilibrium (SRE)
                    # V_m^2 = V_m_ref^2 + 2 * integral(V_theta^2/r dr + ...)
                    # For hub streamline, use reference value
                    if j == 1:
                        V_m_ref = V_m[i, 0]
                    else:
                        V_m_ref = V_m[i, j - 1]

                    r_prev = sl_r[i, j - 1]
                    dr_step = r - r_prev

                    # SRE: dV_m^2/dr = 2*(V_theta^2/r) - 2*V_theta*dV_theta/dr
                    # for irrotational vortex
                    V_m_sq = V_m_ref**2 + 2.0 * centripetal * dr_step

                    # Apply blockage correction
                    blockage = 1.0 - cfg.blockage_factor
                    V_m_sq = V_m_sq / max(blockage**2, 0.01)

                    V_m[i, j] = np.sqrt(max(V_m_sq, 1.0))

            # === Step 4: Update streamline positions ===
            for i in range(ns):
                # Mass flow per streamtube should be constant
                # m_dot_tube = rho * V_m * 2*pi*r * dr (for each tube)
                total_mass = 0.0
                tube_mass = np.zeros(nsl - 1)
                for j in range(nsl - 1):
                    r_mid = 0.5 * (sl_r[i, j] + sl_r[i, j + 1])
                    dr_tube = sl_r[i, j + 1] - sl_r[i, j]
                    rho_mid = 0.5 * (rho[i, j] + rho[i, j + 1])
                    vm_mid = 0.5 * (V_m[i, j] + V_m[i, j + 1])
                    tube_mass[j] = rho_mid * vm_mid * 2.0 * np.pi * r_mid * abs(dr_tube)
                    total_mass += tube_mass[j]

                if total_mass < 1e-10:
                    continue

                # Redistribute streamlines for equal mass flow per tube
                target_tube_mass = total_mass / (nsl - 1)
                new_r = np.zeros(nsl)
                new_r[0] = hub_r[i]
                new_r[-1] = tip_r[i]

                cumul_mass = 0.0
                j_old = 0
                for j in range(1, nsl - 1):
                    target_cumul = j * target_tube_mass
                    while j_old < nsl - 2 and cumul_mass + tube_mass[j_old] < target_cumul:
                        cumul_mass += tube_mass[j_old]
                        j_old += 1
                    # Interpolate within the tube
                    remaining = target_cumul - cumul_mass
                    frac = remaining / max(tube_mass[j_old], 1e-15)
                    frac = np.clip(frac, 0.0, 1.0)
                    new_r[j] = sl_r[i, j_old] + frac * (
                        sl_r[i, min(j_old + 1, nsl - 1)] - sl_r[i, j_old]
                    )

                # Relax the update
                sl_r[i] = (
                    (1.0 - cfg.relaxation_factor) * sl_r[i]
                    + cfg.relaxation_factor * new_r
                )

                # Ensure monotonicity
                for j in range(1, nsl):
                    sl_r[i, j] = max(sl_r[i, j], sl_r[i, j - 1] + 1e-6)
                sl_r[i, -1] = tip_r[i]

            # === Step 5: Check convergence ===
            max_dr = np.max(np.abs(sl_r - sl_r_old))
            span_ref = np.mean(tip_r - hub_r)
            residual = max_dr / max(span_ref, 1e-10)
            residual_history.append(float(residual))

            if residual < cfg.convergence_tolerance:
                result.converged = True
                result.n_iterations = iteration + 1
                break

        # === Build result ===
        result.n_iterations = len(residual_history)
        result.residual_history = residual_history
        result.streamline_r = sl_r

        # Compute final static quantities
        P_static = np.zeros_like(Pt)
        T_static = np.zeros_like(Tt)
        V_abs = np.zeros_like(V_m)
        mach = np.zeros_like(V_m)

        for i in range(ns):
            for j in range(nsl):
                V = np.sqrt(V_m[i, j] ** 2 + V_theta[i, j] ** 2)
                V_abs[i, j] = V
                T_s = Tt[i, j] - V**2 / (2.0 * cfg.cp)
                T_s = max(T_s, 100.0)
                T_static[i, j] = T_s
                P_static[i, j] = Pt[i, j] * (T_s / Tt[i, j]) ** (
                    cfg.gamma / (cfg.gamma - 1.0)
                )
                a = np.sqrt(cfg.gamma * cfg.R_gas * T_s)
                mach[i, j] = V / a if a > 1.0 else 0.0

        result.radius = sl_r
        result.axial_coordinate = np.tile(axial[:, None], (1, nsl))
        result.pressure = P_static
        result.temperature = T_static
        result.total_pressure = Pt
        result.total_temperature = Tt
        result.velocity_meridional = V_m
        result.velocity_tangential = V_theta
        result.velocity_absolute = V_abs
        result.flow_angle = alpha
        result.mach_number = mach
        result.density = rho
        result.row_losses = [dict(d) for d in row_losses]

        return result


def _get_blade_param(
    arr: NDArray[np.float64] | None,
    j: int,
    nsl: int,
    default: float,
) -> float:
    """Get a blade parameter at streamline index j, with fallback to default.

    Args:
        arr: Parameter array of length n_streamlines, or None.
        j: Streamline index.
        nsl: Total number of streamlines.
        default: Default value if arr is None.

    Returns:
        Parameter value at streamline j.
    """
    if arr is None:
        return default
    if j < len(arr):
        return float(arr[j])
    return default
