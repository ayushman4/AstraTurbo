"""Parametric design chain for automated turbomachinery workflows.

Connects the full design pipeline: meanline -> profile -> blade -> mesh -> export.
When any parameter changes, downstream steps are automatically re-executed.

Uses AstraTurbo's signal system (property_changed) to detect parameter
modifications and trigger recomputation of dependent stages.

Usage::

    chain = DesignChain()
    chain.set_parameter("cl0", 1.2)  # triggers: profile -> blade -> mesh -> export

    # Parametric sweep
    results = chain.sweep("stagger_angle", start=25, end=45, steps=5)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .signals import property_changed, computation_finished


def _empty_dict() -> dict[str, Any]:
    return {}


def _empty_stage_list() -> list[StageResult]:
    return []


@dataclass
class StageResult:
    """Result from a single pipeline stage execution."""

    stage_name: str = ""
    success: bool = False
    elapsed_time: float = 0.0
    data: dict[str, Any] = field(default_factory=_empty_dict)
    error: str = ""


@dataclass
class ChainResult:
    """Result from a full design chain execution."""

    success: bool = False
    total_time: float = 0.0
    stages: list[StageResult] = field(default_factory=_empty_stage_list)
    parameters: dict[str, Any] = field(default_factory=_empty_dict)

    @property
    def profile_points(self) -> NDArray[np.float64] | None:
        """Get profile coordinates from the chain result."""
        for s in self.stages:
            if s.stage_name == "profile" and "points" in s.data:
                return s.data["points"]
        return None

    @property
    def mesh_block(self) -> NDArray[np.float64] | None:
        """Get mesh block from the chain result."""
        for s in self.stages:
            if s.stage_name == "mesh" and "block" in s.data:
                return s.data["block"]
        return None


class DesignChain:
    """Parametric design chain connecting all turbomachinery design stages.

    The chain manages a set of parameters and a sequence of processing
    stages. When a parameter changes, the chain determines which stages
    need re-execution (from the first affected stage onward) and runs them.

    Stages in order:
        1. meanline  - Compute velocity triangles, work, pressure ratio
        2. profile   - Generate 2D airfoil profiles
        3. blade     - Stack profiles into 3D blade geometry
        4. mesh      - Generate computational mesh
        5. export    - Write output files (CGNS, VTK, etc.)
    """

    # Stage ordering for dependency tracking
    STAGES = ["meanline", "profile", "blade", "mesh", "export"]

    # Map of which parameters affect which stages
    PARAM_STAGE_MAP: dict[str, str] = {
        # Meanline parameters
        "pressure_ratio": "meanline",
        "mass_flow": "meanline",
        "rpm": "meanline",
        "n_stages": "meanline",
        "hub_tip_ratio": "meanline",
        "work_coefficient": "meanline",
        "flow_coefficient": "meanline",
        "degree_of_reaction": "meanline",
        # Profile parameters
        "cl0": "profile",
        "max_thickness": "profile",
        "camber_type": "profile",
        "thickness_type": "profile",
        "sample_rate": "profile",
        # Blade parameters
        "stagger_angle": "blade",
        "chord": "blade",
        "n_profiles": "blade",
        "span": "blade",
        "twist_distribution": "blade",
        # Mesh parameters
        "mesh_ni": "mesh",
        "mesh_nj": "mesh",
        "ogrid_layers": "mesh",
        "first_cell_height": "mesh",
        "grading_ratio": "mesh",
        # Export parameters
        "export_format": "export",
        "export_path": "export",
    }

    # Type alias for stage functions
    _StageFn = Callable[[dict[str, Any], dict[str, "StageResult"]], dict[str, Any]]

    def __init__(self) -> None:
        self._parameters: dict[str, Any] = self._default_parameters()
        self._stage_functions: dict[str, DesignChain._StageFn] = {
            "meanline": self._run_meanline,
            "profile": self._run_profile,
            "blade": self._run_blade,
            "mesh": self._run_mesh,
            "export": self._run_export,
        }
        self._last_result: ChainResult | None = None
        self._stage_results: dict[str, StageResult] = {}
        self._dirty_stages: set[str] = set(self.STAGES)  # All dirty initially
        self._callbacks: list[Callable[[ChainResult], None]] = []

        # Connect to AstraTurbo signal system
        property_changed.connect(self._on_property_changed)

    def _default_parameters(self) -> dict[str, Any]:
        """Return default parameter values."""
        return {
            # Meanline
            "pressure_ratio": 1.5,
            "mass_flow": 10.0,
            "rpm": 10000.0,
            "n_stages": 1,
            "hub_tip_ratio": 0.6,
            "work_coefficient": 0.35,
            "flow_coefficient": 0.55,
            "degree_of_reaction": 0.5,
            # Profile
            "cl0": 1.0,
            "max_thickness": 0.1,
            "camber_type": "naca65",
            "thickness_type": "naca65",
            "sample_rate": 80,
            # Blade
            "stagger_angle": 30.0,
            "chord": 0.05,
            "n_profiles": 3,
            "span": 0.05,
            "twist_distribution": "linear",
            # Mesh
            "mesh_ni": 40,
            "mesh_nj": 20,
            "ogrid_layers": 15,
            "first_cell_height": 1e-5,
            "grading_ratio": 1.2,
            # Export
            "export_format": "vtk",
            "export_path": "",
        }

    def _on_property_changed(self, sender: Any, **kwargs: Any) -> None:
        """Signal handler for AstraTurbo property changes."""
        _ = sender  # Used by blinker dispatch
        name = kwargs.get("name", "")
        if name in self.PARAM_STAGE_MAP:
            value = kwargs.get("value")
            self.set_parameter(name, value, auto_run=False)

    def set_parameter(
        self,
        name: str,
        value: Any,
        auto_run: bool = True,
    ) -> ChainResult | None:
        """Set a design parameter and optionally trigger recomputation.

        Determines which stages are affected by the parameter change
        and marks them (and all downstream stages) as dirty.

        Args:
            name: Parameter name (must be in PARAM_STAGE_MAP or custom).
            value: New parameter value.
            auto_run: If True, automatically re-run the chain.

        Returns:
            ChainResult if auto_run is True, else None.
        """
        self._parameters[name] = value

        # Determine first affected stage
        affected_stage = self.PARAM_STAGE_MAP.get(name, "profile")
        stage_idx = self.STAGES.index(affected_stage) if affected_stage in self.STAGES else 0

        # Mark this stage and all downstream as dirty
        for i in range(stage_idx, len(self.STAGES)):
            self._dirty_stages.add(self.STAGES[i])

        if auto_run:
            return self.run()
        return None

    def set_parameters(
        self,
        params: dict[str, Any],
        auto_run: bool = True,
    ) -> ChainResult | None:
        """Set multiple parameters at once.

        Args:
            params: Dictionary of parameter name -> value.
            auto_run: If True, run the chain after setting all parameters.

        Returns:
            ChainResult if auto_run is True.
        """
        for name, value in params.items():
            self.set_parameter(name, value, auto_run=False)

        if auto_run:
            return self.run()
        return None

    def get_parameter(self, name: str) -> Any:
        """Get current value of a parameter."""
        return self._parameters.get(name)

    def get_all_parameters(self) -> dict[str, Any]:
        """Get a copy of all current parameters."""
        return dict(self._parameters)

    def on_result(self, callback: Callable[[ChainResult], None]) -> None:
        """Register a callback for chain completion.

        Args:
            callback: Function called with ChainResult after each run.
        """
        self._callbacks.append(callback)

    def run(self) -> ChainResult:
        """Execute the design chain, running only dirty stages.

        Returns:
            ChainResult with all stage results.
        """
        t_start = time.perf_counter()
        result = ChainResult(parameters=dict(self._parameters))

        for stage_name in self.STAGES:
            if stage_name not in self._dirty_stages:
                # Use cached result
                if stage_name in self._stage_results:
                    result.stages.append(self._stage_results[stage_name])
                continue

            # Run the stage
            stage_fn = self._stage_functions.get(stage_name)
            if stage_fn is None:
                continue

            t_stage = time.perf_counter()
            try:
                stage_data = stage_fn(self._parameters, self._stage_results)
                stage_result = StageResult(
                    stage_name=stage_name,
                    success=True,
                    elapsed_time=time.perf_counter() - t_stage,
                    data=stage_data,
                )
            except Exception as e:
                stage_result = StageResult(
                    stage_name=stage_name,
                    success=False,
                    elapsed_time=time.perf_counter() - t_stage,
                    error=str(e),
                )
                # If a stage fails, mark all downstream as failed too
                result.stages.append(stage_result)
                self._stage_results[stage_name] = stage_result
                break

            self._stage_results[stage_name] = stage_result
            self._dirty_stages.discard(stage_name)
            result.stages.append(stage_result)

        result.total_time = time.perf_counter() - t_start
        result.success = all(s.success for s in result.stages)
        self._last_result = result

        # Fire signal
        computation_finished.send(self, result=result)

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception:
                pass

        return result

    def sweep(
        self,
        parameter_name: str,
        start: float,
        end: float,
        steps: int = 5,
    ) -> list[ChainResult]:
        """Perform a parametric sweep over a single parameter.

        Runs the full design chain for each value of the swept parameter,
        returning a list of results for post-processing.

        Args:
            parameter_name: Name of the parameter to sweep.
            start: Start value.
            end: End value.
            steps: Number of evaluation points.

        Returns:
            List of ChainResult, one per sweep point.
        """
        values = np.linspace(start, end, steps)
        results = []

        # Save original value
        original_value = self._parameters.get(parameter_name)

        for val in values:
            # Reset all stages as dirty for clean evaluation
            self._dirty_stages = set(self.STAGES)
            self._stage_results.clear()

            self._parameters[parameter_name] = float(val)
            result = self.run()
            results.append(result)

        # Restore original value
        if original_value is not None:
            self._parameters[parameter_name] = original_value

        return results

    def multi_sweep(
        self,
        sweeps: dict[str, tuple[float, float, int]],
    ) -> list[ChainResult]:
        """Perform a full-factorial sweep over multiple parameters.

        Args:
            sweeps: Dict of parameter_name -> (start, end, steps).

        Returns:
            List of ChainResult for all parameter combinations.
        """
        # Build grid of parameter values
        param_names = list(sweeps.keys())
        param_values = [
            np.linspace(s[0], s[1], s[2]) for s in sweeps.values()
        ]

        # Create meshgrid for all combinations
        grids = np.meshgrid(*param_values, indexing="ij")
        flat_grids = [g.ravel() for g in grids]
        n_total = len(flat_grids[0])

        results = []
        for i in range(n_total):
            self._dirty_stages = set(self.STAGES)
            self._stage_results.clear()

            for j, name in enumerate(param_names):
                self._parameters[name] = float(flat_grids[j][i])

            result = self.run()
            results.append(result)

        return results

    @property
    def last_result(self) -> ChainResult | None:
        """Get the most recent chain execution result."""
        return self._last_result

    # ── Stage implementations ──────────────────────────────────

    def _run_meanline(
        self,
        params: dict[str, Any],
        prev: dict[str, StageResult],
    ) -> dict[str, Any]:
        """Meanline analysis: compute velocity triangles and thermodynamics."""
        _ = prev  # Meanline is the first stage, no predecessors
        gamma = 1.4
        cp = 1004.5
        R_gas = 287.058
        T01 = 288.15
        P01 = 101325.0

        pr = params["pressure_ratio"]
        rpm = params["rpm"]
        htr = params["hub_tip_ratio"]
        psi = params["work_coefficient"]
        phi = params["flow_coefficient"]
        _ = params["degree_of_reaction"]  # Reserved for future radial equilibrium
        _ = params["mass_flow"]  # Reserved for mass flow check

        # Isentropic temperature rise for the given PR
        T_ratio = pr ** ((gamma - 1.0) / gamma)
        T02s = T01 * T_ratio
        delta_T0s = T02s - T01

        # Assume efficiency
        eta = 0.88
        delta_T0 = delta_T0s / eta
        T02 = T01 + delta_T0

        # Work per unit mass
        work = cp * delta_T0

        # Blade speed from work coefficient: psi = delta_h0 / U^2
        U = np.sqrt(work / psi) if psi > 1e-10 else 300.0

        # Axial velocity from flow coefficient: phi = V_ax / U
        V_ax = phi * U

        # Radii
        omega = rpm * 2.0 * np.pi / 60.0
        r_mean = U / omega if omega > 1e-10 else 0.3

        r_tip = r_mean / np.sqrt(0.5 * (1.0 + htr**2))
        r_hub = htr * r_tip

        # Annulus area
        area = np.pi * (r_tip**2 - r_hub**2)

        # Inlet density
        rho_inlet = P01 / (R_gas * T01)

        # Flow angles from reaction
        # R = 1 - (tan(alpha1) + tan(alpha2)) / (2*U/V_ax)
        # With zero inlet swirl (alpha1=0):
        # delta_V_theta = work / U
        delta_V_theta = work / U if abs(U) > 1e-10 else 0.0
        alpha2 = np.degrees(np.arctan2(delta_V_theta, V_ax))
        beta1 = np.degrees(np.arctan2(-U, V_ax))
        beta2 = np.degrees(np.arctan2(delta_V_theta - U, V_ax))

        return {
            "work": work,
            "U": U,
            "V_ax": V_ax,
            "r_mean": r_mean,
            "r_hub": r_hub,
            "r_tip": r_tip,
            "omega": omega,
            "area": area,
            "rho_inlet": rho_inlet,
            "delta_T0": delta_T0,
            "T02": T02,
            "P02": P01 * pr,
            "alpha1": 0.0,
            "alpha2": alpha2,
            "beta1": beta1,
            "beta2": beta2,
            "delta_V_theta": delta_V_theta,
            "efficiency": eta,
        }

    def _run_profile(
        self,
        params: dict[str, Any],
        prev: dict[str, StageResult],
    ) -> dict[str, Any]:
        """Generate 2D airfoil profile."""
        _ = prev  # Profile doesn't depend on meanline results yet
        from ..camberline import create_camberline
        from ..thickness import create_thickness
        from ..profile import Superposition

        camber_type = params.get("camber_type", "naca65")
        thickness_type = params.get("thickness_type", "naca65")
        cl0 = params.get("cl0", 1.0)
        max_thickness = params.get("max_thickness", 0.1)
        sample_rate = params.get("sample_rate", 80)

        # Create camber line
        camber_kwargs = {}
        if camber_type == "naca65":
            camber_kwargs["cl0"] = cl0

        camber = create_camberline(camber_type, **camber_kwargs)
        camber.sample_rate = sample_rate

        # Create thickness distribution
        thickness_kwargs = {"max_thickness": max_thickness}
        thickness_dist = create_thickness(thickness_type, **thickness_kwargs)
        thickness_dist.sample_rate = sample_rate

        # Create profile
        profile = Superposition(
            camber_line=camber,
            thickness_distribution=thickness_dist,
        )

        points = profile.as_array()
        upper = profile.upper_surface()
        lower = profile.lower_surface()

        return {
            "points": points,
            "upper": upper,
            "lower": lower,
            "profile": profile,
            "camber": camber.as_array(),
        }

    def _run_blade(
        self,
        params: dict[str, Any],
        prev: dict[str, StageResult],
    ) -> dict[str, Any]:
        """Stack profiles into 3D blade."""
        profile_result = prev.get("profile")
        if profile_result is None or not profile_result.success:
            raise ValueError("Profile stage must succeed before blade stage")

        profile_pts = profile_result.data["points"]
        stagger = np.radians(params.get("stagger_angle", 30.0))
        chord = params.get("chord", 0.05)
        span = params.get("span", 0.05)
        n_profiles = params.get("n_profiles", 3)

        # Scale profile to chord
        scaled = profile_pts.copy() * chord

        # Rotate by stagger angle
        cos_s = np.cos(stagger)
        sin_s = np.sin(stagger)
        rotated = np.zeros_like(scaled)
        rotated[:, 0] = scaled[:, 0] * cos_s - scaled[:, 1] * sin_s
        rotated[:, 1] = scaled[:, 0] * sin_s + scaled[:, 1] * cos_s

        # Stack along span (z-direction)
        spanwise_positions = np.linspace(0, span, n_profiles)
        profiles_3d = []
        for z in spanwise_positions:
            profile_3d = np.column_stack([
                rotated[:, 0],
                rotated[:, 1],
                np.full(len(rotated), z),
            ])
            profiles_3d.append(profile_3d)

        return {
            "profiles_3d": profiles_3d,
            "spanwise_positions": spanwise_positions,
            "chord": chord,
            "span": span,
            "stagger_deg": params.get("stagger_angle", 30.0),
            "n_profiles": n_profiles,
        }

    def _run_mesh(
        self,
        params: dict[str, Any],
        prev: dict[str, StageResult],
    ) -> dict[str, Any]:
        """Generate computational mesh."""
        from ..mesh.transfinite import tfi_2d
        from ..mesh.quality import mesh_quality_report

        ni = params.get("mesh_ni", 40)
        nj = params.get("mesh_nj", 20)

        profile_result = prev.get("profile")
        if profile_result is None or not profile_result.success:
            raise ValueError("Profile stage must succeed before mesh stage")

        upper: NDArray[np.float64] = profile_result.data["upper"]
        lower: NDArray[np.float64] = profile_result.data["lower"]

        # Create a simple passage mesh around the profile
        chord = params.get("chord", 0.05)
        pitch = chord * 0.8  # Typical solidity ~1.25

        # Resample upper and lower to ni points
        t_upper = np.linspace(0, 1, len(upper))
        t_lower = np.linspace(0, 1, len(lower))
        t_new = np.linspace(0, 1, ni)

        upper_resampled = np.zeros((ni, 2))
        lower_resampled = np.zeros((ni, 2))
        for d in range(2):
            upper_resampled[:, d] = np.interp(t_new, t_upper, upper[:, d])
            lower_resampled[:, d] = np.interp(t_new, t_lower, lower[:, d])

        # Build boundaries for TFI: blade-to-blade mesh
        bottom: NDArray[np.float64] = lower_resampled * chord
        top: NDArray[np.float64] = np.column_stack([
            upper_resampled[:, 0] * chord,
            upper_resampled[:, 1] * chord + pitch,
        ])
        left: NDArray[np.float64] = np.column_stack([
            np.full(nj, float(bottom[0, 0])),
            np.linspace(float(bottom[0, 1]), float(top[0, 1]), nj),
        ])
        right: NDArray[np.float64] = np.column_stack([
            np.full(nj, float(bottom[-1, 0])),
            np.linspace(float(bottom[-1, 1]), float(top[-1, 1]), nj),
        ])

        # Fix corners
        left[0] = bottom[0]
        left[-1] = top[0]
        right[0] = bottom[-1]
        right[-1] = top[-1]

        block = tfi_2d(bottom, top, left, right)
        quality: dict[str, Any] = mesh_quality_report(block)

        return {
            "block": block,
            "quality": quality,
            "ni": ni,
            "nj": nj,
        }

    def _run_export(
        self,
        params: dict[str, Any],
        prev: dict[str, StageResult],
    ) -> dict[str, Any]:
        """Export results to file (if path is specified)."""
        export_path = params.get("export_path", "")
        export_format = params.get("export_format", "vtk")

        mesh_result = prev.get("mesh")
        if mesh_result is None or not mesh_result.success:
            return {"exported": False, "reason": "No mesh to export"}

        if not export_path:
            return {"exported": False, "reason": "No export path specified"}

        block = mesh_result.data.get("block")
        if block is None:
            return {"exported": False, "reason": "No mesh block data"}

        try:
            from ..export import export_structured_as_quads
            export_structured_as_quads(export_path, [block], file_format=export_format)
            return {"exported": True, "path": export_path, "format": export_format}
        except Exception as e:
            return {"exported": False, "reason": str(e)}
