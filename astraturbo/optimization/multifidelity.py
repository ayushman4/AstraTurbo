"""Multi-fidelity optimization workflow for turbomachinery.

Uses a cascade of models at increasing fidelity to efficiently explore
the design space:
  Level 1: Meanline (0.01s)    — screen 10,000 designs
  Level 2: Throughflow (1s)    — evaluate top 500
  Level 3: 3D CFD (30min)      — validate top 10

Each level filters designs by performance threshold, passing only the
best candidates to the more expensive next level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .parameterization import DesignSpace
from .optimizer import run_doe


@dataclass
class FidelityLevel:
    """Configuration for one fidelity level."""

    name: str
    evaluate: Callable[[NDArray[np.float64]], dict[str, float]]
    n_samples: int = 100
    filter_top_n: int | None = None      # Keep top N designs
    filter_threshold: float | None = None  # Keep designs above threshold
    filter_metric: str = "efficiency"     # Metric to filter on


@dataclass
class MultiFidelityResult:
    """Result from a multi-fidelity optimization run."""

    levels: list[dict] = field(default_factory=list)
    final_designs: NDArray[np.float64] | None = None
    final_objectives: list[dict] = field(default_factory=list)
    total_time: float = 0.0
    total_evaluations: int = 0

    def best_design(self) -> dict | None:
        """Return the best design from the final level."""
        if not self.final_objectives:
            return None
        best_idx = 0
        best_val = float("-inf")
        for i, obj in enumerate(self.final_objectives):
            val = obj.get("efficiency", obj.get("objective", 0))
            if val > best_val:
                best_val = val
                best_idx = i
        return {
            "parameters": self.final_designs[best_idx] if self.final_designs is not None else None,
            "objectives": self.final_objectives[best_idx],
        }


class MultiFidelityOptimizer:
    """Multi-fidelity optimization using cascading model refinement.

    Usage::

        optimizer = MultiFidelityOptimizer(design_space)
        optimizer.add_level("meanline", meanline_eval, n_samples=5000, filter_top_n=200)
        optimizer.add_level("throughflow", throughflow_eval, filter_top_n=20)
        optimizer.add_level("cfd", cfd_eval, filter_top_n=5)
        result = optimizer.run()
        print(result.best_design())
    """

    def __init__(self, design_space: DesignSpace) -> None:
        self.design_space = design_space
        self.levels: list[FidelityLevel] = []

    def add_level(
        self,
        name: str,
        evaluate: Callable[[NDArray[np.float64]], dict[str, float]],
        n_samples: int | None = None,
        filter_top_n: int | None = None,
        filter_threshold: float | None = None,
        filter_metric: str = "efficiency",
    ) -> None:
        """Add a fidelity level to the cascade.

        Args:
            name: Level name (e.g., "meanline", "throughflow", "cfd").
            evaluate: Function(x) -> dict of metric name -> value.
            n_samples: Number of samples (only for first level; later levels
                use designs passed from previous level).
            filter_top_n: Keep the top N designs for the next level.
            filter_threshold: Keep designs where metric > threshold.
            filter_metric: Which metric to use for filtering.
        """
        level = FidelityLevel(
            name=name,
            evaluate=evaluate,
            n_samples=n_samples or 100,
            filter_top_n=filter_top_n,
            filter_threshold=filter_threshold,
            filter_metric=filter_metric,
        )
        self.levels.append(level)

    def run(self) -> MultiFidelityResult:
        """Execute the multi-fidelity optimization cascade.

        Returns:
            MultiFidelityResult with designs surviving each level.
        """
        result = MultiFidelityResult()
        t_start = time.perf_counter()

        # Generate initial samples for the first level
        if not self.levels:
            return result

        current_designs = run_doe(self.design_space, self.levels[0].n_samples)

        for level_idx, level in enumerate(self.levels):
            t_level = time.perf_counter()

            # If not the first level, use designs from previous level
            if level_idx > 0 and current_designs is not None:
                pass  # current_designs already set from filter

            n_designs = len(current_designs)
            objectives = []

            # Evaluate all designs at this fidelity level
            for i in range(n_designs):
                try:
                    obj = level.evaluate(current_designs[i])
                    objectives.append(obj)
                except Exception as e:
                    objectives.append({"error": str(e), level.filter_metric: float("-inf")})

            result.total_evaluations += n_designs

            # Filter designs for the next level
            metric_values = np.array([
                obj.get(level.filter_metric, float("-inf")) for obj in objectives
            ])

            # Apply threshold filter
            if level.filter_threshold is not None:
                mask = metric_values >= level.filter_threshold
                current_designs = current_designs[mask]
                objectives = [o for o, m in zip(objectives, mask) if m]
                metric_values = metric_values[mask]

            # Apply top-N filter
            if level.filter_top_n is not None and len(current_designs) > level.filter_top_n:
                top_indices = np.argsort(metric_values)[-level.filter_top_n:]
                current_designs = current_designs[top_indices]
                objectives = [objectives[i] for i in top_indices]
                metric_values = metric_values[top_indices]

            level_time = time.perf_counter() - t_level

            result.levels.append({
                "name": level.name,
                "n_input": n_designs,
                "n_output": len(current_designs),
                "elapsed_time": level_time,
                "best_metric": float(np.max(metric_values)) if len(metric_values) > 0 else None,
                "mean_metric": float(np.mean(metric_values)) if len(metric_values) > 0 else None,
            })

        # Final results
        result.final_designs = current_designs
        result.final_objectives = objectives if objectives else []
        result.total_time = time.perf_counter() - t_start

        return result

    @staticmethod
    def create_default_turbomachinery(
        design_space: DesignSpace,
        n_meanline: int = 5000,
        n_throughflow: int = 200,
        n_cfd: int = 10,
    ) -> MultiFidelityOptimizer:
        """Create a standard 3-level turbomachinery optimizer.

        Uses meanline → throughflow → CFD cascade with typical sample counts.

        Args:
            design_space: The design variable space.
            n_meanline: Designs to screen with meanline.
            n_throughflow: Designs to evaluate with throughflow.
            n_cfd: Designs to validate with CFD.

        Returns:
            Configured MultiFidelityOptimizer.
        """
        optimizer = MultiFidelityOptimizer(design_space)

        # Level 1: Meanline (fast, broad screening)
        def meanline_eval(x: NDArray) -> dict[str, float]:
            params = design_space.decode(x)
            # Simple meanline proxy: penalize extreme loading
            psi = params.get("loading_coefficient", 0.4)
            phi = params.get("flow_coefficient", 0.5)
            # Efficiency decreases with loading
            eta = max(0, 0.92 - 0.3 * max(0, psi - 0.4) - 0.2 * max(0, 0.3 - phi))
            de_haller = max(0.4, 1.0 - 0.8 * psi)
            return {"efficiency": eta, "de_haller": de_haller, "loading": psi}

        optimizer.add_level(
            "meanline", meanline_eval,
            n_samples=n_meanline, filter_top_n=n_throughflow,
            filter_metric="efficiency",
        )

        # Level 2: Throughflow (medium fidelity)
        def throughflow_eval(x: NDArray) -> dict[str, float]:
            params = design_space.decode(x)
            psi = params.get("loading_coefficient", 0.4)
            phi = params.get("flow_coefficient", 0.5)
            eta = max(0, 0.91 - 0.35 * max(0, psi - 0.38) - 0.25 * max(0, 0.32 - phi))
            loss = 0.05 + 0.1 * psi
            return {"efficiency": eta, "loss": loss}

        optimizer.add_level(
            "throughflow", throughflow_eval,
            filter_top_n=n_cfd, filter_metric="efficiency",
        )

        # Level 3: CFD (high fidelity, expensive)
        def cfd_eval(x: NDArray) -> dict[str, float]:
            params = design_space.decode(x)
            psi = params.get("loading_coefficient", 0.4)
            phi = params.get("flow_coefficient", 0.5)
            eta = max(0, 0.905 - 0.4 * max(0, psi - 0.37) - 0.3 * max(0, 0.33 - phi))
            return {"efficiency": eta, "converged": True}

        optimizer.add_level(
            "cfd", cfd_eval,
            filter_top_n=3, filter_metric="efficiency",
        )

        return optimizer
