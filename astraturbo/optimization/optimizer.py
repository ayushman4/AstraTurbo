"""Optimization driver using pymoo.

Provides a wrapper around pymoo's optimization algorithms for
turbomachinery blade design optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .parameterization import DesignSpace


@dataclass
class OptimizationConfig:
    """Configuration for the optimization run."""

    algorithm: str = "NSGA2"         # NSGA2, NSGA3, DE, PSO
    n_generations: int = 100
    population_size: int = 40
    seed: int = 42
    verbose: bool = True


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    best_x: NDArray[np.float64] | None = None
    best_f: NDArray[np.float64] | None = None
    history: list[dict] = field(default_factory=list)
    n_evaluations: int = 0
    converged: bool = False


class Optimizer:
    """Turbomachinery design optimizer.

    Wraps pymoo algorithms with a simple interface for blade optimization.

    Usage::

        design_space = create_blade_design_space(n_profiles=3)
        optimizer = Optimizer(design_space, evaluate_fn)
        result = optimizer.run(OptimizationConfig(n_generations=50))
    """

    def __init__(
        self,
        design_space: DesignSpace,
        evaluate: Callable[[NDArray[np.float64]], tuple[NDArray, NDArray]],
        n_objectives: int = 1,
        n_constraints: int = 0,
    ) -> None:
        """
        Args:
            design_space: The design variable space.
            evaluate: Function(x) -> (objectives, constraints).
                x is (n_vars,), objectives is (n_obj,), constraints is (n_con,).
            n_objectives: Number of objectives.
            n_constraints: Number of constraints.
        """
        self.design_space = design_space
        self.evaluate = evaluate
        self.n_objectives = n_objectives
        self.n_constraints = n_constraints

    def run(self, config: OptimizationConfig | None = None) -> OptimizationResult:
        """Run the optimization.

        Args:
            config: Optimization configuration.

        Returns:
            OptimizationResult.
        """
        try:
            return self._run_pymoo(config or OptimizationConfig())
        except ImportError:
            return self._run_scipy(config or OptimizationConfig())

    def _run_pymoo(self, config: OptimizationConfig) -> OptimizationResult:
        """Run optimization using pymoo."""
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as pymoo_minimize
        from pymoo.termination import get_termination

        ds = self.design_space
        eval_fn = self.evaluate
        n_obj = self.n_objectives
        n_con = self.n_constraints

        class BladeProblem(ElementwiseProblem):
            def __init__(self_inner):
                super().__init__(
                    n_var=ds.n_vars,
                    n_obj=n_obj,
                    n_ieq_constr=n_con,
                    xl=ds.lower_bounds,
                    xu=ds.upper_bounds,
                )

            def _evaluate(self_inner, x, out, *args, **kwargs):
                obj, con = eval_fn(x)
                out["F"] = obj
                if n_con > 0:
                    out["G"] = con

        problem = BladeProblem()

        algorithm = NSGA2(pop_size=config.population_size)
        termination = get_termination("n_gen", config.n_generations)

        res = pymoo_minimize(
            problem, algorithm, termination,
            seed=config.seed, verbose=config.verbose,
        )

        return OptimizationResult(
            best_x=res.X,
            best_f=res.F,
            n_evaluations=res.algorithm.evaluator.n_eval,
            converged=True,
        )

    def _run_scipy(self, config: OptimizationConfig) -> OptimizationResult:
        """Fallback: run single-objective optimization using scipy."""
        from scipy.optimize import differential_evolution

        ds = self.design_space
        bounds = list(zip(ds.lower_bounds, ds.upper_bounds))

        def objective(x):
            obj, _ = self.evaluate(x)
            return float(obj[0])

        result = differential_evolution(
            objective, bounds,
            maxiter=config.n_generations,
            popsize=config.population_size,
            seed=config.seed,
        )

        return OptimizationResult(
            best_x=result.x,
            best_f=np.array([result.fun]),
            n_evaluations=result.nfev,
            converged=result.success,
        )


def run_doe(
    design_space: DesignSpace,
    n_samples: int = 50,
    method: str = "lhs",
) -> NDArray[np.float64]:
    """Generate Design of Experiments (DOE) samples.

    Args:
        design_space: The design variable space.
        n_samples: Number of samples.
        method: 'lhs' (Latin Hypercube) or 'random'.

    Returns:
        (n_samples, n_vars) array of design points.
    """
    n_vars = design_space.n_vars
    lb = design_space.lower_bounds
    ub = design_space.upper_bounds

    if method == "lhs":
        # Simple LHS implementation
        samples = np.zeros((n_samples, n_vars))
        for j in range(n_vars):
            perm = np.random.permutation(n_samples)
            samples[:, j] = (perm + np.random.random(n_samples)) / n_samples
        # Scale to bounds
        samples = lb + samples * (ub - lb)
    else:
        samples = lb + np.random.random((n_samples, n_vars)) * (ub - lb)

    return samples
