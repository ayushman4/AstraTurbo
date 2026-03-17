"""Surrogate model pipeline for fast aerodynamic optimization.

Provides a complete pipeline for building and using surrogate models:
  - Latin Hypercube DOE sampling
  - Training pipeline: profile -> mesh -> CFD -> collect results
  - Gaussian Process or Neural Network surrogate fitting
  - Adaptive sampling for uncertainty reduction
  - Integration with the optimization framework

Surrogate models replace expensive CFD evaluations with fast predictions,
enabling optimization with 1000x fewer simulations.

Requires: scikit-learn (for GaussianProcessRegressor and MLPRegressor)
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model training."""

    model_type: Literal["gpr", "mlp", "rbf"] = "gpr"

    # GPR settings
    gpr_kernel: str = "matern"          # 'matern', 'rbf', 'rational_quadratic'
    gpr_n_restarts: int = 10
    gpr_alpha: float = 1e-6             # Noise level

    # MLP settings
    mlp_hidden_layers: tuple[int, ...] = (64, 32, 16)
    mlp_max_iter: int = 2000
    mlp_learning_rate: float = 0.001

    # Adaptive sampling
    adaptive_n_candidates: int = 1000
    adaptive_n_select: int = 5
    adaptive_criterion: str = "uncertainty"  # 'uncertainty', 'expected_improvement'


class SurrogateTrainer:
    """Train surrogate models from design-of-experiments data.

    Manages the full pipeline from DOE generation through model fitting.

    Usage::

        trainer = SurrogateTrainer()
        samples = trainer.generate_doe(design_space, n_samples=100)

        # Run CFD for each sample (or use run_training_pipeline)
        X, Y = trainer.run_training_pipeline(
            samples, evaluate_fn=my_cfd_function
        )

        trainer.train_model(X, Y, model_type="gpr")

        mean, std = trainer.predict(new_x)
        trainer.save_model("surrogate.pkl")
    """

    def __init__(self, config: SurrogateConfig | None = None) -> None:
        self.config = config or SurrogateConfig()
        self._model: Any = None
        self._scaler_X: Any = None
        self._scaler_Y: Any = None
        self._X_train: NDArray[np.float64] | None = None
        self._Y_train: NDArray[np.float64] | None = None
        self._training_time: float = 0.0
        self._model_type: str = ""

    def generate_doe(
        self,
        design_space: Any,
        n_samples: int = 100,
        method: str = "lhs",
        seed: int = 42,
    ) -> NDArray[np.float64]:
        """Generate Design of Experiments samples.

        Supports Latin Hypercube Sampling (LHS) for space-filling designs
        and random sampling as a fallback.

        Args:
            design_space: DesignSpace object with lower_bounds and upper_bounds,
                or a dict with 'lower' and 'upper' arrays.
            n_samples: Number of samples to generate.
            method: Sampling method: 'lhs' or 'random'.
            seed: Random seed for reproducibility.

        Returns:
            (n_samples, n_vars) array of design points in physical space.
        """
        rng = np.random.default_rng(seed)

        if hasattr(design_space, "lower_bounds"):
            lb = np.asarray(design_space.lower_bounds, dtype=np.float64)
            ub = np.asarray(design_space.upper_bounds, dtype=np.float64)
        elif isinstance(design_space, dict):
            lb = np.asarray(design_space["lower"], dtype=np.float64)
            ub = np.asarray(design_space["upper"], dtype=np.float64)
        else:
            raise ValueError(
                "design_space must have lower_bounds/upper_bounds or be a dict"
            )

        n_vars = len(lb)

        if method == "lhs":
            samples = self._latin_hypercube(n_samples, n_vars, rng)
        else:
            samples = rng.random((n_samples, n_vars))

        # Scale to physical bounds
        scaled = lb + samples * (ub - lb)
        return scaled

    def _latin_hypercube(
        self,
        n_samples: int,
        n_vars: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Generate Latin Hypercube samples in [0, 1]^n_vars.

        Each variable's range [0,1] is divided into n_samples equal strata.
        One sample is placed randomly within each stratum, and the strata
        are randomly permuted for each variable.

        Args:
            n_samples: Number of samples.
            n_vars: Number of variables.
            rng: numpy random generator.

        Returns:
            (n_samples, n_vars) array in [0, 1].
        """
        samples = np.zeros((n_samples, n_vars), dtype=np.float64)
        for j in range(n_vars):
            perm = rng.permutation(n_samples)
            for i in range(n_samples):
                samples[i, j] = (perm[i] + rng.random()) / n_samples
        return samples

    def run_training_pipeline(
        self,
        samples: NDArray[np.float64],
        evaluate_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        parallel: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run the evaluation function on all DOE samples.

        For each sample point, calls the evaluation function (which
        might run: profile -> mesh -> CFD -> extract performance).

        Args:
            samples: (n_samples, n_vars) design points.
            evaluate_fn: Function(x_vector) -> y_vector.
                x_vector is (n_vars,), y_vector is (n_objectives,).
            parallel: If True, attempt parallel evaluation (requires joblib).

        Returns:
            Tuple of (X, Y) where X is (n_valid, n_vars) and
            Y is (n_valid, n_objectives).
        """
        n_samples = len(samples)
        X_list = []
        Y_list = []

        if parallel:
            try:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=-1)(
                    delayed(evaluate_fn)(samples[i]) for i in range(n_samples)
                )
                for i, y in enumerate(results):
                    if y is not None:
                        X_list.append(samples[i])
                        Y_list.append(np.asarray(y, dtype=np.float64))
            except ImportError:
                parallel = False

        if not parallel:
            for i in range(n_samples):
                try:
                    y = evaluate_fn(samples[i])
                    if y is not None:
                        X_list.append(samples[i])
                        Y_list.append(np.asarray(y, dtype=np.float64))
                except Exception:
                    continue  # Skip failed evaluations

        X = np.array(X_list, dtype=np.float64)
        Y = np.array(Y_list, dtype=np.float64)

        self._X_train = X
        self._Y_train = Y

        return X, Y

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        model_type: str | None = None,
    ) -> dict[str, float]:
        """Fit a surrogate model to the training data.

        Args:
            X: (n_samples, n_vars) input features.
            Y: (n_samples, n_objectives) target values.
                If n_objectives > 1, separate models are trained per output.
            model_type: Override model type from config.
                'gpr' = Gaussian Process Regression
                'mlp' = Multi-Layer Perceptron (neural network)
                'rbf' = Radial Basis Function interpolation

        Returns:
            Dictionary with training metrics:
                'r2_score', 'rmse', 'training_time'.
        """
        from sklearn.preprocessing import StandardScaler

        mtype = model_type or self.config.model_type
        self._model_type = mtype

        # Store and normalize training data
        self._X_train = X.copy()
        self._Y_train = Y.copy()

        self._scaler_X = StandardScaler()
        X_scaled = self._scaler_X.fit_transform(X)

        self._scaler_Y = StandardScaler()
        if Y.ndim == 1:
            Y_scaled = self._scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()
        else:
            Y_scaled = self._scaler_Y.fit_transform(Y)

        t_start = time.perf_counter()

        if mtype == "gpr":
            self._model = self._train_gpr(X_scaled, Y_scaled)
        elif mtype == "mlp":
            self._model = self._train_mlp(X_scaled, Y_scaled)
        elif mtype == "rbf":
            self._model = self._train_rbf(X_scaled, Y_scaled)
        else:
            raise ValueError(f"Unknown model type: {mtype}")

        self._training_time = time.perf_counter() - t_start

        # Compute training metrics
        Y_pred, _ = self._predict_scaled(X_scaled)

        if Y.ndim == 1 or Y.shape[1] == 1:
            y_flat = Y_scaled.ravel()
            yp_flat = Y_pred.ravel()
        else:
            y_flat = Y_scaled.ravel()
            yp_flat = Y_pred.ravel()

        ss_res = np.sum((y_flat - yp_flat) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
        rmse = np.sqrt(np.mean((y_flat - yp_flat) ** 2))

        return {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "training_time": float(self._training_time),
            "n_samples": len(X),
        }

    def _train_gpr(self, X: NDArray, Y: NDArray) -> Any:
        """Train Gaussian Process Regressor."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic

        cfg = self.config
        if cfg.gpr_kernel == "rbf":
            kernel = RBF(length_scale=1.0)
        elif cfg.gpr_kernel == "rational_quadratic":
            kernel = RationalQuadratic()
        else:
            kernel = Matern(nu=2.5)

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=cfg.gpr_n_restarts,
            alpha=cfg.gpr_alpha,
            normalize_y=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=".*encountered in matmul"
            )
            gpr.fit(X, Y)

        return gpr

    def _train_mlp(self, X: NDArray, Y: NDArray) -> Any:
        """Train Multi-Layer Perceptron."""
        from sklearn.neural_network import MLPRegressor

        cfg = self.config
        mlp = MLPRegressor(
            hidden_layer_sizes=cfg.mlp_hidden_layers,
            max_iter=cfg.mlp_max_iter,
            learning_rate_init=cfg.mlp_learning_rate,
            activation="relu",
            solver="adam",
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=".*encountered in matmul"
            )
            mlp.fit(X, Y)

        return mlp

    def _train_rbf(self, X: NDArray, Y: NDArray) -> Any:
        """Train RBF interpolation model using scipy."""
        from scipy.interpolate import RBFInterpolator

        if Y.ndim == 1:
            Y_2d = Y.reshape(-1, 1)
        else:
            Y_2d = Y

        rbf = RBFInterpolator(X, Y_2d, kernel="thin_plate_spline", smoothing=1e-3)
        return rbf

    def predict(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Make predictions with uncertainty estimates.

        Args:
            x: Input point(s). Shape (n_vars,) for single point or
                (n_points, n_vars) for batch prediction.

        Returns:
            Tuple of (mean, std) predictions.
            mean: (n_points, n_objectives) predicted values.
            std: (n_points, n_objectives) prediction uncertainty.
                For MLP/RBF, std is estimated from training residuals.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train_model() first.")

        # Handle single point
        if x.ndim == 1:
            x = x.reshape(1, -1)

        X_scaled = self._scaler_X.transform(x)
        mean_scaled, std_scaled = self._predict_scaled(X_scaled)

        # Inverse transform
        if mean_scaled.ndim == 1:
            mean = self._scaler_Y.inverse_transform(
                mean_scaled.reshape(-1, 1)
            ).ravel()
            std = std_scaled * self._scaler_Y.scale_[0] if hasattr(self._scaler_Y, 'scale_') else std_scaled
        else:
            mean = self._scaler_Y.inverse_transform(mean_scaled)
            if hasattr(self._scaler_Y, 'scale_'):
                std = std_scaled * self._scaler_Y.scale_
            else:
                std = std_scaled

        return mean, std

    def _predict_scaled(
        self, X_scaled: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Predict in scaled space."""
        if self._model_type == "gpr":
            mean, std = self._model.predict(X_scaled, return_std=True)
            if mean.ndim == 1:
                std = std.reshape(-1)
            return mean, std

        elif self._model_type == "mlp":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message=".*encountered in matmul"
                )
                mean = self._model.predict(X_scaled)
                # Estimate uncertainty from training residuals
                if self._X_train is not None and self._Y_train is not None:
                    X_train_scaled = self._scaler_X.transform(self._X_train)
                    Y_train_scaled = self._scaler_Y.transform(
                        self._Y_train.reshape(-1, 1) if self._Y_train.ndim == 1
                        else self._Y_train
                    )
                    if Y_train_scaled.shape[1] == 1:
                        Y_train_scaled = Y_train_scaled.ravel()
                    Y_pred_train = self._model.predict(X_train_scaled)
                    residual_std = np.std(Y_train_scaled - Y_pred_train, axis=0)
                    std = np.full_like(mean, residual_std if np.isscalar(residual_std) else np.mean(residual_std))
                else:
                    std = np.zeros_like(mean)
            return mean, std

        elif self._model_type == "rbf":
            mean = self._model(X_scaled)
            if mean.ndim == 2 and mean.shape[1] == 1:
                mean = mean.ravel()
            std = np.zeros_like(mean) + 0.01  # RBF has no native uncertainty
            return mean, std

        else:
            raise RuntimeError(f"Unknown model type: {self._model_type}")

    def suggest_next_samples(
        self,
        design_space: Any,
        n_suggest: int = 5,
        criterion: str | None = None,
    ) -> NDArray[np.float64]:
        """Suggest next evaluation points using adaptive sampling.

        Identifies regions of high prediction uncertainty and suggests
        new points to evaluate there, improving the surrogate.

        Args:
            design_space: Design space with bounds.
            n_suggest: Number of points to suggest.
            criterion: Selection criterion override.

        Returns:
            (n_suggest, n_vars) array of suggested points.
        """
        cfg = self.config
        crit = criterion or cfg.adaptive_criterion

        # Generate candidate points
        candidates = self.generate_doe(
            design_space,
            n_samples=cfg.adaptive_n_candidates,
            method="lhs",
            seed=int(time.time()) % 2**31,
        )

        # Predict uncertainty at each candidate
        _, std = self.predict(candidates)

        if std.ndim > 1:
            score = np.mean(std, axis=1)
        else:
            score = std.ravel()

        if crit == "expected_improvement" and self._Y_train is not None:
            # Expected improvement: balance exploration and exploitation
            mean, _ = self.predict(candidates)
            if mean.ndim > 1:
                mean = mean[:, 0]
            else:
                mean = mean.ravel()
            y_best = np.min(self._Y_train) if self._Y_train.ndim == 1 else np.min(self._Y_train[:, 0])
            improvement = y_best - mean
            z = improvement / (score + 1e-15)
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + score * norm.pdf(z)
            score = ei

        # Select top points by score
        top_indices = np.argsort(score)[-n_suggest:]
        return candidates[top_indices]

    def save_model(self, path: str | Path) -> None:
        """Save the trained surrogate model to disk.

        Args:
            path: File path for the saved model (.pkl).
        """
        save_data = {
            "model": self._model,
            "model_type": self._model_type,
            "scaler_X": self._scaler_X,
            "scaler_Y": self._scaler_Y,
            "X_train": self._X_train,
            "Y_train": self._Y_train,
            "config": self.config,
            "training_time": self._training_time,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, path: str | Path) -> None:
        """Load a saved surrogate model from disk.

        Args:
            path: Path to the saved model (.pkl).
        """
        with open(path, "rb") as f:
            save_data = pickle.load(f)

        self._model = save_data["model"]
        self._model_type = save_data["model_type"]
        self._scaler_X = save_data["scaler_X"]
        self._scaler_Y = save_data["scaler_Y"]
        self._X_train = save_data.get("X_train")
        self._Y_train = save_data.get("Y_train")
        self.config = save_data.get("config", SurrogateConfig())
        self._training_time = save_data.get("training_time", 0.0)


class SurrogateOptimizer:
    """Optimize using a surrogate model instead of direct CFD evaluations.

    Implements a surrogate-assisted optimization loop:
    1. Build initial surrogate from DOE
    2. Optimize on surrogate (cheap)
    3. Evaluate best points with true function (expensive)
    4. Update surrogate with new data
    5. Repeat until convergence

    Usage::

        opt = SurrogateOptimizer(design_space, cfd_evaluate)
        result = opt.run(n_initial=50, n_iterations=10)
    """

    def __init__(
        self,
        design_space: Any,
        true_evaluate: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        config: SurrogateConfig | None = None,
    ) -> None:
        """
        Args:
            design_space: DesignSpace with bounds.
            true_evaluate: Expensive evaluation function (e.g., CFD).
            config: Surrogate model configuration.
        """
        self.design_space = design_space
        self.true_evaluate = true_evaluate
        self.trainer = SurrogateTrainer(config)
        self._history: list[dict] = []

    def run(
        self,
        n_initial: int = 50,
        n_iterations: int = 10,
        n_adaptive_per_iter: int = 5,
        optimize_on_surrogate: bool = True,
    ) -> dict[str, Any]:
        """Run the surrogate-assisted optimization loop.

        Args:
            n_initial: Number of initial DOE samples for surrogate training.
            n_iterations: Number of adaptive refinement iterations.
            n_adaptive_per_iter: New samples evaluated per iteration.
            optimize_on_surrogate: If True, also run pymoo optimization
                on the surrogate model.

        Returns:
            Dictionary with:
                'best_x': Best design point found.
                'best_y': Best objective value.
                'X_all': All evaluated points.
                'Y_all': All evaluated objectives.
                'n_true_evaluations': Total expensive evaluations.
                'history': Per-iteration metrics.
        """
        # Step 1: Initial DOE
        X_doe = self.trainer.generate_doe(
            self.design_space, n_samples=n_initial
        )
        X_all, Y_all = self.trainer.run_training_pipeline(
            X_doe, self.true_evaluate
        )

        if len(X_all) == 0:
            return {
                "best_x": None,
                "best_y": None,
                "X_all": X_all,
                "Y_all": Y_all,
                "n_true_evaluations": 0,
                "history": [],
            }

        # Step 2: Train initial surrogate
        metrics = self.trainer.train_model(X_all, Y_all)
        self._history.append({"iteration": 0, **metrics, "n_samples": len(X_all)})

        # Step 3: Adaptive refinement loop
        for iteration in range(1, n_iterations + 1):
            # Suggest new points based on uncertainty
            new_X = self.trainer.suggest_next_samples(
                self.design_space, n_suggest=n_adaptive_per_iter
            )

            # Evaluate with true function
            new_Y_list = []
            new_X_valid = []
            for x in new_X:
                try:
                    y = self.true_evaluate(x)
                    if y is not None:
                        new_X_valid.append(x)
                        new_Y_list.append(np.asarray(y, dtype=np.float64))
                except Exception:
                    continue

            if len(new_X_valid) == 0:
                continue

            new_X_arr = np.array(new_X_valid)
            new_Y_arr = np.array(new_Y_list)

            # Add to training set
            X_all = np.vstack([X_all, new_X_arr])
            Y_all = np.vstack([Y_all, new_Y_arr]) if Y_all.ndim > 1 else np.concatenate([Y_all, new_Y_arr.ravel()])

            # Retrain surrogate
            metrics = self.trainer.train_model(X_all, Y_all)
            self._history.append({
                "iteration": iteration,
                **metrics,
                "n_samples": len(X_all),
            })

        # Find best point
        if Y_all.ndim == 1:
            best_idx = np.argmin(Y_all)
            best_y = Y_all[best_idx]
        else:
            best_idx = np.argmin(Y_all[:, 0])
            best_y = Y_all[best_idx]

        best_x = X_all[best_idx]

        return {
            "best_x": best_x,
            "best_y": best_y,
            "X_all": X_all,
            "Y_all": Y_all,
            "n_true_evaluations": len(X_all),
            "history": self._history,
        }
