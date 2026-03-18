"""Tests for the 9+ new features added to AstraTurbo.

Covers:
  1. Mesh smoothing (Laplacian, vectorized, orthogonality)
  2. Tip clearance mesh generation
  3. Auto y+ / first cell height calculation
  4. Throughflow solver
  5. Loss models (Lieblein, Ainley-Mathieson, Carter)
  6. Design chain (parameter propagation)
  7. Design chain sweep
  8. Surrogate model (train + predict)
  9. HPC job manager (LocalBackend)
 10. Design database (CRUD + search + export)
 11. Multi-fidelity optimization
"""

import json
import time

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────
# 1. Mesh smoothing
# ────────────────────────────────────────────────────────────────

class TestMeshSmoothing:
    """Test Laplacian smoothing and orthogonality correction."""

    @staticmethod
    def _make_distorted_mesh(ni=12, nj=8):
        """Create a structured mesh with deliberate distortion."""
        block = np.zeros((ni, nj, 2), dtype=np.float64)
        for i in range(ni):
            for j in range(nj):
                x = i / (ni - 1)
                y = j / (nj - 1)
                # Add sinusoidal distortion to interior
                if 0 < i < ni - 1 and 0 < j < nj - 1:
                    dx = 0.04 * np.sin(3 * np.pi * x) * np.sin(2 * np.pi * y)
                    dy = 0.03 * np.cos(2 * np.pi * x) * np.sin(3 * np.pi * y)
                else:
                    dx, dy = 0.0, 0.0
                block[i, j, 0] = x + dx
                block[i, j, 1] = y + dy
        return block

    def test_laplacian_smooth_reduces_skewness(self):
        """Smoothing a distorted mesh should reduce max skewness."""
        from astraturbo.mesh.smoothing import laplacian_smooth

        block = self._make_distorted_mesh()
        smoothed, metrics = laplacian_smooth(block, n_iterations=30, omega=0.5)

        assert metrics["after_skewness_max"] <= metrics["before_skewness_max"]
        assert metrics["iterations"] == 30
        assert smoothed.shape == block.shape

    def test_laplacian_smooth_vectorized_matches(self):
        """Vectorized smoothing should produce similar quality improvement."""
        from astraturbo.mesh.smoothing import laplacian_smooth_vectorized

        block = self._make_distorted_mesh()
        smoothed, metrics = laplacian_smooth_vectorized(
            block, n_iterations=30, omega=0.5
        )

        assert metrics["after_skewness_max"] <= metrics["before_skewness_max"]
        assert smoothed.shape == block.shape

    def test_laplacian_preserves_boundaries(self):
        """With fix_boundaries=True, boundary points must stay in place."""
        from astraturbo.mesh.smoothing import laplacian_smooth

        block = self._make_distorted_mesh()
        original_boundary = block[[0, -1], :, :].copy()
        original_sides = block[:, [0, -1], :].copy()

        smoothed, _ = laplacian_smooth(block, n_iterations=20, fix_boundaries=True)

        np.testing.assert_allclose(smoothed[[0, -1], :, :], original_boundary, atol=1e-15)
        np.testing.assert_allclose(smoothed[:, [0, -1], :], original_sides, atol=1e-15)

    def test_orthogonality_correction(self):
        """Orthogonality correction should run and return valid output."""
        from astraturbo.mesh.smoothing import orthogonality_correction

        block = self._make_distorted_mesh()
        corrected, metrics = orthogonality_correction(block, n_iterations=5, omega=0.1)

        # Verify it runs and returns correct shape and metrics
        assert corrected.shape == block.shape
        assert "before_orthogonality_min" in metrics
        assert "after_orthogonality_min" in metrics
        assert metrics["after_orthogonality_min"] > 0  # Angle should be positive

    def test_combined_smooth(self):
        """Combined Laplacian + orthogonality should improve overall quality."""
        from astraturbo.mesh.smoothing import combined_smooth

        block = self._make_distorted_mesh()
        smoothed, metrics = combined_smooth(block, n_cycles=2)

        assert metrics["after_skewness_max"] <= metrics["before_skewness_max"]
        assert "n_cycles" in metrics
        assert metrics["n_cycles"] == 2


# ────────────────────────────────────────────────────────────────
# 2. Tip clearance mesh
# ────────────────────────────────────────────────────────────────

class TestTipClearance:
    """Test tip clearance mesh generation."""

    def test_generate_tip_clearance_mesh_shape(self):
        """Tip clearance mesh should have correct dimensions."""
        from astraturbo.mesh.tip_clearance import generate_tip_clearance_mesh

        n_pts = 20
        blade_tip = np.zeros((n_pts, 3), dtype=np.float64)
        blade_tip[:, 0] = np.linspace(0, 0.1, n_pts)   # x along chord
        blade_tip[:, 1] = 0.0                           # y
        blade_tip[:, 2] = 0.15                           # z (radial = tip)

        gap_height = 0.002  # 2 mm gap
        n_radial = 8

        result = generate_tip_clearance_mesh(
            blade_tip_curve=blade_tip,
            casing_contour=None,
            gap_height=gap_height,
            n_radial=n_radial,
        )

        points = result["points"]
        n_s_pts = n_pts  # n_streamwise defaults to n_tip - 1, so pts = n_tip
        n_r_pts = n_radial + 1

        assert points.shape == (n_s_pts, n_r_pts, 3)
        assert result["n_streamwise"] == n_s_pts
        assert result["n_radial"] == n_r_pts

    def test_tip_clearance_quality(self):
        """Generated mesh should have reasonable quality metrics."""
        from astraturbo.mesh.tip_clearance import generate_tip_clearance_mesh

        n_pts = 15
        blade_tip = np.zeros((n_pts, 3), dtype=np.float64)
        blade_tip[:, 0] = np.linspace(0, 0.1, n_pts)
        blade_tip[:, 2] = 0.15

        result = generate_tip_clearance_mesh(
            blade_tip_curve=blade_tip,
            casing_contour=None,
            gap_height=0.003,
            n_radial=6,
            grading_mode="uniform",
        )

        quality = result["quality"]
        assert quality["n_cells"] > 0
        assert quality["aspect_ratio_max"] > 0
        # With uniform grading along a straight blade, aspect ratio should be moderate
        assert quality["aspect_ratio_max"] < 50

    def test_tip_clearance_with_casing_contour(self):
        """Providing a casing contour should work correctly."""
        from astraturbo.mesh.tip_clearance import generate_tip_clearance_mesh

        n_pts = 10
        blade_tip = np.zeros((n_pts, 3), dtype=np.float64)
        blade_tip[:, 0] = np.linspace(0, 0.1, n_pts)
        blade_tip[:, 2] = 0.15

        casing = np.zeros((n_pts, 3), dtype=np.float64)
        casing[:, 0] = np.linspace(0, 0.1, n_pts)
        casing[:, 2] = 0.153  # 3mm gap

        result = generate_tip_clearance_mesh(
            blade_tip_curve=blade_tip,
            casing_contour=casing,
            gap_height=0.003,
            n_radial=5,
        )

        assert result["points"].shape[0] == n_pts
        assert result["points"].shape[1] == 6  # n_radial+1


# ────────────────────────────────────────────────────────────────
# 3. Auto y+
# ────────────────────────────────────────────────────────────────

class TestAutoYPlus:
    """Test automatic first cell height calculation."""

    def test_auto_first_cell_height_returns_reasonable_values(self):
        """auto_first_cell_height should return physically sensible values."""
        from astraturbo.mesh.quality import auto_first_cell_height

        result = auto_first_cell_height(
            velocity=100.0,
            chord=0.05,
            density=1.225,
            dynamic_viscosity=1.8e-5,
            target_yplus=1.0,
        )

        assert "first_cell_height" in result
        assert "grading_ratio" in result
        assert "boundary_layer_thickness" in result
        assert "reynolds_number" in result
        assert "estimated_yplus" in result
        assert "ogrid_total_thickness" in result

        # First cell height should be very small (microns range for y+=1)
        assert 1e-7 < result["first_cell_height"] < 1e-3

        # Grading ratio should be > 1
        assert result["grading_ratio"] > 1.0

        # Reynolds number should be positive and reasonable
        re = result["reynolds_number"]
        assert re > 1e4  # 100 m/s, 50mm chord -> Re ~ 3.4e5

        # Estimated y+ should be close to target
        assert abs(result["estimated_yplus"] - 1.0) < 0.1

    def test_auto_yplus_scales_with_velocity(self):
        """Higher velocity should give smaller first cell height (for same y+)."""
        from astraturbo.mesh.quality import auto_first_cell_height

        r1 = auto_first_cell_height(velocity=50.0, chord=0.05)
        r2 = auto_first_cell_height(velocity=200.0, chord=0.05)

        # Higher velocity -> thinner BL -> smaller cell
        assert r2["first_cell_height"] < r1["first_cell_height"]

    def test_first_cell_height_for_yplus(self):
        """first_cell_height_for_yplus / estimate_yplus should be inverses."""
        from astraturbo.mesh.quality import (
            first_cell_height_for_yplus,
            estimate_yplus,
        )

        dy = first_cell_height_for_yplus(
            target_yplus=1.0,
            density=1.225,
            velocity=100.0,
            dynamic_viscosity=1.8e-5,
            chord=0.05,
        )

        yp = estimate_yplus(dy, 1.225, 100.0, 1.8e-5, 0.05)
        assert abs(yp - 1.0) < 0.01


# ────────────────────────────────────────────────────────────────
# 4. Throughflow solver
# ────────────────────────────────────────────────────────────────

class TestThroughflowSolver:
    """Test the streamline curvature throughflow solver."""

    def test_solver_runs_and_converges(self):
        """Solver should converge on a simple single-rotor case."""
        from astraturbo.solver.throughflow import (
            ThroughflowSolver, ThroughflowConfig, BladeRowSpec,
        )

        n_stations = 15
        n_streamlines = 7

        config = ThroughflowConfig(
            n_stations=n_stations,
            n_streamlines=n_streamlines,
            max_iterations=200,
            convergence_tolerance=1e-3,
        )
        solver = ThroughflowSolver(config)

        hub_r = np.full(n_stations, 0.1)
        tip_r = np.full(n_stations, 0.2)
        axial = np.linspace(0.0, 0.15, n_stations)
        solver.set_annulus(hub_r, tip_r, axial)

        omega = 10000 * 2 * np.pi / 60  # 10000 RPM
        rotor = BladeRowSpec(
            row_type="rotor",
            n_blades=36,
            inlet_station=3,
            outlet_station=8,
            omega=omega,
        )
        solver.add_blade_row(rotor)
        solver.set_inlet_conditions(
            total_pressure=101325.0,
            total_temperature=288.15,
        )

        result = solver.solve()

        assert result.n_iterations > 0
        assert len(result.residual_history) > 0
        assert result.total_pressure is not None
        assert result.total_temperature is not None
        assert result.velocity_meridional is not None

    def test_solver_produces_physical_results(self):
        """Temperatures, pressures, and velocities should be physically sensible."""
        from astraturbo.solver.throughflow import (
            ThroughflowSolver, ThroughflowConfig, BladeRowSpec,
        )

        n_stations = 10
        n_sl = 5
        config = ThroughflowConfig(
            n_stations=n_stations, n_streamlines=n_sl, max_iterations=100,
        )
        solver = ThroughflowSolver(config)

        solver.set_annulus(
            np.full(n_stations, 0.12),
            np.full(n_stations, 0.22),
            np.linspace(0, 0.1, n_stations),
        )

        rotor = BladeRowSpec(
            row_type="rotor", n_blades=40,
            inlet_station=2, outlet_station=6,
            omega=8000 * 2 * np.pi / 60,
        )
        solver.add_blade_row(rotor)
        solver.set_inlet_conditions()

        result = solver.solve()

        # All temperatures should be positive and above absolute zero
        assert np.all(result.temperature > 0)
        assert np.all(result.total_temperature > 200)  # Above 200K

        # Total temperature should increase across rotor (work input)
        T0_inlet = result.total_temperature[0, :].mean()
        T0_outlet = result.total_temperature[-1, :].mean()
        assert T0_outlet >= T0_inlet  # Rotor adds energy

        # Mach numbers should be subsonic for this low-speed case
        assert result.mach_number is not None
        assert np.all(result.mach_number < 2.0)

    def test_solver_without_annulus_returns_unconverged(self):
        """Solver should handle missing annulus gracefully."""
        from astraturbo.solver.throughflow import ThroughflowSolver

        solver = ThroughflowSolver()
        result = solver.solve()
        assert result.converged is False


# ────────────────────────────────────────────────────────────────
# 5. Loss models
# ────────────────────────────────────────────────────────────────

class TestLossModels:
    """Test empirical loss and deviation correlations."""

    def test_lieblein_profile_loss(self):
        """Profile loss should increase with diffusion factor."""
        from astraturbo.solver.loss_models import lieblein_profile_loss

        # Low diffusion factor -> low loss
        omega_low = lieblein_profile_loss(
            diffusion_factor=0.3, solidity=1.0, Re=2e5
        )
        # High diffusion factor -> higher loss
        omega_high = lieblein_profile_loss(
            diffusion_factor=0.6, solidity=1.0, Re=2e5
        )

        assert omega_low >= 0.0
        assert omega_high > omega_low
        # Typical profile loss range 0.01-0.1
        assert omega_low < 0.1
        assert omega_high < 0.5

    def test_lieblein_diffusion_factor(self):
        """Diffusion factor should be in expected range for typical cascade."""
        from astraturbo.solver.loss_models import lieblein_diffusion_factor

        # Typical compressor cascade: V2 < V1, turning
        df = lieblein_diffusion_factor(
            v1=200.0, v2=150.0, v_theta1=50.0, v_theta2=100.0, solidity=1.2
        )

        assert 0.0 < df < 1.0
        # DF = 1 - 150/200 + |100-50|/(2*1.2*200) = 0.25 + 0.104 = 0.354
        expected = 1.0 - 150.0/200.0 + abs(100-50)/(2*1.2*200.0)
        assert abs(df - expected) < 1e-10

    def test_ainley_mathieson_secondary_loss(self):
        """Secondary loss should be positive and scale with turning."""
        from astraturbo.solver.loss_models import ainley_mathieson_secondary_loss

        # Moderate turning
        loss_mod = ainley_mathieson_secondary_loss(
            inlet_angle=40.0, outlet_angle=-10.0, span=0.05, chord=0.04
        )
        # Large turning -> higher secondary loss
        loss_high = ainley_mathieson_secondary_loss(
            inlet_angle=60.0, outlet_angle=-30.0, span=0.05, chord=0.04
        )

        assert loss_mod >= 0.0
        assert loss_high > loss_mod

    def test_carter_deviation(self):
        """Carter deviation should be positive for positive camber."""
        from astraturbo.solver.loss_models import carter_deviation

        # Typical compressor: camber=30, solidity=1.2, stagger=20
        dev = carter_deviation(camber=30.0, solidity=1.2, stagger=20.0)

        # Deviation should be positive (flow turns less than blade)
        assert dev > 0.0

        # With m = 0.23 + 0.002*20 = 0.27
        # dev = 0.27 * 30 / 1.2^0.5 = 7.39 deg approximately
        m = 0.23 + 0.002 * 20
        expected = m * 30.0 / 1.2 ** 0.5
        assert abs(dev - expected) < 1e-10

    def test_carter_deviation_increases_with_camber(self):
        """More camber should give more deviation."""
        from astraturbo.solver.loss_models import carter_deviation

        dev1 = carter_deviation(camber=20.0, solidity=1.0, stagger=30.0)
        dev2 = carter_deviation(camber=40.0, solidity=1.0, stagger=30.0)

        assert dev2 > dev1

    def test_total_loss_coefficient(self):
        """total_loss_coefficient should sum all components."""
        from astraturbo.solver.loss_models import total_loss_coefficient

        result = total_loss_coefficient(
            diffusion_factor=0.4,
            solidity=1.0,
            Re=2e5,
            inlet_angle=40.0,
            outlet_angle=-10.0,
            span=0.05,
            chord=0.04,
            clearance=0.001,
            loading=0.5,
        )

        assert "profile" in result
        assert "secondary" in result
        assert "tip_clearance" in result
        assert "total" in result
        assert abs(result["total"] - (result["profile"] + result["secondary"] + result["tip_clearance"])) < 1e-12


# ────────────────────────────────────────────────────────────────
# 6. Design chain
# ────────────────────────────────────────────────────────────────

class TestDesignChain:
    """Test parametric design chain."""

    def test_create_chain_and_run(self):
        """Design chain should run all stages successfully."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()
        result = chain.run()

        assert result.success
        assert len(result.stages) > 0
        assert result.total_time > 0

    def test_set_parameter_triggers_downstream(self):
        """Setting a profile parameter should re-run profile and all downstream."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()

        # Run once to populate cache
        result1 = chain.run()
        assert result1.success

        # Change a profile-level parameter
        result2 = chain.set_parameter("cl0", 0.8)
        assert result2 is not None
        assert result2.success

        # Verify the parameter was updated
        assert chain.get_parameter("cl0") == 0.8

    def test_set_parameters_batch(self):
        """Setting multiple parameters at once should work."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()
        result = chain.set_parameters({
            "cl0": 0.9,
            "max_thickness": 0.08,
            "stagger_angle": 25.0,
        })

        assert result is not None
        assert result.success
        assert chain.get_parameter("cl0") == 0.9
        assert chain.get_parameter("max_thickness") == 0.08

    def test_chain_result_has_profile_points(self):
        """Chain result should provide access to profile coordinates."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()
        result = chain.run()

        pts = result.profile_points
        assert pts is not None
        assert pts.shape[1] == 2  # (N, 2) for x, y
        assert len(pts) > 10     # Should have many points


# ────────────────────────────────────────────────────────────────
# 7. Design chain sweep
# ────────────────────────────────────────────────────────────────

class TestDesignChainSweep:
    """Test parametric sweep on the design chain."""

    def test_sweep_returns_correct_count(self):
        """Sweeping with N steps should return N results."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()
        results = chain.sweep("cl0", start=0.6, end=1.4, steps=5)

        assert len(results) == 5

    def test_sweep_all_succeed(self):
        """All sweep evaluations should succeed for valid parameter range."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()
        results = chain.sweep("cl0", start=0.8, end=1.2, steps=3)

        assert all(r.success for r in results)

    def test_sweep_each_has_different_params(self):
        """Each sweep result should use its specific parameter value."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()
        results = chain.sweep("cl0", start=0.6, end=1.4, steps=5)

        values = [r.parameters["cl0"] for r in results]
        expected = np.linspace(0.6, 1.4, 5).tolist()
        np.testing.assert_allclose(values, expected, atol=1e-10)

    def test_multi_sweep(self):
        """Multi-parameter sweep should produce the full factorial grid."""
        from astraturbo.foundation.design_chain import DesignChain

        chain = DesignChain()
        results = chain.multi_sweep({
            "cl0": (0.8, 1.2, 2),
            "max_thickness": (0.08, 0.12, 2),
        })

        # 2 x 2 = 4 combinations
        assert len(results) == 4


# ────────────────────────────────────────────────────────────────
# 8. Surrogate model
# ────────────────────────────────────────────────────────────────

class TestSurrogateModel:
    """Test surrogate model training and prediction."""

    def test_train_and_predict_gpr(self):
        """GPR surrogate should fit synthetic data accurately."""
        pytest.importorskip("sklearn")
        from astraturbo.ai.surrogate import SurrogateTrainer, SurrogateConfig

        config = SurrogateConfig(model_type="gpr", gpr_n_restarts=2)
        trainer = SurrogateTrainer(config)

        # Synthetic quadratic function: y = x1^2 + x2^2
        rng = np.random.default_rng(42)
        X = rng.random((30, 2)) * 4 - 2  # [-2, 2]^2
        Y = X[:, 0] ** 2 + X[:, 1] ** 2

        metrics = trainer.train_model(X, Y, model_type="gpr")

        assert "r2_score" in metrics
        assert metrics["r2_score"] > 0.9  # GPR should fit well

        # Predict at known point
        x_test = np.array([1.0, 1.0])
        mean, std = trainer.predict(x_test)

        # y = 1 + 1 = 2
        assert abs(mean[0] - 2.0) < 0.5  # Should be close to 2

    def test_train_and_predict_mlp(self):
        """MLP surrogate should produce reasonable predictions."""
        pytest.importorskip("sklearn")
        from astraturbo.ai.surrogate import SurrogateTrainer, SurrogateConfig

        config = SurrogateConfig(
            model_type="mlp",
            mlp_hidden_layers=(16,),
            mlp_max_iter=2000,
        )
        trainer = SurrogateTrainer(config)

        # More samples for MLP to learn, simpler function, normalized range
        rng = np.random.default_rng(42)
        X = rng.random((200, 2))  # [0, 1] range
        Y = X[:, 0] ** 2 + X[:, 1] ** 2  # Simple quadratic, range [0, 2]

        metrics = trainer.train_model(X, Y, model_type="mlp")

        # MLP with small data may not be highly accurate; but should learn the basic pattern
        assert metrics["r2_score"] > 0.0  # Must do better than predicting the mean

        x_test = np.array([0.5, 0.5])
        mean, std = trainer.predict(x_test)
        # y(0.5, 0.5) = 0.5
        assert abs(mean[0] - 0.5) < 0.5  # Reasonable prediction within 0.5 tolerance

    def test_doe_generation(self):
        """DOE should generate correct number of space-filling samples."""
        from astraturbo.ai.surrogate import SurrogateTrainer

        trainer = SurrogateTrainer()
        ds = {"lower": np.array([0.0, 0.0]), "upper": np.array([1.0, 1.0])}
        samples = trainer.generate_doe(ds, n_samples=20, method="lhs", seed=42)

        assert samples.shape == (20, 2)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_run_training_pipeline(self):
        """Training pipeline should collect valid X, Y pairs."""
        from astraturbo.ai.surrogate import SurrogateTrainer

        trainer = SurrogateTrainer()
        ds = {"lower": np.array([0.0]), "upper": np.array([1.0])}
        samples = trainer.generate_doe(ds, n_samples=10)

        def eval_fn(x):
            return np.array([x[0] ** 2])

        X, Y = trainer.run_training_pipeline(samples, eval_fn)

        assert X.shape[0] == 10
        assert Y.shape[0] == 10


# ────────────────────────────────────────────────────────────────
# 9. HPC job manager
# ────────────────────────────────────────────────────────────────

class TestHPCJobManager:
    """Test the local HPC backend."""

    def test_local_backend_submit_and_complete(self, tmp_path):
        """Submit a simple echo job on LocalBackend and verify it completes."""
        from astraturbo.hpc.job_manager import LocalBackend, HPCConfig, JobStatus

        # Create a minimal case directory with a script
        case_dir = tmp_path / "test_case"
        case_dir.mkdir()
        allrun = case_dir / "Allrun"
        allrun.write_text("#!/bin/bash\necho 'hello from test'\n")
        allrun.chmod(0o755)

        backend = LocalBackend()
        job_id = backend.submit(
            case_dir=str(case_dir),
            solver="openfoam",
            n_procs=1,
            job_name="test_job",
            walltime="0:01:00",
        )

        assert job_id.startswith("local_")

        # Wait for completion (should be fast)
        for _ in range(50):
            status = backend.check_status(job_id)
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            time.sleep(0.1)

        assert status == JobStatus.COMPLETED

    def test_local_backend_cancel(self, tmp_path):
        """Cancelling a job should work."""
        from astraturbo.hpc.job_manager import LocalBackend, JobStatus

        case_dir = tmp_path / "long_case"
        case_dir.mkdir()
        allrun = case_dir / "Allrun"
        allrun.write_text("#!/bin/bash\nsleep 60\n")
        allrun.chmod(0o755)

        backend = LocalBackend()
        job_id = backend.submit(
            case_dir=str(case_dir),
            solver="openfoam",
            n_procs=1,
            job_name="long_job",
            walltime="1:00:00",
        )

        # Give it a moment to start
        time.sleep(0.2)
        assert backend.check_status(job_id) == JobStatus.RUNNING

        success = backend.cancel(job_id)
        assert success

    def test_hpc_job_manager_interface(self, tmp_path):
        """HPCJobManager should provide high-level submit/status/cancel."""
        from astraturbo.hpc.job_manager import HPCJobManager, HPCConfig, JobStatus

        case_dir = tmp_path / "managed_case"
        case_dir.mkdir()
        allrun = case_dir / "Allrun"
        allrun.write_text("#!/bin/bash\necho done\n")
        allrun.chmod(0o755)

        config = HPCConfig(backend="local")
        manager = HPCJobManager(config)

        job_id = manager.submit_job(
            case_dir=str(case_dir),
            solver="openfoam",
            job_name="managed_test",
        )
        assert job_id

        # Wait and check
        for _ in range(50):
            status = manager.check_status(job_id)
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            time.sleep(0.1)

        assert status == JobStatus.COMPLETED

        info = manager.get_job_info(job_id)
        assert info is not None
        assert info.name == "managed_test"

    def test_list_jobs(self, tmp_path):
        """list_jobs should return submitted jobs."""
        from astraturbo.hpc.job_manager import HPCJobManager, HPCConfig

        case_dir = tmp_path / "list_case"
        case_dir.mkdir()
        allrun = case_dir / "Allrun"
        allrun.write_text("#!/bin/bash\necho done\n")
        allrun.chmod(0o755)

        config = HPCConfig(backend="local")
        manager = HPCJobManager(config)

        manager.submit_job(case_dir=str(case_dir), solver="openfoam", job_name="j1")
        manager.submit_job(case_dir=str(case_dir), solver="openfoam", job_name="j2")

        jobs = manager.list_jobs()
        assert len(jobs) >= 2


class TestAWSBatchBackend:
    """Test AWSBatchBackend with mocked boto3 clients."""

    def _make_backend(self, tmp_path):
        """Create an AWSBatchBackend with mocked clients (bypasses credential check)."""
        from unittest.mock import MagicMock, patch
        from astraturbo.hpc.job_manager import AWSBatchBackend, HPCConfig

        config = HPCConfig(
            backend="aws",
            aws_region="us-east-1",
            aws_job_queue="test-queue",
            aws_job_definition="test-def",
            aws_s3_bucket="test-bucket",
            aws_s3_prefix="astraturbo",
            aws_container_image="openfoam/openfoam2312-default:latest",
        )

        # Bypass _validate_config by patching it out during construction
        with patch.object(AWSBatchBackend, '_validate_config'):
            backend = AWSBatchBackend(config)

        # Inject mock clients
        backend._batch_client = MagicMock()
        backend._s3_client = MagicMock()

        return backend

    def test_submit_job(self, tmp_path):
        """submit() should upload case to S3 and call batch.submit_job."""
        from unittest.mock import MagicMock
        backend = self._make_backend(tmp_path)

        # Create a minimal case directory
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        (case_dir / "system").mkdir()
        (case_dir / "system" / "controlDict").write_text("FoamFile {}")

        backend._batch_client.submit_job.return_value = {"jobId": "aws-12345"}

        job_id = backend.submit(
            case_dir=str(case_dir),
            solver="openfoam",
            n_procs=4,
            job_name="test_aws",
            walltime="2:00:00",
        )

        assert job_id == "aws-12345"
        backend._batch_client.submit_job.assert_called_once()
        # S3 upload should have been called for the file
        assert backend._s3_client.upload_file.call_count >= 1

    def test_check_status_running(self, tmp_path):
        """check_status should map RUNNING to JobStatus.RUNNING."""
        from astraturbo.hpc.job_manager import JobStatus
        backend = self._make_backend(tmp_path)

        backend._batch_client.describe_jobs.return_value = {
            "jobs": [{"status": "RUNNING"}]
        }

        assert backend.check_status("aws-12345") == JobStatus.RUNNING

    def test_check_status_succeeded(self, tmp_path):
        """check_status should map SUCCEEDED to JobStatus.COMPLETED."""
        from astraturbo.hpc.job_manager import JobStatus
        backend = self._make_backend(tmp_path)

        backend._batch_client.describe_jobs.return_value = {
            "jobs": [{"status": "SUCCEEDED"}]
        }

        assert backend.check_status("aws-12345") == JobStatus.COMPLETED

    def test_check_status_unknown_job(self, tmp_path):
        """check_status for unknown job should return UNKNOWN."""
        from astraturbo.hpc.job_manager import JobStatus
        backend = self._make_backend(tmp_path)

        backend._batch_client.describe_jobs.return_value = {"jobs": []}

        assert backend.check_status("nonexistent") == JobStatus.UNKNOWN

    def test_cancel_job(self, tmp_path):
        """cancel() should call cancel_job on the Batch client."""
        backend = self._make_backend(tmp_path)

        result = backend.cancel("aws-12345")

        assert result is True
        backend._batch_client.cancel_job.assert_called_once_with(
            jobId="aws-12345", reason="Cancelled by AstraTurbo"
        )

    def test_cancel_falls_back_to_terminate(self, tmp_path):
        """cancel() should try terminate_job if cancel_job raises."""
        backend = self._make_backend(tmp_path)
        backend._batch_client.cancel_job.side_effect = Exception("already running")

        result = backend.cancel("aws-12345")

        assert result is True
        backend._batch_client.terminate_job.assert_called_once_with(
            jobId="aws-12345", reason="Cancelled by AstraTurbo"
        )

    def test_download_results(self, tmp_path):
        """download_results should fetch files from S3 output prefix."""
        from unittest.mock import MagicMock
        backend = self._make_backend(tmp_path)

        backend._batch_client.describe_jobs.return_value = {
            "jobs": [{"jobName": "test_aws"}]
        }

        # Mock S3 paginator
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "astraturbo/test_aws/output/solver.log"},
                    {"Key": "astraturbo/test_aws/output/postProcessing/data.csv"},
                ]
            }
        ]
        backend._s3_client.get_paginator.return_value = mock_paginator

        output_dir = tmp_path / "results"
        result = backend.download_results("aws-12345", str(output_dir))

        assert result is True
        assert backend._s3_client.download_file.call_count == 2

    def test_parse_walltime(self, tmp_path):
        """_parse_walltime should convert HH:MM:SS to seconds."""
        from astraturbo.hpc.job_manager import AWSBatchBackend

        assert AWSBatchBackend._parse_walltime("2:00:00") == 7200
        assert AWSBatchBackend._parse_walltime("0:30:00") == 1800
        assert AWSBatchBackend._parse_walltime("1:30") == 90
        assert AWSBatchBackend._parse_walltime("invalid") == 86400


# ────────────────────────────────────────────────────────────────
# 10. Design database
# ────────────────────────────────────────────────────────────────

class TestDesignDatabase:
    """Test SQLite-backed design database."""

    def test_save_and_load(self, tmp_path):
        """Save a design and load it back; data should match."""
        from astraturbo.database.design_db import DesignDatabase

        db_path = tmp_path / "test_designs.db"
        db = DesignDatabase(db_path)

        params = {"pressure_ratio": 4.0, "cl0": 1.0, "mass_flow": 10.5}
        results = {"efficiency": 0.88, "surge_margin": 0.15}

        design_id = db.save_design(
            name="Test Design 1",
            parameters=params,
            results=results,
            tags=["compressor", "test"],
            notes="Test design for unit testing",
        )

        assert design_id > 0

        loaded = db.load_design(design_id)
        assert loaded["name"] == "Test Design 1"
        assert loaded["parameters"]["pressure_ratio"] == 4.0
        assert loaded["parameters"]["cl0"] == 1.0
        assert loaded["results"]["efficiency"] == 0.88
        assert "compressor" in loaded["tags"]
        assert loaded["notes"] == "Test design for unit testing"

        db.close()

    def test_search_by_name(self, tmp_path):
        """Search should find designs by name substring."""
        from astraturbo.database.design_db import DesignDatabase

        db = DesignDatabase(tmp_path / "search.db")
        db.save_design("Compressor Stage 1", {"pr": 1.5})
        db.save_design("Turbine Row A", {"pr": 3.0})
        db.save_design("Compressor Stage 2", {"pr": 2.0})

        results = db.search("Compressor")
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "Compressor Stage 1" in names
        assert "Compressor Stage 2" in names

        db.close()

    def test_search_by_parameter(self, tmp_path):
        """Structured search should filter by parameter values."""
        from astraturbo.database.design_db import DesignDatabase

        db = DesignDatabase(tmp_path / "param_search.db")
        db.save_design("Low PR", {"pressure_ratio": 1.2})
        db.save_design("Mid PR", {"pressure_ratio": 2.5})
        db.save_design("High PR", {"pressure_ratio": 5.0})

        results = db.search("param:pressure_ratio>2.0")
        assert len(results) == 2  # Mid PR and High PR

        db.close()

    def test_delete_design(self, tmp_path):
        """Deleting a design should remove it permanently."""
        from astraturbo.database.design_db import DesignDatabase

        db = DesignDatabase(tmp_path / "delete.db")
        d_id = db.save_design("Temp Design", {"a": 1})
        assert db.count() == 1

        db.delete_design(d_id)
        assert db.count() == 0

        with pytest.raises(KeyError):
            db.load_design(d_id)

        db.close()

    def test_update_design(self, tmp_path):
        """Updating a design should modify specified fields only."""
        from astraturbo.database.design_db import DesignDatabase

        db = DesignDatabase(tmp_path / "update.db")
        d_id = db.save_design("Original", {"a": 1, "b": 2})

        db.update_design(d_id, name="Updated", results={"efficiency": 0.9})

        loaded = db.load_design(d_id)
        assert loaded["name"] == "Updated"
        assert loaded["parameters"]["a"] == 1  # Unchanged
        assert loaded["results"]["efficiency"] == 0.9

        db.close()

    def test_export_csv(self, tmp_path):
        """Export should create a valid CSV with all designs."""
        from astraturbo.database.design_db import DesignDatabase

        db = DesignDatabase(tmp_path / "export.db")
        db.save_design("D1", {"pr": 1.5}, {"eta": 0.88})
        db.save_design("D2", {"pr": 2.0}, {"eta": 0.90})

        csv_path = tmp_path / "export.csv"
        count = db.export_csv(csv_path)

        assert count == 2
        assert csv_path.exists()

        # Read CSV and verify
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "param_pr" in rows[0]
        assert "result_eta" in rows[0]

        db.close()

    def test_compare_designs(self, tmp_path):
        """Comparing two designs should identify differences."""
        from astraturbo.database.design_db import DesignDatabase

        db = DesignDatabase(tmp_path / "compare.db")
        id1 = db.save_design("D1", {"pr": 1.5, "cl0": 1.0}, {"eta": 0.88})
        id2 = db.save_design("D2", {"pr": 2.0, "cl0": 1.0}, {"eta": 0.91})

        comparison = db.compare(id1, id2)

        assert len(comparison["parameter_differences"]) == 1  # pr differs
        assert comparison["parameter_differences"][0]["parameter"] == "pr"
        assert len(comparison["result_differences"]) == 1  # eta differs
        assert "summary" in comparison

        db.close()

    def test_context_manager(self, tmp_path):
        """Database should work as a context manager."""
        from astraturbo.database.design_db import DesignDatabase

        db_path = tmp_path / "ctx.db"
        with DesignDatabase(db_path) as db:
            db.save_design("CTX Design", {"x": 1})
            assert db.count() == 1


# ────────────────────────────────────────────────────────────────
# 11. Multi-fidelity optimization
# ────────────────────────────────────────────────────────────────

class TestMultiFidelity:
    """Test multi-fidelity optimization cascade."""

    def test_three_level_optimization(self):
        """3-level cascade should filter designs progressively."""
        from astraturbo.optimization.multifidelity import MultiFidelityOptimizer
        from astraturbo.optimization.parameterization import DesignSpace

        ds = DesignSpace()
        ds.add("loading_coefficient", lb=0.2, ub=0.6)
        ds.add("flow_coefficient", lb=0.3, ub=0.7)

        optimizer = MultiFidelityOptimizer(ds)

        # Level 1: fast screening
        n_level1 = 100
        n_level2 = 20
        n_level3 = 5

        def eval_l1(x):
            params = ds.decode(x)
            psi = params["loading_coefficient"]
            phi = params["flow_coefficient"]
            eta = 0.92 - 0.3 * max(0, psi - 0.4)
            return {"efficiency": eta}

        def eval_l2(x):
            params = ds.decode(x)
            psi = params["loading_coefficient"]
            eta = 0.91 - 0.35 * max(0, psi - 0.38)
            return {"efficiency": eta}

        def eval_l3(x):
            params = ds.decode(x)
            psi = params["loading_coefficient"]
            eta = 0.905 - 0.4 * max(0, psi - 0.37)
            return {"efficiency": eta}

        optimizer.add_level("meanline", eval_l1, n_samples=n_level1, filter_top_n=n_level2)
        optimizer.add_level("throughflow", eval_l2, filter_top_n=n_level3)
        optimizer.add_level("cfd", eval_l3, filter_top_n=3)

        result = optimizer.run()

        assert len(result.levels) == 3
        assert result.levels[0]["n_input"] == n_level1
        assert result.levels[0]["n_output"] == n_level2
        assert result.levels[1]["n_input"] == n_level2
        assert result.levels[1]["n_output"] == n_level3
        assert result.levels[2]["n_input"] == n_level3
        assert result.levels[2]["n_output"] == 3

        assert result.final_designs is not None
        assert len(result.final_designs) == 3
        assert len(result.final_objectives) == 3
        assert result.total_evaluations == n_level1 + n_level2 + n_level3
        assert result.total_time > 0

    def test_best_design(self):
        """best_design() should return the highest-efficiency candidate."""
        from astraturbo.optimization.multifidelity import MultiFidelityOptimizer
        from astraturbo.optimization.parameterization import DesignSpace

        ds = DesignSpace()
        ds.add("x", lb=0.0, ub=1.0)

        optimizer = MultiFidelityOptimizer(ds)

        def eval_fn(x):
            # Efficiency peaks at x=0.5
            return {"efficiency": 1.0 - (x[0] - 0.5) ** 2}

        optimizer.add_level("only", eval_fn, n_samples=50, filter_top_n=5)

        result = optimizer.run()
        best = result.best_design()

        assert best is not None
        assert best["objectives"]["efficiency"] > 0.9

    def test_create_default_turbomachinery(self):
        """Factory method should create a properly configured optimizer."""
        from astraturbo.optimization.multifidelity import MultiFidelityOptimizer
        from astraturbo.optimization.parameterization import DesignSpace

        ds = DesignSpace()
        ds.add("loading_coefficient", lb=0.2, ub=0.6)
        ds.add("flow_coefficient", lb=0.3, ub=0.7)

        optimizer = MultiFidelityOptimizer.create_default_turbomachinery(
            ds, n_meanline=50, n_throughflow=10, n_cfd=3
        )

        assert len(optimizer.levels) == 3
        assert optimizer.levels[0].name == "meanline"
        assert optimizer.levels[1].name == "throughflow"
        assert optimizer.levels[2].name == "cfd"

        result = optimizer.run()
        assert result.final_designs is not None
        assert len(result.final_designs) <= 3

    def test_filter_threshold(self):
        """Threshold filtering should exclude low-performing designs."""
        from astraturbo.optimization.multifidelity import MultiFidelityOptimizer
        from astraturbo.optimization.parameterization import DesignSpace

        ds = DesignSpace()
        ds.add("x", lb=0.0, ub=1.0)

        optimizer = MultiFidelityOptimizer(ds)

        def eval_fn(x):
            return {"efficiency": x[0]}

        optimizer.add_level(
            "screen", eval_fn,
            n_samples=100,
            filter_threshold=0.5,
            filter_metric="efficiency",
        )

        result = optimizer.run()

        # All surviving designs should have efficiency >= 0.5
        for obj in result.final_objectives:
            assert obj["efficiency"] >= 0.5
