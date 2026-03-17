"""Validation test: meanline thermodynamic consistency.

Verifies that core thermodynamic relations are correctly implemented:
  - Isentropic relations: T2/T1 = (P2/P1)^((gamma-1)/gamma)
  - Euler equation: work = U * delta_C_theta
  - Total-static relations: T0 = T + V^2 / (2*cp)
  - Energy conservation through stages

These are fundamental checks that any meanline solver must satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===== Thermodynamic helper functions (independent implementation) =====

def isentropic_temperature_ratio(
    pressure_ratio: float, gamma: float = 1.4
) -> float:
    """Compute T2/T1 for an isentropic process.

    T2/T1 = (P2/P1)^((gamma-1)/gamma)

    Args:
        pressure_ratio: P2/P1.
        gamma: Ratio of specific heats.

    Returns:
        Temperature ratio T2/T1.
    """
    return pressure_ratio ** ((gamma - 1.0) / gamma)


def isentropic_pressure_ratio(
    temperature_ratio: float, gamma: float = 1.4
) -> float:
    """Compute P2/P1 for an isentropic process.

    P2/P1 = (T2/T1)^(gamma/(gamma-1))

    Args:
        temperature_ratio: T2/T1.
        gamma: Ratio of specific heats.

    Returns:
        Pressure ratio P2/P1.
    """
    return temperature_ratio ** (gamma / (gamma - 1.0))


def total_temperature(
    T_static: float, velocity: float, cp: float = 1004.5
) -> float:
    """Compute total temperature from static temperature and velocity.

    T0 = T + V^2 / (2 * cp)

    Args:
        T_static: Static temperature (K).
        velocity: Flow velocity (m/s).
        cp: Specific heat at constant pressure (J/kg/K).

    Returns:
        Total temperature (K).
    """
    return T_static + velocity**2 / (2.0 * cp)


def euler_work(U: float, delta_C_theta: float) -> float:
    """Euler turbomachinery equation.

    delta_h0 = U * delta_C_theta

    Args:
        U: Blade speed (m/s).
        delta_C_theta: Change in tangential velocity (m/s).

    Returns:
        Specific work (J/kg).
    """
    return U * delta_C_theta


def stage_efficiency(
    actual_work: float,
    ideal_work: float,
    machine_type: str = "compressor",
) -> float:
    """Compute isentropic stage efficiency.

    For compressor: eta = ideal_work / actual_work (ideal < actual)
    For turbine: eta = actual_work / ideal_work (actual < ideal)

    Args:
        actual_work: Actual specific work (J/kg).
        ideal_work: Isentropic specific work (J/kg).
        machine_type: 'compressor' or 'turbine'.

    Returns:
        Isentropic efficiency.
    """
    if machine_type == "compressor":
        return ideal_work / actual_work if abs(actual_work) > 1e-10 else 0.0
    else:
        return actual_work / ideal_work if abs(ideal_work) > 1e-10 else 0.0


class TestIsentropicRelations:
    """Test isentropic thermodynamic relations."""

    def test_isentropic_T_ratio_basic(self) -> None:
        """T2/T1 = (P2/P1)^((gamma-1)/gamma) for ideal gas."""
        # Standard air, gamma=1.4
        # PR=2: T_ratio = 2^(0.4/1.4) = 2^0.2857 = 1.2190
        T_ratio = isentropic_temperature_ratio(2.0, gamma=1.4)
        assert T_ratio == pytest.approx(2.0**0.2857142857, rel=1e-6)
        assert T_ratio == pytest.approx(1.2190, abs=0.001)

    def test_isentropic_P_ratio_basic(self) -> None:
        """P2/P1 = (T2/T1)^(gamma/(gamma-1))"""
        # T_ratio = 1.5: P_ratio = 1.5^3.5 = 6.838
        P_ratio = isentropic_pressure_ratio(1.5, gamma=1.4)
        assert P_ratio == pytest.approx(1.5**3.5, rel=1e-6)

    def test_isentropic_round_trip(self) -> None:
        """Converting PR -> TR -> PR should be identity."""
        for pr in [1.5, 2.0, 3.0, 5.0, 8.0]:
            tr = isentropic_temperature_ratio(pr, gamma=1.4)
            pr_back = isentropic_pressure_ratio(tr, gamma=1.4)
            assert pr_back == pytest.approx(pr, rel=1e-10), (
                f"Round-trip failed: PR={pr} -> TR={tr} -> PR={pr_back}"
            )

    def test_isentropic_with_different_gamma(self) -> None:
        """Test with different gamma values (monatomic, diatomic)."""
        # Monatomic gas: gamma=5/3
        tr = isentropic_temperature_ratio(2.0, gamma=5.0/3.0)
        assert tr == pytest.approx(2.0**0.4, rel=1e-6)

        # Diatomic at high T: gamma=7/5=1.4
        tr = isentropic_temperature_ratio(3.0, gamma=1.4)
        assert tr == pytest.approx(3.0**(2.0/7.0), rel=1e-6)

    def test_unity_pressure_ratio(self) -> None:
        """PR=1 should give TR=1 (no compression)."""
        assert isentropic_temperature_ratio(1.0) == pytest.approx(1.0, rel=1e-15)


class TestTotalStaticRelations:
    """Test total-static temperature and pressure relations."""

    def test_total_temperature_at_rest(self) -> None:
        """At zero velocity, T0 = T."""
        assert total_temperature(288.15, 0.0) == pytest.approx(288.15, rel=1e-15)

    def test_total_temperature_known(self) -> None:
        """T0 = T + V^2/(2*cp) with known values.

        T=288.15 K, V=200 m/s, cp=1004.5:
        T0 = 288.15 + 200^2/2009 = 288.15 + 19.91 = 308.06 K
        """
        T0 = total_temperature(288.15, 200.0, cp=1004.5)
        expected = 288.15 + 200.0**2 / (2.0 * 1004.5)
        assert T0 == pytest.approx(expected, rel=1e-10)

    def test_total_temperature_conservation(self) -> None:
        """If velocity changes but no work done, T0 stays constant.

        At station 1: T1=300, V1=100 -> T01
        At station 2: T2=?, V2=150 -> T02 = T01 (no work)
        -> T2 = T01 - V2^2/(2*cp)
        """
        cp = 1004.5
        T01 = total_temperature(300.0, 100.0, cp)
        # T2 should be:
        T2 = T01 - 150.0**2 / (2.0 * cp)
        T02 = total_temperature(T2, 150.0, cp)
        assert T02 == pytest.approx(T01, rel=1e-10)


class TestEulerEquation:
    """Test the Euler turbomachinery equation."""

    def test_euler_basic(self) -> None:
        """Work = U * delta_C_theta."""
        work = euler_work(U=300.0, delta_C_theta=100.0)
        assert work == pytest.approx(30000.0, rel=1e-10)

    def test_euler_zero_turning(self) -> None:
        """No tangential velocity change -> no work."""
        work = euler_work(U=300.0, delta_C_theta=0.0)
        assert work == pytest.approx(0.0, abs=1e-15)

    def test_euler_compressor_work_positive(self) -> None:
        """For a compressor, work input should be positive.

        Compressor increases C_theta (in direction of rotation).
        """
        # Inlet: no swirl, Exit: C_theta = 150 m/s
        work = euler_work(U=300.0, delta_C_theta=150.0)
        assert work > 0, "Compressor work should be positive"
        assert work == pytest.approx(45000.0, rel=1e-10)

    def test_euler_turbine_work_negative(self) -> None:
        """For a turbine, work is extracted (negative delta_C_theta from rotor perspective).

        Turbine: flow enters with high swirl, exits with less.
        """
        # Exit swirl less than inlet
        work = euler_work(U=350.0, delta_C_theta=-200.0)
        assert work < 0, "Turbine should extract work (negative)"

    def test_euler_temperature_rise(self) -> None:
        """Verify that Euler work equals cp * delta_T0.

        delta_h0 = U * delta_C_theta = cp * (T02 - T01)
        -> delta_T0 = U * delta_C_theta / cp
        """
        U = 300.0
        delta_C_theta = 120.0
        cp = 1004.5

        work = euler_work(U, delta_C_theta)
        delta_T0 = work / cp

        # Expected
        expected_delta_T0 = 300.0 * 120.0 / 1004.5
        assert delta_T0 == pytest.approx(expected_delta_T0, rel=1e-10)


class TestThermodynamicConsistency:
    """End-to-end thermodynamic consistency checks for a compressor stage."""

    def test_single_stage_compressor(self) -> None:
        """Verify thermodynamic consistency of a single compressor stage.

        Given: T01=288.15K, P01=101325Pa, U=300m/s, delta_C_theta=120m/s
        Assume eta_s = 0.88

        1. Work = U * delta_C_theta
        2. delta_T0 = work / cp
        3. T02 = T01 + delta_T0
        4. T02s (isentropic) = T01 + eta_s * delta_T0  -- wait, inverse
           Actually: T02s = T01 + delta_T0_ideal, where delta_T0_ideal = eta_s * delta_T0
        5. P02/P01 = (T02s/T01)^(gamma/(gamma-1))  -- wrong, need to think carefully

        Correct relations:
        - actual work: w = U * delta_C_theta
        - delta_T0 = w / cp  (actual total temperature rise)
        - T02 = T01 + delta_T0
        - ideal (isentropic) temperature rise: delta_T0s = eta * delta_T0
        - T02s = T01 + delta_T0s
        - P02/P01 = (T02s/T01)^(gamma/(gamma-1))

        Verify all these are self-consistent.
        """
        T01 = 288.15  # K
        P01 = 101325.0  # Pa
        U = 300.0  # m/s
        delta_C_theta = 120.0  # m/s
        eta_s = 0.88
        gamma = 1.4
        cp = 1004.5  # J/kg/K

        # Step 1: Work
        work = euler_work(U, delta_C_theta)
        assert work == pytest.approx(36000.0, rel=1e-10)

        # Step 2: Temperature rise
        delta_T0 = work / cp
        T02 = T01 + delta_T0
        assert T02 > T01, "Compressor should increase temperature"

        # Step 3: Isentropic temperature rise
        delta_T0s = eta_s * delta_T0
        T02s = T01 + delta_T0s

        # Step 4: Pressure ratio
        T_ratio_s = T02s / T01
        PR = isentropic_pressure_ratio(T_ratio_s, gamma)

        # Step 5: Verify consistency - if we go PR -> T_ratio, we get T02s/T01
        T_ratio_check = isentropic_temperature_ratio(PR, gamma)
        assert T_ratio_check == pytest.approx(T_ratio_s, rel=1e-10)

        # Step 6: Verify PR is reasonable (typical single stage: 1.2-1.8)
        assert 1.05 < PR < 2.5, f"PR={PR} seems unreasonable for single stage"

        # Step 7: Verify efficiency consistency
        ideal_work = cp * delta_T0s
        eta_check = stage_efficiency(work, ideal_work, "compressor")
        assert eta_check == pytest.approx(eta_s, rel=1e-10)

    def test_multi_stage_compressor(self) -> None:
        """Verify thermodynamic consistency across multiple stages.

        The overall pressure ratio should equal the product of stage PRs.
        """
        T0 = 288.15
        P0 = 101325.0
        gamma = 1.4
        cp = 1004.5
        n_stages = 5
        eta_s = 0.88
        U = 300.0
        delta_C_theta = 100.0  # per stage

        T0_current = T0
        P0_current = P0
        PR_product = 1.0

        for _ in range(n_stages):
            work = euler_work(U, delta_C_theta)
            delta_T0 = work / cp
            T02 = T0_current + delta_T0
            delta_T0s = eta_s * delta_T0
            T02s = T0_current + delta_T0s
            PR_stage = isentropic_pressure_ratio(T02s / T0_current, gamma)

            PR_product *= PR_stage
            P0_current *= PR_stage
            T0_current = T02

        # Verify overall PR from temperature ratio
        overall_T_ratio = T0_current / T0
        # Note: overall PR != (overall T ratio)^(gamma/(gamma-1)) due to losses
        # But PR_product should equal P0_current / P0
        assert PR_product == pytest.approx(P0_current / P0, rel=1e-10)

        # Verify PR is reasonable for 5-stage compressor (3-10)
        assert 2.0 < PR_product < 15.0, (
            f"Overall PR={PR_product} seems unreasonable for {n_stages} stages"
        )

    def test_energy_conservation(self) -> None:
        """Total enthalpy is conserved in a stator (no work).

        h01 = h02 for a stator:
        cp*T01 + V1^2/2 == cp*T02 + V2^2/2 (but T01=T02, so just verify)
        Actually: T01 = T02 for stator (no work).
        """
        cp = 1004.5
        T01 = 350.0  # K

        # At station 1: V1=200 m/s
        T_static_1 = T01 - 200.0**2 / (2.0 * cp)
        T0_check_1 = total_temperature(T_static_1, 200.0, cp)
        assert T0_check_1 == pytest.approx(T01, rel=1e-10)

        # At station 2 (stator exit): V2=250 m/s, T02=T01 (no work)
        T_static_2 = T01 - 250.0**2 / (2.0 * cp)
        T0_check_2 = total_temperature(T_static_2, 250.0, cp)
        assert T0_check_2 == pytest.approx(T01, rel=1e-10)

        # Static temperature changes but total is conserved
        assert T_static_2 < T_static_1, (
            "Higher velocity should give lower static temperature"
        )
