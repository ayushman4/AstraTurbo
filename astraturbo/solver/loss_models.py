"""Loss and deviation models for throughflow analysis.

Provides empirical correlations widely used in axial compressor
and turbine preliminary design:
  - Lieblein diffusion factor and profile loss
  - Ainley-Mathieson secondary loss
  - Tip clearance loss
  - Carter's deviation rule

References:
    Lieblein, S., "Loss and stall analysis of compressor cascades",
    ASME J. Basic Eng., 1959.

    Ainley, D.G. and Mathieson, G.C.R., "A Method of Performance
    Estimation for Axial-Flow Turbines", ARC R&M 2974, 1951.

    Carter, A.D.S., "The Low Speed Performance of Related Aerofoils
    in Cascade", ARC CP 29, 1950.
"""

from __future__ import annotations

import numpy as np


def lieblein_diffusion_factor(
    v1: float,
    v2: float,
    v_theta1: float,
    v_theta2: float,
    solidity: float,
) -> float:
    """Compute the Lieblein diffusion factor for a compressor cascade.

    The diffusion factor measures the deceleration on the suction surface
    and is the primary parameter controlling profile loss.

    DF = 1 - V2/V1 + |delta_V_theta| / (2 * sigma * V1)

    Args:
        v1: Inlet relative velocity (m/s).
        v2: Exit relative velocity (m/s).
        v_theta1: Inlet tangential velocity component (m/s).
        v_theta2: Exit tangential velocity component (m/s).
        solidity: Blade solidity (chord / pitch).

    Returns:
        Diffusion factor (dimensionless). Typically 0.3-0.6 for good designs.
    """
    if abs(v1) < 1e-10:
        return 0.0

    df = 1.0 - v2 / v1 + abs(v_theta2 - v_theta1) / (2.0 * solidity * v1)
    return float(df)


def lieblein_profile_loss(
    diffusion_factor: float,
    solidity: float,
    Re: float,
    surface_roughness: float = 0.0,
) -> float:
    """Compute profile loss coefficient using Lieblein's correlation.

    The loss coefficient omega is related to the momentum thickness
    ratio theta/c through the diffusion factor:

        theta/c = f(DF)  (empirical curve)
        omega = 2 * (theta/c) * sigma / cos(beta_2)

    For simplicity, we use the Koch-Smith correlation:
        omega_bar = 0.004 * (1 + 3.1 * (DF - 0.4)^2 + 0.4 * (DF - 0.4)^8)

    with a Reynolds number correction factor.

    Args:
        diffusion_factor: Lieblein diffusion factor.
        solidity: Blade solidity (chord / pitch).
        Re: Reynolds number based on chord and inlet velocity.
        surface_roughness: Surface roughness relative to chord (k/c).
            0.0 for smooth blades.

    Returns:
        Total pressure loss coefficient: omega = delta_Pt / (0.5 * rho * V1^2).
    """
    # Clamp DF to valid range
    df = max(0.0, min(diffusion_factor, 0.9))

    # Base profile loss (Koch-Smith type correlation)
    if df <= 0.4:
        omega_base = 0.004 * (1.0 + 3.1 * (df - 0.4) ** 2)
    else:
        omega_base = 0.004 * (
            1.0 + 3.1 * (df - 0.4) ** 2 + 0.4 * (df - 0.4) ** 8
        )

    # Reynolds number correction (reference Re = 2e5)
    re_ref = 2.0e5
    if Re > 1.0:
        re_correction = (re_ref / Re) ** 0.2
    else:
        re_correction = 1.0

    # Surface roughness correction
    roughness_correction = 1.0 + 50.0 * surface_roughness

    # Scale by solidity (loss scales roughly with solidity)
    omega = omega_base * re_correction * roughness_correction * solidity

    return float(max(omega, 0.0))


def ainley_mathieson_secondary_loss(
    inlet_angle: float,
    outlet_angle: float,
    span: float,
    chord: float,
    cl: float | None = None,
) -> float:
    """Compute secondary (endwall) loss using Ainley-Mathieson correlation.

    Secondary losses arise from the interaction of the boundary layer
    with the blade passage pressure gradient, creating passage vortices.

    The correlation is:
        Y_sec = 0.0334 * f(inlet_angle) * (cos(alpha_2) / cos(alpha_1))
                * (CL^2 / (s/c)) * (c/h)

    Simplified form when CL is not available:
        Y_sec = 0.018 * sigma * (cos^2(alpha_2) / cos^3(alpha_m))
                * 2 * (tan(alpha_1) - tan(alpha_2))^2 * (c/h)

    Args:
        inlet_angle: Inlet flow angle (degrees from axial).
        outlet_angle: Exit flow angle (degrees from axial).
        span: Blade span / height (m).
        chord: Blade chord (m).
        cl: Lift coefficient (optional). If None, computed from angles.

    Returns:
        Secondary loss coefficient.
    """
    alpha1_rad = np.radians(inlet_angle)
    alpha2_rad = np.radians(outlet_angle)

    cos_a1 = np.cos(alpha1_rad)
    cos_a2 = np.cos(alpha2_rad)
    tan_a1 = np.tan(alpha1_rad)
    tan_a2 = np.tan(alpha2_rad)

    # Mean angle
    alpha_m = np.arctan(0.5 * (tan_a1 + tan_a2))
    cos_am = np.cos(alpha_m)

    # Aspect ratio
    aspect_ratio = span / chord if chord > 1e-10 else 1e10

    if cl is not None:
        # Direct correlation with lift coefficient
        y_sec = 0.0334 * (cos_a2 / max(cos_a1, 1e-10)) * cl**2 / aspect_ratio
    else:
        # Ainley-Mathieson simplified
        delta_tan = tan_a1 - tan_a2
        if abs(cos_am) < 1e-10:
            return 0.0
        y_sec = (
            0.018
            * cos_a2**2
            / cos_am**3
            * delta_tan**2
            / aspect_ratio
        )

    return float(max(y_sec, 0.0))


def tip_clearance_loss(
    clearance: float,
    span: float,
    loading: float,
    efficiency_factor: float = 0.93,
) -> float:
    """Compute tip clearance loss coefficient.

    Based on Denton's model:
        delta_eta = k * (tau/h) * CL / (s/c)

    Simplified correlation:
        Y_tc = 0.7 * (tau / h) * (delta_Ctheta / U)

    For general use:
        Y_tc = B * (tau/h) * loading_parameter

    where B ~ 0.5-0.7 depending on the tip geometry.

    Args:
        clearance: Tip clearance gap (m).
        span: Blade span (m).
        loading: Loading parameter (delta_Ctheta / U or CL * sigma).
            Typical range 0.3-0.8.
        efficiency_factor: Empirical coefficient (default 0.93 from Denton).

    Returns:
        Tip clearance loss coefficient.
    """
    if span < 1e-10:
        return 0.0

    tau_over_h = clearance / span
    y_tc = (1.0 - efficiency_factor) * 2.0 * tau_over_h * loading

    return float(max(y_tc, 0.0))


def carter_deviation(
    camber: float,
    solidity: float,
    stagger: float,
    exponent: float = 0.5,
) -> float:
    """Compute flow deviation angle using Carter's rule.

    Carter's rule relates the deviation angle (difference between
    the metal angle and flow angle at the trailing edge) to the
    cascade geometry:

        delta = m * theta / sigma^n

    where:
        delta = deviation angle (degrees)
        theta = camber angle (degrees)
        m = function of stagger angle (typically 0.23 + 0.002 * stagger)
        sigma = solidity
        n = exponent (typically 0.5 for compressors)

    Args:
        camber: Blade camber angle (degrees).
        solidity: Blade solidity (chord / pitch).
        stagger: Blade stagger angle (degrees from axial).
        exponent: Solidity exponent (default 0.5).

    Returns:
        Deviation angle (degrees). Positive means flow turns less
        than the blade metal angle.
    """
    if solidity < 1e-10:
        return camber  # Zero solidity = no blade effect

    # Carter's m-factor
    # For circular arc camber: m ~ 0.23 * (2a/c)^2 + alpha2/500
    # Simplified: m = 0.23 + 0.002 * |stagger|
    m = 0.23 + 0.002 * abs(stagger)

    # Deviation angle
    deviation = m * camber / solidity**exponent

    return float(deviation)


def total_loss_coefficient(
    diffusion_factor: float,
    solidity: float,
    Re: float,
    inlet_angle: float,
    outlet_angle: float,
    span: float,
    chord: float,
    clearance: float = 0.0,
    loading: float = 0.5,
) -> dict[str, float]:
    """Compute total loss as the sum of profile, secondary, and tip clearance.

    Convenience function that calls all three loss models and sums them.

    Args:
        diffusion_factor: Lieblein diffusion factor.
        solidity: Blade solidity.
        Re: Reynolds number.
        inlet_angle: Inlet flow angle (degrees).
        outlet_angle: Exit flow angle (degrees).
        span: Blade span (m).
        chord: Blade chord (m).
        clearance: Tip clearance (m). 0 for stator or shrouded.
        loading: Loading parameter for tip loss.

    Returns:
        Dictionary with 'profile', 'secondary', 'tip_clearance', 'total'.
    """
    omega_p = lieblein_profile_loss(diffusion_factor, solidity, Re)
    omega_s = ainley_mathieson_secondary_loss(
        inlet_angle, outlet_angle, span, chord
    )
    omega_tc = tip_clearance_loss(clearance, span, loading) if clearance > 0 else 0.0
    omega_total = omega_p + omega_s + omega_tc

    return {
        "profile": float(omega_p),
        "secondary": float(omega_s),
        "tip_clearance": float(omega_tc),
        "total": float(omega_total),
    }
