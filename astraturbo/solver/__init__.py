"""Throughflow solver module for AstraTurbo.

Provides a Streamline Curvature Method (SCM) solver for axisymmetric
throughflow analysis on S2m meridional planes:
  - throughflow: Main SCM solver with radial equilibrium
  - loss_models: Profile, secondary, and tip clearance loss correlations
"""

from .throughflow import ThroughflowSolver, ThroughflowConfig, ThroughflowResult
from .loss_models import (
    lieblein_profile_loss,
    ainley_mathieson_secondary_loss,
    tip_clearance_loss,
    carter_deviation,
)

__all__ = [
    "ThroughflowSolver",
    "ThroughflowConfig",
    "ThroughflowResult",
    "lieblein_profile_loss",
    "ainley_mathieson_secondary_loss",
    "tip_clearance_loss",
    "carter_deviation",
]
