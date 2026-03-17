"""2D airfoil profile module for AstraTurbo.

Available types:
  - Superposition: Camber line + thickness distribution
"""

from .profile import Profile
from .superposition import Superposition

__all__ = ["Profile", "Superposition"]
