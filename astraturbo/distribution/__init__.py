"""Point sampling distributions for AstraTurbo.

Distributions generate arrays of parameter values in [0, 1] with different
clustering behaviors, used for sampling camber lines and thickness distributions.
"""

from .chebyshev import Chebyshev
from .linear import Linear

__all__ = ["Chebyshev", "Linear"]
