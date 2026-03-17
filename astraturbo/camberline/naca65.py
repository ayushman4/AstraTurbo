"""NACA 65-series camber line.

The NACA 65-series uses a theoretical lift coefficient (CL0) to define
the camber line shape via a logarithmic formula.

Formula:
    f(x) = -CL0 / (4 * pi) * ((1 - x) * ln(1 - x) + x * ln(x))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import NumericProperty, memoize
from .camberline import CamberLine


class NACA65(CamberLine):
    """NACA 65-series camber line defined by design lift coefficient.

    Parameters:
        cl0: Design lift coefficient (CL0). Typical range 0.2-1.8.
    """

    cl0 = NumericProperty(default=1.0)

    def __init__(self, cl0: float = 1.0) -> None:
        super().__init__()
        self.cl0 = cl0

    @classmethod
    def default(cls) -> NACA65:
        return cls(cl0=1.0)

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        # Clamp to avoid log(0)
        x_safe = np.clip(x, 1e-12, 1.0 - 1e-12)
        # dy/dx = -CL0 / (4*pi) * (ln(x) - ln(1-x))
        return -self.cl0 / (4.0 * np.pi) * (np.log(x_safe) - np.log(1.0 - x_safe))

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        # Clamp to avoid log(0)
        x_safe = np.clip(x, 1e-12, 1.0 - 1e-12)
        # f(x) = -CL0 / (4*pi) * ((1-x)*ln(1-x) + x*ln(x))
        y = -self.cl0 / (4.0 * np.pi) * (
            (1.0 - x_safe) * np.log(1.0 - x_safe) + x_safe * np.log(x_safe)
        )
        return np.column_stack((x, y))
