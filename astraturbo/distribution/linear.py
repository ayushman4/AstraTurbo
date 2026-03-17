"""Linear (uniform) point distribution.

Evenly spaces points from 0 to 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..baseclass import ATObject


class Linear(ATObject):
    """Uniform linear distribution — equal spacing from 0 to 1."""

    def __call__(self, sample_rate: int) -> NDArray[np.float64]:
        """Generate sample_rate evenly spaced points in [0, 1]."""
        return np.linspace(0, 1, sample_rate)
