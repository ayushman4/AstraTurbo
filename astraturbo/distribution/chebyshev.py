"""Chebyshev point distribution.

Clusters points near both endpoints (leading and trailing edge),
which improves resolution where curvature is typically highest
on airfoil profiles.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..baseclass import ATObject


class Chebyshev(ATObject):
    """Chebyshev distribution — clusters points near x=0 and x=1."""

    def __call__(self, sample_rate: int) -> NDArray[np.float64]:
        """Generate sample_rate points in [0, 1] with Chebyshev clustering."""
        t = np.cos(np.pi * np.arange(sample_rate) / (sample_rate - 1))
        return (t * 0.5 + 0.5)[::-1]
