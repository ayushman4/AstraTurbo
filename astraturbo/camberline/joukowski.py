"""Joukowski camber line.

Simple parabolic camber line y = m * x * (1 - x) from Joukowski theory.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..foundation import NumericProperty, memoize
from .camberline import CamberLine


class Joukowski(CamberLine):
    """Joukowski camber line: y = max_camber * x * (1 - x).

    Parameters:
        max_camber: Maximum camber as fraction of chord.
    """

    max_camber = NumericProperty(default=0.12)

    def __init__(self, max_camber: float = 0.12) -> None:
        super().__init__()
        self.max_camber = max_camber

    @classmethod
    def default(cls) -> Joukowski:
        return cls(max_camber=0.12)

    @memoize
    def get_derivations(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        return self.max_camber * (1 - 2 * x)

    @memoize
    def as_array(self) -> NDArray[np.float64]:
        x = self.distribution(self.sample_rate)
        y = self.max_camber * x * (1 - x)
        return np.column_stack((x, y))
