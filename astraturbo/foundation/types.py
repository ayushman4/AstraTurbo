"""Type aliases and protocols for AstraTurbo."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# Common array types used throughout the codebase
FloatArray = NDArray[np.float64]
PointArray2D = NDArray[np.float64]   # Shape (N, 2)
PointArray3D = NDArray[np.float64]   # Shape (N, 3)
GridArray3D = NDArray[np.float64]    # Shape (Ni, Nj, Nk)


@runtime_checkable
class Updatable(Protocol):
    """Protocol for objects that can receive update notifications."""

    def update(self, broadcast: bool = True) -> None: ...
