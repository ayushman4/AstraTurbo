"""Thickness distribution module for AstraTurbo.

Available types:
  - NACA4Digit: Standard NACA 4-digit series
  - NACA65Series: NACA 65-series (compressor blades)
  - JoukowskiThickness: Joukowski-based thickness
  - Elliptic: Semi-elliptical thickness
"""

from .elliptic import Elliptic
from .joukowski import JoukowskiThickness
from .naca4digit import NACA4Digit
from .naca65_series import NACA65Series
from .thickness import ThicknessDistribution

_REGISTRY: dict[str, type[ThicknessDistribution]] = {
    "naca4digit": NACA4Digit,
    "naca65": NACA65Series,
    "joukowski": JoukowskiThickness,
    "elliptic": Elliptic,
}


def create_thickness(type_name: str, **kwargs) -> ThicknessDistribution:
    """Factory function to create a thickness distribution by type name."""
    cls = _REGISTRY.get(type_name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown thickness type '{type_name}'. Available: {available}"
        )
    if kwargs:
        return cls(**kwargs)
    return cls.default()


__all__ = [
    "ThicknessDistribution",
    "NACA4Digit",
    "NACA65Series",
    "JoukowskiThickness",
    "Elliptic",
    "create_thickness",
]
