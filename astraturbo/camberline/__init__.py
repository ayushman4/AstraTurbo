"""Camber line generators for AstraTurbo.

Available types:
  - CircularArc: Circular arc camber line
  - QuadraticPolynomial: Degree 2 polynomial
  - CubicPolynomial: Degree 3 polynomial
  - QuarticPolynomial: Degree 4 polynomial
  - Joukowski: Parabolic camber line
  - NACA2Digit: NACA 2-digit series
  - NACA65: NACA 65-series (design lift coefficient)
  - NURBSCamberLine: Arbitrary NURBS curve
"""

from .camberline import CamberLine
from .circular_arc import CircularArc
from .joukowski import Joukowski
from .naca2digit import NACA2Digit
from .naca65 import NACA65
from .nurbs import NURBSCamberLine
from .polynomial import CubicPolynomial, QuadraticPolynomial, QuarticPolynomial

_REGISTRY: dict[str, type[CamberLine]] = {
    "circular_arc": CircularArc,
    "quadratic": QuadraticPolynomial,
    "cubic": CubicPolynomial,
    "quartic": QuarticPolynomial,
    "joukowski": Joukowski,
    "naca2digit": NACA2Digit,
    "naca65": NACA65,
    "nurbs": NURBSCamberLine,
}


def create_camberline(type_name: str, **kwargs) -> CamberLine:
    """Factory function to create a camber line by type name.

    Args:
        type_name: One of 'circular_arc', 'quadratic', 'cubic', 'quartic',
                   'joukowski', 'naca2digit', 'naca65', 'nurbs'.
        **kwargs: Parameters for the specific camber line type.

    Returns:
        A CamberLine instance.

    Raises:
        ValueError: If type_name is not recognized.
    """
    cls = _REGISTRY.get(type_name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown camber line type '{type_name}'. Available: {available}"
        )
    if kwargs:
        return cls(**kwargs)
    return cls.default()


__all__ = [
    "CamberLine",
    "CircularArc",
    "QuadraticPolynomial",
    "CubicPolynomial",
    "QuarticPolynomial",
    "Joukowski",
    "NACA2Digit",
    "NACA65",
    "NURBSCamberLine",
    "create_camberline",
]
