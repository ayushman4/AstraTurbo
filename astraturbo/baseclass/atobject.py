"""Base object for all AstraTurbo domain classes.

Modernized for Python 3.10+:
  - Uses __init_subclass__ instead of metaclass (simpler, more Pythonic)
  - Properties auto-discovered via __set_name__ (no manual name assignment)
  - Python 3.10+ type hints
"""

from __future__ import annotations

import re
from typing import Generator

from ..foundation.properties import Property


def _camel_to_display(name: str) -> str:
    """Convert CamelCase class name to display string.

    Example: 'CircularArc' -> 'Circular Arc'
    """
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", name)
    name = re.sub(r"(.)([0-9]+)", r"\1 \2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)


class ATObject:
    """Base class for all AstraTurbo domain objects.

    Automatically discovers Property descriptors defined on the class and its
    bases. Provides:
      - ``properties``: iterator over all Property descriptors
      - ``invalidate_cache()``: clear memoized computation results
      - Human-readable __str__ from class name
    """

    def __str__(self) -> str:
        return _camel_to_display(type(self).__name__)

    @property
    def properties(self) -> Generator[Property, None, None]:
        """Yield all Property descriptors from this class and its bases."""
        seen: set[str] = set()
        for cls in type(self).__mro__:
            for name, value in vars(cls).items():
                if isinstance(value, Property) and name not in seen:
                    seen.add(name)
                    yield value

    def invalidate_cache(self) -> None:
        """Clear all memoized computation results."""
        self.__dict__.pop("_cache", None)
