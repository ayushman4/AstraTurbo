"""Caching and utility decorators for AstraTurbo."""

from __future__ import annotations

import copy
import functools
from typing import Any, Callable


def memoize(method: Callable) -> Callable:
    """Cache a method's return value on the instance.

    The cached value is stored in ``instance.__dict__['_cache'][method_name]``.
    Returns a copy of the cached value to prevent mutation.

    To invalidate the cache (e.g., when inputs change), delete the
    ``_cache`` key from the instance's __dict__ or call ``instance.invalidate_cache()``.

    Example::

        class Airfoil:
            @memoize
            def compute_coordinates(self):
                # expensive computation
                return result
    """

    @functools.wraps(method)
    def wrapper(obj: Any) -> Any:
        cache = obj.__dict__.setdefault("_cache", {})
        key = method.__name__
        try:
            return copy.copy(cache[key])
        except KeyError:
            result = method(obj)
            cache[key] = result
            return result

    return wrapper
