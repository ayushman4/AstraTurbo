"""Design database module for AstraTurbo.

Provides SQLite-backed storage for turbomachinery design parameters
and results with search, comparison, and export capabilities.
"""

from .design_db import DesignDatabase

__all__ = ["DesignDatabase"]
