"""Report generation module for AstraTurbo.

Generates HTML design reports from analysis results — the deliverable
that engineers hand to their manager or include in a design review.
"""

from .generator import generate_report, ReportConfig

__all__ = ["generate_report", "ReportConfig"]
