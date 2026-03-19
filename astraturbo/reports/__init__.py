"""Report generation module for AstraTurbo.

Generates HTML design reports from analysis results — the deliverable
that engineers hand to their manager or include in a design review.
"""

from .generator import generate_report, ReportConfig
from .plots import (
    plot_blade_loading,
    plot_blade_profile,
    plot_cfd_pressure_field,
    plot_cfd_residuals,
    plot_cfd_temperature_field,
    plot_cfd_velocity_field,
    plot_compressor_map_chart,
    plot_cooling_rows,
    plot_engine_stations,
    plot_mesh_2d,
    plot_motor_summary,
    plot_propeller_summary,
    plot_ts_diagram,
    plot_turbopump_power,
    plot_velocity_triangles,
)

__all__ = [
    "generate_report",
    "ReportConfig",
    "plot_blade_loading",
    "plot_blade_profile",
    "plot_cfd_pressure_field",
    "plot_cfd_residuals",
    "plot_cfd_temperature_field",
    "plot_cfd_velocity_field",
    "plot_compressor_map_chart",
    "plot_cooling_rows",
    "plot_engine_stations",
    "plot_mesh_2d",
    "plot_motor_summary",
    "plot_propeller_summary",
    "plot_ts_diagram",
    "plot_turbopump_power",
    "plot_velocity_triangles",
]
