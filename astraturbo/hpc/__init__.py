"""HPC / Cloud integration module for AstraTurbo.

Provides job submission and management for remote HPC clusters
and local execution:
  - job_manager: HPCJobManager with SLURM, PBS, and Local backends
"""

from .job_manager import (
    HPCJobManager,
    HPCConfig,
    SLURMBackend,
    PBSBackend,
    LocalBackend,
    JobStatus,
)

__all__ = [
    "HPCJobManager",
    "HPCConfig",
    "SLURMBackend",
    "PBSBackend",
    "LocalBackend",
    "JobStatus",
]
