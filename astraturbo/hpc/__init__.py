"""HPC / Cloud integration module for AstraTurbo.

Provides job submission and management for remote HPC clusters,
local execution, and AWS Batch:
  - job_manager: HPCJobManager with SLURM, PBS, Local, and AWS Batch backends
"""

from .job_manager import (
    HPCJobManager,
    HPCConfig,
    SLURMBackend,
    PBSBackend,
    LocalBackend,
    AWSBatchBackend,
    JobStatus,
)

__all__ = [
    "HPCJobManager",
    "HPCConfig",
    "SLURMBackend",
    "PBSBackend",
    "LocalBackend",
    "AWSBatchBackend",
    "JobStatus",
]
