"""HPC / Cloud integration module for AstraTurbo.

Provides job submission and management for remote HPC clusters,
local execution, and AWS Batch:
  - job_manager: HPCJobManager with SLURM, PBS, Local, and AWS Batch backends
  - aws_setup: AWSBatchProvisioner for one-command infrastructure provisioning
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
from .aws_setup import AWSBatchProvisioner

__all__ = [
    "HPCJobManager",
    "HPCConfig",
    "SLURMBackend",
    "PBSBackend",
    "LocalBackend",
    "AWSBatchBackend",
    "AWSBatchProvisioner",
    "JobStatus",
]
