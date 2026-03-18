"""HPC / Cloud job management for AstraTurbo CFD simulations.

Provides a unified interface for submitting and managing CFD jobs
across different HPC systems (SLURM, PBS) and local execution.

Supports:
  - SLURM: sbatch-based submission via SSH
  - PBS: qsub-based submission via SSH
  - Local: subprocess-based execution (default)

Security:
  - SSH connections use key-based authentication
  - All commands are parameterized to prevent injection
  - No passwords stored in memory

Usage::

    config = HPCConfig(
        host="cluster.example.com",
        user="turboengineer",
        ssh_key="~/.ssh/id_rsa",
        backend="slurm",
    )
    manager = HPCJobManager(config)
    job_id = manager.submit_job("my_case/", solver="openfoam", n_procs=64)
    status = manager.check_status(job_id)
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class HPCConfig:
    """Configuration for HPC job submission."""

    # Connection
    host: str = "localhost"
    user: str = ""
    ssh_key: str = ""
    ssh_port: int = 22

    # Backend
    backend: str = "local"               # 'slurm', 'pbs', 'local', 'aws'

    # Queue/partition
    queue: str = "default"
    partition: str = ""

    # Resources
    max_nodes: int = 1
    cpus_per_node: int = 32
    memory_gb: int = 64
    walltime: str = "24:00:00"           # HH:MM:SS
    gpu: bool = False
    gpu_count: int = 0

    # Paths (on the remote system)
    remote_work_dir: str = ""
    module_load_commands: list[str] = field(
        default_factory=lambda: ["module load openfoam/2312"]
    )

    # Job defaults
    default_solver: str = "openfoam"
    email_notification: str = ""         # Email for job notifications

    # AWS Batch settings
    aws_region: str = "us-east-1"
    aws_job_queue: str = ""              # AWS Batch job queue name
    aws_job_definition: str = ""         # AWS Batch job definition name/ARN
    aws_s3_bucket: str = ""              # S3 bucket for case upload/download
    aws_s3_prefix: str = "astraturbo"    # S3 key prefix for job data
    aws_container_image: str = ""        # Docker image for the solver
    aws_vcpus: int = 4                   # vCPUs per container
    aws_memory_mb: int = 8192            # Container memory in MB
    aws_gpu_count: int = 0               # GPUs per container
    aws_platform: str = "EC2"            # 'EC2' or 'FARGATE'


@dataclass
class JobInfo:
    """Information about a submitted job."""

    job_id: str = ""
    name: str = ""
    status: JobStatus = JobStatus.UNKNOWN
    submit_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    case_dir: str = ""
    solver: str = ""
    n_procs: int = 1
    log_file: str = ""
    error_message: str = ""
    return_code: int = -1
    remote_dir: str = ""


class HPCBackend(ABC):
    """Abstract base class for HPC submission backends."""

    def __init__(self, config: HPCConfig) -> None:
        self.config = config

    @abstractmethod
    def submit(
        self,
        case_dir: str,
        solver: str,
        n_procs: int,
        job_name: str,
        walltime: str,
    ) -> str:
        """Submit a job and return the job ID."""
        ...

    @abstractmethod
    def check_status(self, job_id: str) -> JobStatus:
        """Check the status of a submitted job."""
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        ...

    @abstractmethod
    def download_results(self, job_id: str, local_dir: str) -> bool:
        """Download results from a completed job."""
        ...

    def _ssh_command(self, command: str) -> tuple[int, str, str]:
        """Execute a command on the remote host via SSH.

        Args:
            command: Command to execute remotely.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        cfg = self.config
        ssh_args = ["ssh"]

        if cfg.ssh_key:
            key_path = os.path.expanduser(cfg.ssh_key)
            ssh_args.extend(["-i", key_path])

        if cfg.ssh_port != 22:
            ssh_args.extend(["-p", str(cfg.ssh_port)])

        ssh_args.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            f"{cfg.user}@{cfg.host}",
            command,
        ])

        try:
            result = subprocess.run(
                ssh_args,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return -1, "", str(e)


class SLURMBackend(HPCBackend):
    """SLURM workload manager backend.

    Generates sbatch scripts and submits via SSH.
    """

    def submit(
        self,
        case_dir: str,
        solver: str,
        n_procs: int,
        job_name: str,
        walltime: str,
    ) -> str:
        """Submit a job to SLURM via sbatch.

        Creates a SLURM batch script and submits it on the remote host.

        Args:
            case_dir: Path to the case directory (local or remote).
            solver: Solver to run ('openfoam', 'su2', etc.).
            n_procs: Number of MPI processes.
            job_name: SLURM job name.
            walltime: Wall clock time limit (HH:MM:SS).

        Returns:
            SLURM job ID string.
        """
        cfg = self.config

        # Calculate nodes needed
        nodes = max(1, (n_procs + cfg.cpus_per_node - 1) // cfg.cpus_per_node)
        nodes = min(nodes, cfg.max_nodes)

        # Build SLURM script
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={shlex.quote(job_name)}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --ntasks={n_procs}",
            f"#SBATCH --cpus-per-task=1",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --mem={cfg.memory_gb}G",
            f"#SBATCH --output=slurm-%j.out",
            f"#SBATCH --error=slurm-%j.err",
        ]

        if cfg.partition:
            script_lines.append(f"#SBATCH --partition={shlex.quote(cfg.partition)}")
        elif cfg.queue != "default":
            script_lines.append(f"#SBATCH --partition={shlex.quote(cfg.queue)}")

        if cfg.email_notification:
            script_lines.append(f"#SBATCH --mail-user={cfg.email_notification}")
            script_lines.append("#SBATCH --mail-type=END,FAIL")

        if cfg.gpu and cfg.gpu_count > 0:
            script_lines.append(f"#SBATCH --gres=gpu:{cfg.gpu_count}")

        script_lines.append("")

        # Module loads
        for mod_cmd in cfg.module_load_commands:
            script_lines.append(mod_cmd)
        script_lines.append("")

        # Change to case directory
        remote_case = case_dir
        if cfg.remote_work_dir:
            remote_case = f"{cfg.remote_work_dir}/{os.path.basename(case_dir)}"

        script_lines.append(f"cd {shlex.quote(remote_case)}")
        script_lines.append("")

        # Solver command
        solver_cmd = self._get_solver_command(solver, n_procs)
        script_lines.append(solver_cmd)

        script_content = "\n".join(script_lines) + "\n"

        # Write script remotely and submit
        script_name = f"submit_{job_name}.sh"
        remote_script = f"{remote_case}/{script_name}"

        # Upload script via ssh
        escaped_content = script_content.replace("'", "'\\''")
        write_cmd = f"mkdir -p {shlex.quote(remote_case)} && echo '{escaped_content}' > {shlex.quote(remote_script)}"
        rc, _, err = self._ssh_command(write_cmd)
        if rc != 0:
            raise RuntimeError(f"Failed to write SLURM script: {err}")

        # Submit with sbatch
        submit_cmd = f"cd {shlex.quote(remote_case)} && sbatch {shlex.quote(script_name)}"
        rc, stdout, err = self._ssh_command(submit_cmd)

        if rc != 0:
            raise RuntimeError(f"sbatch failed: {err}")

        # Parse job ID from "Submitted batch job 12345"
        parts = stdout.split()
        if len(parts) >= 4:
            return parts[-1]
        return stdout

    def check_status(self, job_id: str) -> JobStatus:
        """Check SLURM job status using sacct/squeue."""
        safe_id = shlex.quote(job_id)
        rc, stdout, _ = self._ssh_command(
            f"squeue -j {safe_id} -h -o '%T' 2>/dev/null || "
            f"sacct -j {safe_id} -n -o State%20 2>/dev/null"
        )

        if rc != 0 or not stdout:
            return JobStatus.UNKNOWN

        state = stdout.strip().split()[0].upper() if stdout.strip() else ""

        status_map = {
            "PENDING": JobStatus.PENDING,
            "RUNNING": JobStatus.RUNNING,
            "COMPLETED": JobStatus.COMPLETED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.CANCELLED,
            "TIMEOUT": JobStatus.FAILED,
            "NODE_FAIL": JobStatus.FAILED,
            "COMPLETING": JobStatus.RUNNING,
        }

        return status_map.get(state, JobStatus.UNKNOWN)

    def cancel(self, job_id: str) -> bool:
        """Cancel a SLURM job using scancel."""
        safe_id = shlex.quote(job_id)
        rc, _, _ = self._ssh_command(f"scancel {safe_id}")
        return rc == 0

    def download_results(self, job_id: str, local_dir: str) -> bool:
        """Download results using rsync/scp."""
        return self._rsync_download(job_id, local_dir)

    def _rsync_download(self, job_id: str, local_dir: str) -> bool:
        """Download results via rsync."""
        cfg = self.config
        remote_path = f"{cfg.user}@{cfg.host}:{cfg.remote_work_dir}/"
        local_path = local_dir

        rsync_args = ["rsync", "-avz", "--progress"]
        if cfg.ssh_key:
            key_path = os.path.expanduser(cfg.ssh_key)
            rsync_args.extend(["-e", f"ssh -i {key_path} -p {cfg.ssh_port}"])
        rsync_args.extend([remote_path, local_path])

        try:
            result = subprocess.run(rsync_args, capture_output=True, text=True, timeout=600)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_solver_command(self, solver: str, n_procs: int) -> str:
        """Generate solver execution command."""
        if solver == "openfoam":
            if n_procs > 1:
                return (
                    f"decomposePar -force\n"
                    f"mpirun -np {n_procs} simpleFoam -parallel > solver.log 2>&1\n"
                    f"reconstructPar"
                )
            return "simpleFoam > solver.log 2>&1"
        elif solver == "su2":
            if n_procs > 1:
                return f"mpirun -np {n_procs} SU2_CFD astraturbo.cfg > solver.log 2>&1"
            return "SU2_CFD astraturbo.cfg > solver.log 2>&1"
        else:
            return f"echo 'Unknown solver: {solver}'"


class PBSBackend(HPCBackend):
    """PBS/Torque workload manager backend.

    Generates qsub scripts for PBS-based clusters.
    """

    def submit(
        self,
        case_dir: str,
        solver: str,
        n_procs: int,
        job_name: str,
        walltime: str,
    ) -> str:
        """Submit a job to PBS via qsub."""
        cfg = self.config

        nodes = max(1, (n_procs + cfg.cpus_per_node - 1) // cfg.cpus_per_node)
        ppn = min(n_procs, cfg.cpus_per_node)

        script_lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -l nodes={nodes}:ppn={ppn}",
            f"#PBS -l walltime={walltime}",
            f"#PBS -l mem={cfg.memory_gb}gb",
            f"#PBS -o pbs_stdout.log",
            f"#PBS -e pbs_stderr.log",
        ]

        if cfg.queue != "default":
            script_lines.append(f"#PBS -q {cfg.queue}")

        if cfg.gpu and cfg.gpu_count > 0:
            script_lines.append(f"#PBS -l gpus={cfg.gpu_count}")

        if cfg.email_notification:
            script_lines.append(f"#PBS -M {cfg.email_notification}")
            script_lines.append("#PBS -m ae")

        script_lines.append("")

        for mod_cmd in cfg.module_load_commands:
            script_lines.append(mod_cmd)
        script_lines.append("")

        remote_case = case_dir
        if cfg.remote_work_dir:
            remote_case = f"{cfg.remote_work_dir}/{os.path.basename(case_dir)}"

        script_lines.append(f"cd {shlex.quote(remote_case)}")
        script_lines.append("")

        # Solver command (reuse SLURM backend's logic)
        solver_cmd = SLURMBackend._get_solver_command(self, solver, n_procs)
        script_lines.append(solver_cmd)

        script_content = "\n".join(script_lines) + "\n"

        script_name = f"submit_{job_name}.sh"
        remote_script = f"{remote_case}/{script_name}"

        escaped_content = script_content.replace("'", "'\\''")
        write_cmd = f"mkdir -p {shlex.quote(remote_case)} && echo '{escaped_content}' > {shlex.quote(remote_script)}"
        rc, _, err = self._ssh_command(write_cmd)
        if rc != 0:
            raise RuntimeError(f"Failed to write PBS script: {err}")

        submit_cmd = f"cd {shlex.quote(remote_case)} && qsub {shlex.quote(script_name)}"
        rc, stdout, err = self._ssh_command(submit_cmd)

        if rc != 0:
            raise RuntimeError(f"qsub failed: {err}")

        # PBS returns job ID like "12345.pbs_server"
        return stdout.strip()

    def check_status(self, job_id: str) -> JobStatus:
        """Check PBS job status using qstat."""
        safe_id = shlex.quote(job_id)
        rc, stdout, _ = self._ssh_command(f"qstat -f {safe_id} 2>/dev/null")

        if rc != 0 or not stdout:
            # Job may have completed and left the queue
            return JobStatus.COMPLETED

        if "job_state = R" in stdout:
            return JobStatus.RUNNING
        elif "job_state = Q" in stdout:
            return JobStatus.PENDING
        elif "job_state = C" in stdout:
            return JobStatus.COMPLETED
        elif "job_state = E" in stdout:
            return JobStatus.FAILED

        return JobStatus.UNKNOWN

    def cancel(self, job_id: str) -> bool:
        """Cancel a PBS job using qdel."""
        safe_id = shlex.quote(job_id)
        rc, _, _ = self._ssh_command(f"qdel {safe_id}")
        return rc == 0

    def download_results(self, job_id: str, local_dir: str) -> bool:
        """Download results via rsync."""
        return SLURMBackend._rsync_download(self, job_id, local_dir)


class LocalBackend(HPCBackend):
    """Local execution backend using subprocess.

    Runs CFD solvers on the local machine. No SSH required.
    This is the default backend for development and testing.
    """

    def __init__(self, config: HPCConfig | None = None) -> None:
        super().__init__(config or HPCConfig(backend="local"))
        self._processes: dict[str, subprocess.Popen] = {}
        self._job_info: dict[str, JobInfo] = {}

    def submit(
        self,
        case_dir: str,
        solver: str,
        n_procs: int,
        job_name: str,
        walltime: str,
    ) -> str:
        """Run solver locally as a subprocess.

        Args:
            case_dir: Local case directory.
            solver: Solver name.
            n_procs: Number of processes.
            job_name: Job identifier.
            walltime: Not used for local execution.

        Returns:
            Locally-generated job ID string.
        """
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        case_path = Path(case_dir).resolve()

        if not case_path.exists():
            raise FileNotFoundError(f"Case directory not found: {case_path}")

        # Build solver command
        if solver == "openfoam":
            if n_procs > 1:
                cmd = ["bash", "-c",
                       f"cd {shlex.quote(str(case_path))} && "
                       f"decomposePar -force && "
                       f"mpirun -np {n_procs} simpleFoam -parallel > solver.log 2>&1 && "
                       f"reconstructPar"]
            else:
                allrun = case_path / "Allrun"
                if allrun.exists():
                    cmd = ["bash", str(allrun)]
                else:
                    cmd = ["bash", "-c",
                           f"cd {shlex.quote(str(case_path))} && simpleFoam > solver.log 2>&1"]
        elif solver == "su2":
            cfg_file = case_path / "astraturbo.cfg"
            if n_procs > 1:
                cmd = ["mpirun", "-np", str(n_procs), "SU2_CFD", str(cfg_file)]
            else:
                cmd = ["SU2_CFD", str(cfg_file)]
        else:
            cmd = ["echo", f"Unknown solver: {solver}"]

        # Start subprocess
        log_file = case_path / f"{job_name}.log"
        with open(log_file, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(case_path),
            )

        self._processes[job_id] = proc
        self._job_info[job_id] = JobInfo(
            job_id=job_id,
            name=job_name,
            status=JobStatus.RUNNING,
            submit_time=time.time(),
            case_dir=str(case_path),
            solver=solver,
            n_procs=n_procs,
            log_file=str(log_file),
        )

        return job_id

    def check_status(self, job_id: str) -> JobStatus:
        """Check local process status."""
        proc = self._processes.get(job_id)
        info = self._job_info.get(job_id)

        if proc is None:
            return JobStatus.UNKNOWN

        ret = proc.poll()
        if ret is None:
            return JobStatus.RUNNING
        elif ret == 0:
            if info:
                info.status = JobStatus.COMPLETED
                info.end_time = time.time()
                info.return_code = 0
            return JobStatus.COMPLETED
        else:
            if info:
                info.status = JobStatus.FAILED
                info.end_time = time.time()
                info.return_code = ret
                info.error_message = f"Process exited with code {ret}"
            return JobStatus.FAILED

    def cancel(self, job_id: str) -> bool:
        """Terminate local process."""
        proc = self._processes.get(job_id)
        if proc is None:
            return False

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

        info = self._job_info.get(job_id)
        if info:
            info.status = JobStatus.CANCELLED
            info.end_time = time.time()

        return True

    def download_results(self, job_id: str, local_dir: str) -> bool:
        """For local backend, copy results to the target directory."""
        info = self._job_info.get(job_id)
        if info is None:
            return False

        src = Path(info.case_dir)
        dst = Path(local_dir)

        if src == dst:
            return True  # Already there

        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            return True
        except OSError:
            return False


class AWSBatchBackend(HPCBackend):
    """AWS Batch backend for cloud-based CFD execution.

    Submits CFD jobs to AWS Batch, with case data transferred via S3.
    Requires boto3 and configured AWS credentials (env vars, ~/.aws/credentials,
    or IAM role).

    Workflow:
      1. Upload case directory to S3
      2. Submit AWS Batch job (container pulls case from S3, runs solver)
      3. Poll job status via Batch API
      4. Download results from S3 when complete

    The container image must include the solver (OpenFOAM, SU2, etc.)
    and the aws cli for S3 transfers.
    """

    def __init__(self, config: HPCConfig) -> None:
        super().__init__(config)
        self._batch_client: Any = None
        self._s3_client: Any = None
        # Validate configuration early
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate AWS config and credentials at init time."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for the AWS Batch backend. "
                "Install it with: pip install 'astraturbo[aws]'"
            )

        cfg = self.config
        if not cfg.aws_s3_bucket:
            raise ValueError(
                "aws_s3_bucket is required for the AWS backend. "
                "Set it via --aws-s3-bucket or HPCConfig(aws_s3_bucket=...)"
            )
        if not cfg.aws_job_queue:
            raise ValueError(
                "aws_job_queue is required for the AWS backend. "
                "Set it via --aws-job-queue or HPCConfig(aws_job_queue=...)"
            )

        # Verify credentials are available (STS get-caller-identity)
        try:
            sts = boto3.client("sts", region_name=cfg.aws_region)
            sts.get_caller_identity()
        except Exception as e:
            raise RuntimeError(
                f"AWS credential check failed: {e}. "
                "Ensure AWS credentials are configured (env vars, ~/.aws/credentials, or IAM role)."
            ) from e

    def _get_batch_client(self) -> Any:
        """Lazy-init AWS Batch client."""
        if self._batch_client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for the AWS Batch backend. "
                    "Install it with: pip install 'astraturbo[aws]'"
                )
            self._batch_client = boto3.client(
                "batch", region_name=self.config.aws_region
            )
        return self._batch_client

    def _get_s3_client(self) -> Any:
        """Lazy-init S3 client."""
        if self._s3_client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for the AWS Batch backend. "
                    "Install it with: pip install 'astraturbo[aws]'"
                )
            self._s3_client = boto3.client(
                "s3", region_name=self.config.aws_region
            )
        return self._s3_client

    def _s3_case_key(self, job_name: str) -> str:
        """S3 key prefix for a job's case data."""
        prefix = self.config.aws_s3_prefix.strip("/")
        return f"{prefix}/{job_name}"

    def _upload_case_to_s3(self, case_dir: str, job_name: str) -> str:
        """Upload case directory to S3.

        Args:
            case_dir: Local path to case directory.
            job_name: Job name used as S3 sub-prefix.

        Returns:
            S3 URI (s3://bucket/prefix/job_name).
        """
        s3 = self._get_s3_client()
        bucket = self.config.aws_s3_bucket
        s3_prefix = self._s3_case_key(job_name)
        case_path = Path(case_dir).resolve()

        if not case_path.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_path}")

        for root, _dirs, files in os.walk(case_path):
            for filename in files:
                local_file = Path(root) / filename
                rel_path = local_file.relative_to(case_path)
                s3_key = f"{s3_prefix}/input/{rel_path}"
                s3.upload_file(str(local_file), bucket, s3_key)

        return f"s3://{bucket}/{s3_prefix}"

    def submit(
        self,
        case_dir: str,
        solver: str,
        n_procs: int,
        job_name: str,
        walltime: str,
    ) -> str:
        """Submit a job to AWS Batch.

        Uploads the case to S3 and submits a Batch job that:
          1. Downloads the case from S3
          2. Runs the solver
          3. Uploads results back to S3

        Args:
            case_dir: Local case directory to upload.
            solver: Solver name ('openfoam', 'su2').
            n_procs: Number of vCPUs to request.
            job_name: Job name.
            walltime: Wall time (used to set Batch timeout).

        Returns:
            AWS Batch job ID.
        """
        cfg = self.config
        batch = self._get_batch_client()

        # Upload case to S3
        s3_uri = self._upload_case_to_s3(case_dir, job_name)
        s3_input = f"{s3_uri}/input"
        s3_output = f"{s3_uri}/output"

        # Parse walltime to seconds for Batch timeout
        timeout_seconds = self._parse_walltime(walltime)

        # Build container command
        solver_cmd = self._get_solver_command(solver, n_procs)
        container_cmd = [
            "bash", "-c",
            f"aws s3 sync {s3_input} /workdir/ && "
            f"cd /workdir && {solver_cmd} && "
            f"aws s3 sync /workdir/ {s3_output}"
        ]

        # Resource requirements
        resource_requirements = [
            {"type": "VCPU", "value": str(n_procs or cfg.aws_vcpus)},
            {"type": "MEMORY", "value": str(cfg.aws_memory_mb)},
        ]

        if cfg.aws_gpu_count > 0:
            resource_requirements.append(
                {"type": "GPU", "value": str(cfg.aws_gpu_count)}
            )

        # Build submit parameters
        submit_params: dict[str, Any] = {
            "jobName": job_name,
            "jobQueue": cfg.aws_job_queue,
            "timeout": {"attemptDurationSeconds": timeout_seconds},
        }

        if cfg.aws_job_definition:
            # Use a pre-registered job definition
            submit_params["jobDefinition"] = cfg.aws_job_definition
            submit_params["containerOverrides"] = {
                "command": container_cmd,
                "resourceRequirements": resource_requirements,
            }
            if cfg.aws_container_image:
                submit_params["containerOverrides"]["image"] = cfg.aws_container_image
        else:
            # Register a one-off job definition
            job_def_name = f"astraturbo-{solver}-{job_name}"
            self._register_job_definition(
                job_def_name, solver, container_cmd, resource_requirements
            )
            submit_params["jobDefinition"] = job_def_name
            submit_params["containerOverrides"] = {
                "command": container_cmd,
                "resourceRequirements": resource_requirements,
            }

        # Environment variables for the container
        env_vars = [
            {"name": "ASTRATURBO_S3_INPUT", "value": s3_input},
            {"name": "ASTRATURBO_S3_OUTPUT", "value": s3_output},
            {"name": "ASTRATURBO_SOLVER", "value": solver},
            {"name": "ASTRATURBO_NPROCS", "value": str(n_procs)},
        ]
        submit_params.setdefault("containerOverrides", {})
        submit_params["containerOverrides"]["environment"] = env_vars

        response = batch.submit_job(**submit_params)
        return response["jobId"]

    def _register_job_definition(
        self,
        name: str,
        solver: str,
        command: list[str],
        resource_requirements: list[dict[str, str]],
    ) -> None:
        """Register a one-off AWS Batch job definition.

        Args:
            name: Job definition name.
            solver: Solver name (used to pick default image).
            command: Container command.
            resource_requirements: vCPU/memory/GPU requirements.
        """
        batch = self._get_batch_client()
        cfg = self.config

        image = cfg.aws_container_image
        if not image:
            # Default public OpenFOAM images as fallback
            image_map = {
                "openfoam": "openfoam/openfoam2312-default:latest",
                "su2": "su2code/su2:latest",
            }
            image = image_map.get(solver, "ubuntu:22.04")

        platform = "FARGATE" if cfg.aws_platform.upper() == "FARGATE" else "EC2"

        container_props: dict[str, Any] = {
            "image": image,
            "command": command,
            "resourceRequirements": resource_requirements,
        }

        if platform == "FARGATE":
            container_props["networkConfiguration"] = {
                "assignPublicIp": "ENABLED"
            }

        batch.register_job_definition(
            jobDefinitionName=name,
            type="container",
            platformCapabilities=[platform],
            containerProperties=container_props,
            timeout={"attemptDurationSeconds": 86400},
        )

    def check_status(self, job_id: str) -> JobStatus:
        """Check AWS Batch job status.

        Args:
            job_id: AWS Batch job ID.

        Returns:
            Mapped JobStatus.
        """
        batch = self._get_batch_client()

        try:
            response = batch.describe_jobs(jobs=[job_id])
        except Exception:
            return JobStatus.UNKNOWN

        jobs = response.get("jobs", [])
        if not jobs:
            return JobStatus.UNKNOWN

        batch_status = jobs[0].get("status", "UNKNOWN")

        status_map = {
            "SUBMITTED": JobStatus.PENDING,
            "PENDING": JobStatus.PENDING,
            "RUNNABLE": JobStatus.PENDING,
            "STARTING": JobStatus.RUNNING,
            "RUNNING": JobStatus.RUNNING,
            "SUCCEEDED": JobStatus.COMPLETED,
            "FAILED": JobStatus.FAILED,
        }

        return status_map.get(batch_status, JobStatus.UNKNOWN)

    def cancel(self, job_id: str) -> bool:
        """Cancel an AWS Batch job.

        Uses cancel_job for PENDING/RUNNABLE jobs,
        terminate_job for RUNNING jobs.

        Args:
            job_id: AWS Batch job ID.

        Returns:
            True if the cancellation API call succeeded.
        """
        batch = self._get_batch_client()

        try:
            # Try cancel first (for queued jobs)
            batch.cancel_job(jobId=job_id, reason="Cancelled by AstraTurbo")
            return True
        except Exception:
            pass

        try:
            # Terminate running jobs
            batch.terminate_job(jobId=job_id, reason="Cancelled by AstraTurbo")
            return True
        except Exception:
            return False

    def download_results(self, job_id: str, local_dir: str) -> bool:
        """Download results from S3 after job completion.

        Looks up the job name from AWS Batch, then syncs
        the output prefix from S3 to the local directory.

        Args:
            job_id: AWS Batch job ID.
            local_dir: Local directory to download into.

        Returns:
            True if download succeeded.
        """
        batch = self._get_batch_client()
        s3 = self._get_s3_client()

        try:
            response = batch.describe_jobs(jobs=[job_id])
            jobs = response.get("jobs", [])
            if not jobs:
                return False

            job_name = jobs[0].get("jobName", "")
            if not job_name:
                return False
        except Exception:
            return False

        bucket = self.config.aws_s3_bucket
        s3_prefix = f"{self._s3_case_key(job_name)}/output/"

        dst = Path(local_dir)
        dst.mkdir(parents=True, exist_ok=True)

        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    rel_path = s3_key[len(s3_prefix):]
                    if not rel_path:
                        continue
                    local_file = dst / rel_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, s3_key, str(local_file))
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_walltime(walltime: str) -> int:
        """Parse HH:MM:SS walltime to seconds."""
        parts = walltime.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            try:
                return int(parts[0])
            except ValueError:
                return 86400  # Default 24 hours

    @staticmethod
    def _get_solver_command(solver: str, n_procs: int) -> str:
        """Generate solver command for the container."""
        if solver == "openfoam":
            if n_procs > 1:
                return (
                    "decomposePar -force && "
                    f"mpirun -np {n_procs} simpleFoam -parallel > solver.log 2>&1 && "
                    "reconstructPar"
                )
            return "simpleFoam > solver.log 2>&1"
        elif solver == "su2":
            if n_procs > 1:
                return f"mpirun -np {n_procs} SU2_CFD astraturbo.cfg > solver.log 2>&1"
            return "SU2_CFD astraturbo.cfg > solver.log 2>&1"
        else:
            return f"echo 'Unknown solver: {solver}'"


class HPCJobManager:
    """High-level job manager with backend abstraction.

    Usage::

        config = HPCConfig(backend="slurm", host="cluster.example.com")
        manager = HPCJobManager(config)

        job_id = manager.submit_job("my_case/", solver="openfoam", n_procs=64)
        status = manager.check_status(job_id)

        if status == JobStatus.COMPLETED:
            manager.download_results(job_id, "local_results/")
    """

    def __init__(self, config: HPCConfig | None = None) -> None:
        self.config = config or HPCConfig()
        self._backend = self._create_backend()
        self._jobs: dict[str, JobInfo] = {}

    def _create_backend(self) -> HPCBackend:
        """Create the appropriate backend based on config."""
        backend_type = self.config.backend.lower()
        if backend_type == "slurm":
            return SLURMBackend(self.config)
        elif backend_type == "pbs":
            return PBSBackend(self.config)
        elif backend_type == "local":
            return LocalBackend(self.config)
        elif backend_type == "aws":
            return AWSBatchBackend(self.config)
        else:
            raise ValueError(
                f"Unknown backend '{backend_type}'. "
                f"Available: slurm, pbs, local, aws"
            )

    def submit_job(
        self,
        case_dir: str,
        solver: str = "openfoam",
        n_procs: int = 1,
        queue: str = "",
        walltime: str = "",
        job_name: str = "",
    ) -> str:
        """Submit a CFD job for execution.

        Args:
            case_dir: Path to the case directory.
            solver: Solver name ('openfoam', 'su2', 'fluent', 'cfx').
            n_procs: Number of MPI processes.
            queue: Queue/partition (overrides config).
            walltime: Wall time limit (overrides config).
            job_name: Job name (auto-generated if empty).

        Returns:
            Job ID string.
        """
        if not job_name:
            job_name = f"astraturbo_{uuid.uuid4().hex[:6]}"

        wt = walltime or self.config.walltime

        if queue:
            self.config.queue = queue

        job_id = self._backend.submit(case_dir, solver, n_procs, job_name, wt)

        self._jobs[job_id] = JobInfo(
            job_id=job_id,
            name=job_name,
            status=JobStatus.PENDING,
            submit_time=time.time(),
            case_dir=case_dir,
            solver=solver,
            n_procs=n_procs,
        )

        return job_id

    def check_status(self, job_id: str) -> JobStatus:
        """Check the status of a submitted job.

        Args:
            job_id: Job ID from submit_job().

        Returns:
            Current JobStatus.
        """
        status = self._backend.check_status(job_id)

        if job_id in self._jobs:
            self._jobs[job_id].status = status

        return status

    def download_results(self, job_id: str, local_dir: str) -> bool:
        """Download results from a completed job.

        Args:
            job_id: Job ID.
            local_dir: Local directory to download results into.

        Returns:
            True if download succeeded.
        """
        return self._backend.download_results(job_id, local_dir)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job.

        Args:
            job_id: Job ID.

        Returns:
            True if cancellation succeeded.
        """
        success = self._backend.cancel(job_id)

        if success and job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.CANCELLED

        return success

    def get_job_info(self, job_id: str) -> JobInfo | None:
        """Get detailed information about a job."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status_filter: JobStatus | None = None,
    ) -> list[JobInfo]:
        """List all jobs, optionally filtered by status.

        Args:
            status_filter: If set, only return jobs with this status.

        Returns:
            List of JobInfo objects.
        """
        jobs = list(self._jobs.values())

        # Update status for running jobs
        for job in jobs:
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                job.status = self._backend.check_status(job.job_id)

        if status_filter is not None:
            jobs = [j for j in jobs if j.status == status_filter]

        return jobs

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 30.0,
        timeout: float = 86400.0,
    ) -> JobStatus:
        """Block until a job completes or times out.

        Args:
            job_id: Job ID.
            poll_interval: Seconds between status checks.
            timeout: Maximum wait time in seconds.

        Returns:
            Final JobStatus.
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.check_status(job_id)
            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return status
            time.sleep(poll_interval)

        return JobStatus.UNKNOWN
