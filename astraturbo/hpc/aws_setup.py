"""AWS Batch infrastructure provisioner for AstraTurbo.

Automates the creation of all AWS resources needed to run CFD jobs
on AWS Batch: S3 bucket, IAM roles, compute environment, and job queue.

All operations are idempotent — existing resources are reused, not duplicated.

Usage::

    from astraturbo.hpc.aws_setup import AWSBatchProvisioner

    provisioner = AWSBatchProvisioner(region="us-east-1")
    result = provisioner.setup()
    # result = {"bucket": "astraturbo-...", "job_queue": "astraturbo-queue", ...}

    # Later, to tear down:
    provisioner.teardown()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AWSSetupResult:
    """Result of AWS Batch provisioning."""

    bucket_name: str = ""
    job_queue: str = ""
    compute_environment: str = ""
    job_role_arn: str = ""
    service_role_arn: str = ""
    instance_role_arn: str = ""
    security_group_id: str = ""
    subnet_ids: list[str] = field(default_factory=list)
    region: str = ""
    created_resources: list[str] = field(default_factory=list)
    skipped_resources: list[str] = field(default_factory=list)


class AWSBatchProvisioner:
    """Provisions and tears down AWS Batch infrastructure for AstraTurbo.

    Creates these resources (all prefixed with ``astraturbo-``):
      - S3 bucket for case data transfer
      - IAM roles: Batch service, ECS instance, task execution, job role
      - EC2 instance profile
      - VPC security group (outbound-only)
      - Batch compute environment (EC2 or Fargate)
      - Batch job queue

    All operations are idempotent — if a resource already exists it is
    reused and reported as "skipped".

    Requires ``boto3`` and valid AWS credentials.
    """

    # Resource name prefix
    PREFIX = "astraturbo"

    def __init__(
        self,
        region: str = "us-east-1",
        platform: str = "EC2",
        max_vcpus: int = 256,
        instance_types: list[str] | None = None,
        bucket_name: str = "",
    ) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS provisioning. "
                "Install with: pip install 'astraturbo[aws]'"
            )

        self.region = region
        self.platform = platform.upper()
        self.max_vcpus = max_vcpus
        self.instance_types = instance_types or [
            "c5.large", "c5.xlarge", "c5.2xlarge", "c5.4xlarge",
            "m5.large", "m5.xlarge",
        ]

        self._iam = boto3.client("iam")
        self._ec2 = boto3.client("ec2", region_name=region)
        self._s3 = boto3.client("s3", region_name=region)
        self._batch = boto3.client("batch", region_name=region)
        self._sts = boto3.client("sts", region_name=region)
        self._logs = boto3.client("logs", region_name=region)

        self._account_id = self._sts.get_caller_identity()["Account"]

        if bucket_name:
            self._bucket_name = bucket_name
        else:
            self._bucket_name = f"{self.PREFIX}-batch-{self._account_id}-{region}"

        self._compute_env_name = f"{self.PREFIX}-compute-{self.platform.lower()}"
        self._queue_name = f"{self.PREFIX}-queue"
        self._sg_name = f"{self.PREFIX}-batch-sg"

        # Role names
        self._service_role = f"{self.PREFIX}-batch-service-role"
        self._instance_role = f"{self.PREFIX}-ecs-instance-role"
        self._exec_role = f"{self.PREFIX}-ecs-task-exec-role"
        self._job_role = f"{self.PREFIX}-job-role"
        self._instance_profile = f"{self.PREFIX}-ecs-instance-profile"

    # ─── Public API ───────────────────────────────────────────

    def setup(self, log_fn=print) -> AWSSetupResult:
        """Provision all AWS resources. Idempotent.

        Args:
            log_fn: Callable for progress messages (default: print).

        Returns:
            AWSSetupResult with all resource identifiers.
        """
        result = AWSSetupResult(region=self.region)

        log_fn(f"Provisioning AWS Batch in {self.region} ({self.platform})...")

        # 1. S3 bucket
        result.bucket_name = self._ensure_s3_bucket(log_fn, result)

        # 2. IAM roles
        result.service_role_arn = self._ensure_role(
            self._service_role,
            "batch.amazonaws.com",
            ["arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"],
            log_fn, result,
        )
        result.instance_role_arn = self._ensure_role(
            self._instance_role,
            "ec2.amazonaws.com",
            ["arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"],
            log_fn, result,
        )
        self._ensure_role(
            self._exec_role,
            "ecs-tasks.amazonaws.com",
            ["arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"],
            log_fn, result,
        )
        result.job_role_arn = self._ensure_job_role(log_fn, result)

        # 3. Instance profile (for EC2 platform)
        if self.platform == "EC2":
            self._ensure_instance_profile(log_fn, result)

        # 4. VPC / security group
        vpc_id, subnet_ids = self._get_default_vpc()
        result.subnet_ids = subnet_ids
        result.security_group_id = self._ensure_security_group(vpc_id, log_fn, result)

        # 5. Compute environment
        result.compute_environment = self._ensure_compute_environment(
            result, log_fn,
        )

        # 6. Job queue
        result.job_queue = self._ensure_job_queue(log_fn, result)

        # 7. CloudWatch log group
        self._ensure_log_group(log_fn, result)

        log_fn(f"\nProvisioning complete.")
        log_fn(f"  Bucket:      {result.bucket_name}")
        log_fn(f"  Queue:       {result.job_queue}")
        log_fn(f"  Compute:     {result.compute_environment}")
        log_fn(f"  Region:      {result.region}")
        log_fn(f"  Created:     {len(result.created_resources)}")
        log_fn(f"  Skipped:     {len(result.skipped_resources)}")
        log_fn(f"\nSubmit jobs with:")
        log_fn(f"  astraturbo hpc submit ./case --backend aws \\")
        log_fn(f"    --aws-s3-bucket {result.bucket_name} \\")
        log_fn(f"    --aws-job-queue {result.job_queue} \\")
        log_fn(f"    --aws-region {result.region}")

        return result

    def teardown(self, log_fn=print) -> None:
        """Delete all AstraTurbo AWS resources.

        Deletes in reverse dependency order: queue → compute env → SG →
        IAM roles → S3 bucket (only if empty).
        """
        from botocore.exceptions import ClientError

        log_fn(f"Tearing down AstraTurbo AWS resources in {self.region}...")

        # 1. Disable and delete job queue
        try:
            self._batch.update_job_queue(jobQueue=self._queue_name, state="DISABLED")
            log_fn(f"  Disabling queue: {self._queue_name}")
            time.sleep(5)
            self._batch.delete_job_queue(jobQueue=self._queue_name)
            log_fn(f"  Deleted queue: {self._queue_name}")
        except ClientError:
            log_fn(f"  Queue {self._queue_name} not found, skipping")

        # 2. Disable and delete compute environment
        try:
            self._batch.update_compute_environment(
                computeEnvironment=self._compute_env_name, state="DISABLED"
            )
            log_fn(f"  Disabling compute env: {self._compute_env_name}")
            time.sleep(10)
            self._batch.delete_compute_environment(
                computeEnvironment=self._compute_env_name
            )
            log_fn(f"  Deleted compute env: {self._compute_env_name}")
        except ClientError:
            log_fn(f"  Compute env {self._compute_env_name} not found, skipping")

        # 3. Delete security group
        try:
            sgs = self._ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [self._sg_name]}]
            )
            for sg in sgs.get("SecurityGroups", []):
                self._ec2.delete_security_group(GroupId=sg["GroupId"])
                log_fn(f"  Deleted security group: {sg['GroupId']}")
        except ClientError:
            pass

        # 4. Delete IAM roles
        for role_name in [self._job_role, self._exec_role,
                          self._instance_role, self._service_role]:
            self._delete_role(role_name, log_fn)

        # 5. Delete instance profile
        try:
            self._iam.remove_role_from_instance_profile(
                InstanceProfileName=self._instance_profile,
                RoleName=self._instance_role,
            )
        except ClientError:
            pass
        try:
            self._iam.delete_instance_profile(
                InstanceProfileName=self._instance_profile
            )
            log_fn(f"  Deleted instance profile: {self._instance_profile}")
        except ClientError:
            pass

        # 6. Delete S3 bucket (only if empty)
        try:
            objects = self._s3.list_objects_v2(
                Bucket=self._bucket_name, MaxKeys=1
            )
            if objects.get("KeyCount", 0) == 0:
                self._s3.delete_bucket(Bucket=self._bucket_name)
                log_fn(f"  Deleted bucket: {self._bucket_name}")
            else:
                log_fn(f"  Bucket {self._bucket_name} not empty, skipping")
        except ClientError:
            log_fn(f"  Bucket {self._bucket_name} not found, skipping")

        # 7. Delete log group
        try:
            self._logs.delete_log_group(logGroupName=f"/aws/batch/{self.PREFIX}")
            log_fn(f"  Deleted log group: /aws/batch/{self.PREFIX}")
        except ClientError:
            pass

        log_fn("Teardown complete.")

    # ─── Internal helpers ─────────────────────────────────────

    def _ensure_s3_bucket(self, log_fn, result: AWSSetupResult) -> str:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_bucket(Bucket=self._bucket_name)
            log_fn(f"  S3 bucket exists: {self._bucket_name}")
            result.skipped_resources.append(f"s3:{self._bucket_name}")
        except ClientError:
            create_args: dict[str, Any] = {"Bucket": self._bucket_name}
            if self.region != "us-east-1":
                create_args["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.region
                }
            self._s3.create_bucket(**create_args)
            self._s3.put_public_access_block(
                Bucket=self._bucket_name,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
            log_fn(f"  Created S3 bucket: {self._bucket_name}")
            result.created_resources.append(f"s3:{self._bucket_name}")
        return self._bucket_name

    def _ensure_role(
        self, role_name: str, service: str,
        policy_arns: list[str], log_fn, result: AWSSetupResult,
    ) -> str:
        from botocore.exceptions import ClientError

        try:
            resp = self._iam.get_role(RoleName=role_name)
            log_fn(f"  IAM role exists: {role_name}")
            result.skipped_resources.append(f"iam:{role_name}")
            return resp["Role"]["Arn"]
        except ClientError:
            pass

        trust = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": service},
                "Action": "sts:AssumeRole",
            }],
        })
        resp = self._iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=trust,
            Tags=[{"Key": "Project", "Value": "AstraTurbo"}],
        )
        for arn in policy_arns:
            self._iam.attach_role_policy(RoleName=role_name, PolicyArn=arn)
        log_fn(f"  Created IAM role: {role_name}")
        result.created_resources.append(f"iam:{role_name}")
        return resp["Role"]["Arn"]

    def _ensure_job_role(self, log_fn, result: AWSSetupResult) -> str:
        from botocore.exceptions import ClientError

        try:
            resp = self._iam.get_role(RoleName=self._job_role)
            log_fn(f"  IAM role exists: {self._job_role}")
            result.skipped_resources.append(f"iam:{self._job_role}")
            return resp["Role"]["Arn"]
        except ClientError:
            pass

        trust = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        })
        resp = self._iam.create_role(
            RoleName=self._job_role,
            AssumeRolePolicyDocument=trust,
            Tags=[{"Key": "Project", "Value": "AstraTurbo"}],
        )
        policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject",
                               "s3:ListBucket", "s3:DeleteObject"],
                    "Resource": [
                        f"arn:aws:s3:::{self._bucket_name}",
                        f"arn:aws:s3:::{self._bucket_name}/*",
                    ],
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                    "Resource": "arn:aws:logs:*:*:/aws/batch/*",
                },
            ],
        })
        self._iam.put_role_policy(
            RoleName=self._job_role,
            PolicyName=f"{self.PREFIX}-job-policy",
            PolicyDocument=policy,
        )
        log_fn(f"  Created IAM role: {self._job_role}")
        result.created_resources.append(f"iam:{self._job_role}")
        return resp["Role"]["Arn"]

    def _ensure_instance_profile(self, log_fn, result: AWSSetupResult) -> str:
        from botocore.exceptions import ClientError

        try:
            resp = self._iam.get_instance_profile(
                InstanceProfileName=self._instance_profile
            )
            result.skipped_resources.append(f"iam-profile:{self._instance_profile}")
            return resp["InstanceProfile"]["Arn"]
        except ClientError:
            pass

        resp = self._iam.create_instance_profile(
            InstanceProfileName=self._instance_profile
        )
        self._iam.add_role_to_instance_profile(
            InstanceProfileName=self._instance_profile,
            RoleName=self._instance_role,
        )
        log_fn(f"  Created instance profile: {self._instance_profile}")
        result.created_resources.append(f"iam-profile:{self._instance_profile}")
        # IAM propagation delay
        time.sleep(10)
        return resp["InstanceProfile"]["Arn"]

    def _get_default_vpc(self) -> tuple[str, list[str]]:
        vpcs = self._ec2.describe_vpcs(
            Filters=[{"Name": "isDefault", "Values": ["true"]}]
        )
        if not vpcs["Vpcs"]:
            raise RuntimeError(
                "No default VPC found. Create one with: "
                "aws ec2 create-default-vpc --region " + self.region
            )
        vpc_id = vpcs["Vpcs"][0]["VpcId"]
        subnets = self._ec2.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        subnet_ids = [s["SubnetId"] for s in subnets["Subnets"][:3]]
        return vpc_id, subnet_ids

    def _ensure_security_group(
        self, vpc_id: str, log_fn, result: AWSSetupResult,
    ) -> str:
        from botocore.exceptions import ClientError

        sgs = self._ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [self._sg_name]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )
        if sgs["SecurityGroups"]:
            sg_id = sgs["SecurityGroups"][0]["GroupId"]
            log_fn(f"  Security group exists: {sg_id}")
            result.skipped_resources.append(f"sg:{sg_id}")
            return sg_id

        resp = self._ec2.create_security_group(
            GroupName=self._sg_name,
            Description="AstraTurbo Batch — outbound-only for S3/ECR/CloudWatch",
            VpcId=vpc_id,
        )
        sg_id = resp["GroupId"]
        log_fn(f"  Created security group: {sg_id}")
        result.created_resources.append(f"sg:{sg_id}")
        return sg_id

    def _ensure_compute_environment(
        self, result: AWSSetupResult, log_fn,
    ) -> str:
        from botocore.exceptions import ClientError

        try:
            resp = self._batch.describe_compute_environments(
                computeEnvironments=[self._compute_env_name]
            )
            envs = resp.get("computeEnvironments", [])
            if envs and envs[0]["status"] != "DELETED":
                log_fn(f"  Compute env exists: {self._compute_env_name}")
                result.skipped_resources.append(f"batch-ce:{self._compute_env_name}")
                return self._compute_env_name
        except ClientError:
            pass

        compute_resources: dict[str, Any] = {
            "maxvCpus": self.max_vcpus,
            "subnets": result.subnet_ids,
            "securityGroupIds": [result.security_group_id],
        }

        if self.platform == "EC2":
            compute_resources.update({
                "type": "EC2",
                "minvCpus": 0,
                "desiredvCpus": 0,
                "instanceTypes": self.instance_types,
                "instanceRole": (
                    f"arn:aws:iam::{self._account_id}:"
                    f"instance-profile/{self._instance_profile}"
                ),
                "tags": {"Project": "AstraTurbo"},
            })
        else:
            compute_resources.update({
                "type": "FARGATE",
                "maxvCpus": self.max_vcpus,
            })

        self._batch.create_compute_environment(
            computeEnvironmentName=self._compute_env_name,
            type="MANAGED",
            state="ENABLED",
            computeResources=compute_resources,
            serviceRole=result.service_role_arn,
        )
        log_fn(f"  Created compute env: {self._compute_env_name} ({self.platform})")
        result.created_resources.append(f"batch-ce:{self._compute_env_name}")

        # Wait for VALID state
        for _ in range(30):
            resp = self._batch.describe_compute_environments(
                computeEnvironments=[self._compute_env_name]
            )
            envs = resp.get("computeEnvironments", [])
            if envs and envs[0].get("status") == "VALID":
                break
            time.sleep(5)

        return self._compute_env_name

    def _ensure_job_queue(self, log_fn, result: AWSSetupResult) -> str:
        from botocore.exceptions import ClientError

        try:
            resp = self._batch.describe_job_queues(
                jobQueues=[self._queue_name]
            )
            queues = resp.get("jobQueues", [])
            if queues and queues[0]["status"] != "DELETED":
                log_fn(f"  Job queue exists: {self._queue_name}")
                result.skipped_resources.append(f"batch-jq:{self._queue_name}")
                return self._queue_name
        except ClientError:
            pass

        self._batch.create_job_queue(
            jobQueueName=self._queue_name,
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[{
                "order": 1,
                "computeEnvironment": result.compute_environment,
            }],
        )
        log_fn(f"  Created job queue: {self._queue_name}")
        result.created_resources.append(f"batch-jq:{self._queue_name}")
        return self._queue_name

    def _ensure_log_group(self, log_fn, result: AWSSetupResult) -> None:
        from botocore.exceptions import ClientError

        group_name = f"/aws/batch/{self.PREFIX}"
        try:
            self._logs.create_log_group(logGroupName=group_name)
            log_fn(f"  Created log group: {group_name}")
            result.created_resources.append(f"logs:{group_name}")
        except ClientError:
            result.skipped_resources.append(f"logs:{group_name}")

    def _delete_role(self, role_name: str, log_fn) -> None:
        from botocore.exceptions import ClientError

        try:
            # Detach managed policies
            policies = self._iam.list_attached_role_policies(RoleName=role_name)
            for p in policies.get("AttachedPolicies", []):
                self._iam.detach_role_policy(
                    RoleName=role_name, PolicyArn=p["PolicyArn"]
                )
            # Delete inline policies
            inline = self._iam.list_role_policies(RoleName=role_name)
            for p_name in inline.get("PolicyNames", []):
                self._iam.delete_role_policy(
                    RoleName=role_name, PolicyName=p_name
                )
            self._iam.delete_role(RoleName=role_name)
            log_fn(f"  Deleted role: {role_name}")
        except ClientError:
            pass
