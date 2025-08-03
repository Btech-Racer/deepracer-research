import time
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from deepracer_research.deployment.aws_ec2.config.aws_config import AWSConfig
from deepracer_research.deployment.aws_ec2.enum.instance_status import EC2InstanceStatus
from deepracer_research.deployment.aws_ec2.enum.instance_type import EC2InstanceType
from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion
from deepracer_research.deployment.aws_ec2.models.api import EC2ApiError
from deepracer_research.deployment.aws_ec2.models.instance import EC2InstanceDetails, EC2InstanceInfo, EC2InstanceResponse
from deepracer_research.utils.logger import debug, error, info, warning


class EC2Client:
    """AWS EC2 client for instance management

    Parameters
    ----------
    aws_config : AWSConfig
        AWS configuration for EC2 operations.
    """

    def __init__(self, aws_config: AWSConfig):
        """Initialize EC2 client.

        Parameters
        ----------
        aws_config : AWSConfig
            AWS configuration.
        """
        self.aws_config = aws_config
        self.session = None
        self.ec2_client = None
        self.ec2_resource = None

        self._initialize_clients()

        info("EC2 client initialized", extra={"region": self.aws_config.region.value, "profile": self.aws_config.profile_name})

    def _initialize_clients(self) -> None:
        """Initialize boto3 session and clients."""
        try:
            session_kwargs = self.aws_config.get_boto3_session_kwargs()
            self.session = boto3.Session(**session_kwargs)
            self.ec2_client = self.session.client("ec2")
            self.ec2_resource = self.session.resource("ec2")
            self.ec2_client.describe_regions(RegionNames=[self.aws_config.region.value])

        except NoCredentialsError as e:
            error("AWS credentials not found", extra={"error": str(e)})
            raise EC2ApiError("AWS credentials not configured. Please configure AWS credentials.", error_code="NoCredentials")
        except ClientError as e:
            error("Failed to initialize AWS client", extra={"error": str(e)})
            raise EC2ApiError(
                f"Failed to initialize AWS EC2 client: {e}", error_code=e.response.get("Error", {}).get("Code", "ClientError")
            )

    def create_instance(
        self, instance_config: Dict[str, Any], wait_for_running: bool = True, timeout: int = 300
    ) -> EC2InstanceResponse:
        """Create a new EC2 instance.

        Parameters
        ----------
        instance_config : Dict[str, Any]
            Instance configuration dictionary.
        wait_for_running : bool, optional
            Wait for instance to be running, by default True.
        timeout : int, optional
            Timeout in seconds for waiting, by default 300.

        Returns
        -------
        EC2InstanceResponse
            Response containing instance information.

        Raises
        ------
        EC2ApiError
            If instance creation fails.
        """
        try:
            info("Creating EC2 instance", extra=instance_config)

            response = self.ec2_client.run_instances(**instance_config)
            instance = response["Instances"][0]
            instance_id = instance["InstanceId"]

            info(
                "EC2 instance created",
                extra={"instance_id": instance_id, "instance_type": instance["InstanceType"], "ami_id": instance["ImageId"]},
            )

            if wait_for_running:
                self._wait_for_instance_running(instance_id, timeout)

            return EC2InstanceResponse(
                instance_id=instance_id,
                ami_id=instance["ImageId"],
                instance_type=instance["InstanceType"],
                region=self.aws_config.region.value,
                launch_time=instance.get("LaunchTime"),
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error("EC2 instance creation failed", extra={"error_code": error_code, "error_message": error_message})

            raise EC2ApiError(
                f"Failed to create EC2 instance: {error_message}", error_code=error_code, operation="create_instance"
            )

    def get_instance(self, instance_id: str, include_terminated: bool = False) -> EC2InstanceDetails:
        """Get detailed information about an EC2 instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.
        include_terminated : bool, optional
            Include terminated instances, by default False.

        Returns
        -------
        EC2InstanceDetails
            Detailed instance information.

        Raises
        ------
        EC2ApiError
            If instance is not found or access fails.
        """
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])

            if not response["Reservations"]:
                raise EC2ApiError(
                    f"Instance {instance_id} not found",
                    error_code="InstanceNotFound",
                    instance_id=instance_id,
                    operation="get_instance",
                )

            instance = response["Reservations"][0]["Instances"][0]
            status_name = instance["State"]["Name"]

            if not include_terminated and status_name == "terminated":
                raise EC2ApiError(
                    f"Instance {instance_id} is terminated",
                    error_code="InstanceTerminated",
                    instance_id=instance_id,
                    operation="get_instance",
                )

            return self._parse_instance_details(instance)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to get instance details",
                extra={"instance_id": instance_id, "error_code": error_code, "error_message": error_message},
            )

            raise EC2ApiError(
                f"Failed to get instance {instance_id}: {error_message}",
                error_code=error_code,
                instance_id=instance_id,
                operation="get_instance",
            )

    def list_instances(self, include_terminated: bool = False, tags: Optional[Dict[str, str]] = None) -> List[EC2InstanceInfo]:
        """List EC2 instances in the configured region.

        Parameters
        ----------
        include_terminated : bool, optional
            Include terminated instances, by default False.
        tags : Dict[str, str], optional
            Filter by tags, by default None.

        Returns
        -------
        List[EC2InstanceInfo]
            List of instance information.

        Raises
        ------
        EC2ApiError
            If listing fails.
        """
        try:
            filters = []

            if not include_terminated:
                filters.append({"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]})

            if tags:
                for key, value in tags.items():
                    filters.append({"Name": f"tag:{key}", "Values": [value]})

            kwargs = {}
            if filters:
                kwargs["Filters"] = filters

            response = self.ec2_client.describe_instances(**kwargs)

            instances = []
            for reservation in response["Reservations"]:
                for instance in reservation["Instances"]:
                    instances.append(self._parse_instance_info(instance))

            info("Listed EC2 instances", extra={"count": len(instances), "include_terminated": include_terminated})

            return instances

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error("Failed to list instances", extra={"error_code": error_code, "error_message": error_message})

            raise EC2ApiError(f"Failed to list instances: {error_message}", error_code=error_code, operation="list_instances")

    def start_instance(self, instance_id: str) -> bool:
        """Start a stopped EC2 instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        bool
            True if start was successful.

        Raises
        ------
        EC2ApiError
            If start operation fails.
        """
        try:
            info("Starting EC2 instance", extra={"instance_id": instance_id})

            response = self.ec2_client.start_instances(InstanceIds=[instance_id])

            if response["StartingInstances"]:
                info("EC2 instance start initiated", extra={"instance_id": instance_id})
                return True
            else:
                warning("No instances were started", extra={"instance_id": instance_id})
                return False

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to start instance",
                extra={"instance_id": instance_id, "error_code": error_code, "error_message": error_message},
            )

            raise EC2ApiError(
                f"Failed to start instance {instance_id}: {error_message}",
                error_code=error_code,
                instance_id=instance_id,
                operation="start_instance",
            )

    def stop_instance(self, instance_id: str, force: bool = False) -> bool:
        """Stop a running EC2 instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.
        force : bool, optional
            Force stop the instance, by default False.

        Returns
        -------
        bool
            True if stop was successful.

        Raises
        ------
        EC2ApiError
            If stop operation fails.
        """
        try:
            info("Stopping EC2 instance", extra={"instance_id": instance_id, "force": force})

            response = self.ec2_client.stop_instances(InstanceIds=[instance_id], Force=force)

            if response["StoppingInstances"]:
                info("EC2 instance stop initiated", extra={"instance_id": instance_id})
                return True
            else:
                warning("No instances were stopped", extra={"instance_id": instance_id})
                return False

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to stop instance",
                extra={"instance_id": instance_id, "error_code": error_code, "error_message": error_message},
            )

            raise EC2ApiError(
                f"Failed to stop instance {instance_id}: {error_message}",
                error_code=error_code,
                instance_id=instance_id,
                operation="stop_instance",
            )

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an EC2 instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        bool
            True if termination was successful.

        Raises
        ------
        EC2ApiError
            If termination fails.
        """
        try:
            info("Terminating EC2 instance", extra={"instance_id": instance_id})

            response = self.ec2_client.terminate_instances(InstanceIds=[instance_id])

            if response["TerminatingInstances"]:
                info("EC2 instance termination initiated", extra={"instance_id": instance_id})
                return True
            else:
                warning("No instances were terminated", extra={"instance_id": instance_id})
                return False

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to terminate instance",
                extra={"instance_id": instance_id, "error_code": error_code, "error_message": error_message},
            )

            raise EC2ApiError(
                f"Failed to terminate instance {instance_id}: {error_message}",
                error_code=error_code,
                instance_id=instance_id,
                operation="terminate_instance",
            )

    def _wait_for_instance_running(self, instance_id: str, timeout: int = 300) -> None:
        """Wait for instance to reach running state.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.
        timeout : int, optional
            Timeout in seconds, by default 300.

        Raises
        ------
        EC2ApiError
            If timeout is reached or instance fails.
        """
        info("Waiting for instance to be running", extra={"instance_id": instance_id, "timeout": timeout})

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                instance = self.get_instance(instance_id, include_terminated=True)

                if instance.status == EC2InstanceStatus.RUNNING:
                    info("Instance is running", extra={"instance_id": instance_id})
                    return
                elif instance.status == EC2InstanceStatus.TERMINATED:
                    raise EC2ApiError(
                        f"Instance {instance_id} was terminated during startup",
                        error_code="InstanceTerminated",
                        instance_id=instance_id,
                        operation="wait_for_running",
                    )

                debug("Instance not yet running", extra={"instance_id": instance_id, "status": instance.status.value})

                time.sleep(10)

            except EC2ApiError:
                raise
            except Exception as e:
                warning("Error checking instance status", extra={"instance_id": instance_id, "error": str(e)})
                time.sleep(10)

        raise EC2ApiError(
            f"Timeout waiting for instance {instance_id} to be running",
            error_code="Timeout",
            instance_id=instance_id,
            operation="wait_for_running",
        )

    def _parse_instance_details(self, instance_data: Dict[str, Any]) -> EC2InstanceDetails:
        """Parse AWS instance data into EC2InstanceDetails.

        Parameters
        ----------
        instance_data : Dict[str, Any]
            Raw instance data from AWS API.

        Returns
        -------
        EC2InstanceDetails
            Parsed instance details.
        """
        tags = {}
        name = None
        for tag in instance_data.get("Tags", []):
            tags[tag["Key"]] = tag["Value"]
            if tag["Key"] == "Name":
                name = tag["Value"]

        status = EC2InstanceStatus(instance_data["State"]["Name"])

        instance_type = EC2InstanceType(instance_data["InstanceType"])

        region = AWSRegion(self.aws_config.region.value)

        return EC2InstanceDetails(
            instance_id=instance_data["InstanceId"],
            instance_type=instance_type,
            status=status,
            region=region,
            ami_id=instance_data["ImageId"],
            vpc_id=instance_data.get("VpcId"),
            subnet_id=instance_data.get("SubnetId"),
            security_groups=[sg["GroupId"] for sg in instance_data.get("SecurityGroups", [])],
            public_ip=instance_data.get("PublicIpAddress"),
            private_ip=instance_data.get("PrivateIpAddress"),
            public_dns=instance_data.get("PublicDnsName"),
            private_dns=instance_data.get("PrivateDnsName"),
            key_name=instance_data.get("KeyName"),
            name=name,
            launch_time=instance_data.get("LaunchTime"),
            tags=tags,
            instance_profile=instance_data.get("IamInstanceProfile", {}).get("Arn"),
            monitoring_enabled=instance_data.get("Monitoring", {}).get("State") == "enabled",
        )

    def _parse_instance_info(self, instance_data: Dict[str, Any]) -> EC2InstanceInfo:
        """Parse AWS instance data into EC2InstanceInfo.

        Parameters
        ----------
        instance_data : Dict[str, Any]
            Raw instance data from AWS API.

        Returns
        -------
        EC2InstanceInfo
            Parsed instance information.
        """
        name = None
        for tag in instance_data.get("Tags", []):
            if tag["Key"] == "Name":
                name = tag["Value"]
                break

        status = EC2InstanceStatus(instance_data["State"]["Name"])

        return EC2InstanceInfo(
            instance_id=instance_data["InstanceId"],
            instance_type=instance_data["InstanceType"],
            status=status,
            public_ip=instance_data.get("PublicIpAddress"),
            private_ip=instance_data.get("PrivateIpAddress"),
            name=name,
            launch_time=instance_data.get("LaunchTime"),
        )
