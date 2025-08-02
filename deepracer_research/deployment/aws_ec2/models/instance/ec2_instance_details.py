from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from deepracer_research.deployment.aws_ec2.enum.instance_status import EC2InstanceStatus
from deepracer_research.deployment.aws_ec2.enum.instance_type import EC2InstanceType
from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion


@dataclass
class EC2InstanceDetails:
    """Detailed EC2 instance information

    Parameters
    ----------
    instance_id : str
        EC2 instance ID.
    instance_type : EC2InstanceType
        EC2 instance type enum.
    status : EC2InstanceStatus
        Current instance status.
    region : AWSRegion
        AWS region where instance is located.
    ami_id : str
        Amazon Machine Image ID.
    vpc_id : str, optional
        VPC ID where instance is located, by default None.
    subnet_id : str, optional
        Subnet ID where instance is located, by default None.
    security_groups : List[str], optional
        List of security group IDs, by default empty list.
    public_ip : str, optional
        Public IP address, by default None.
    private_ip : str, optional
        Private IP address, by default None.
    public_dns : str, optional
        Public DNS name, by default None.
    private_dns : str, optional
        Private DNS name, by default None.
    key_name : str, optional
        EC2 key pair name for SSH access, by default None.
    name : str, optional
        Instance name from Name tag, by default None.
    launch_time : datetime, optional
        Time when instance was launched, by default None.
    tags : Dict[str, str], optional
        Instance tags, by default empty dict.
    user_data : str, optional
        User data script, by default None.
    instance_profile : str, optional
        IAM instance profile name, by default None.
    monitoring_enabled : bool, optional
        Whether detailed monitoring is enabled, by default False.
    """

    instance_id: str
    instance_type: EC2InstanceType
    status: EC2InstanceStatus
    region: AWSRegion
    ami_id: str
    vpc_id: Optional[str] = None
    subnet_id: Optional[str] = None
    security_groups: List[str] = None
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    public_dns: Optional[str] = None
    private_dns: Optional[str] = None
    key_name: Optional[str] = None
    name: Optional[str] = None
    launch_time: Optional[datetime] = None
    tags: Dict[str, str] = None
    user_data: Optional[str] = None
    instance_profile: Optional[str] = None
    monitoring_enabled: bool = False

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.security_groups is None:
            self.security_groups = []
        if self.tags is None:
            self.tags = {}

    @property
    def is_ready_for_ssh(self) -> bool:
        """Check if instance is ready for SSH connections.

        Returns
        -------
        bool
            True if instance can accept SSH connections.
        """
        return self.status.is_ready_for_ssh and self.public_ip is not None and self.key_name is not None

    @property
    def is_billable(self) -> bool:
        """Check if instance is consuming billable resources.

        Returns
        -------
        bool
            True if instance is consuming compute resources.
        """
        return self.status.is_billable

    @property
    def display_name(self) -> str:
        """Get a display-friendly name for the instance.

        Returns
        -------
        str
            Display name combining name and instance ID.
        """
        if self.name:
            return f"{self.name} ({self.instance_id})"
        return self.instance_id

    @property
    def resource_summary(self) -> str:
        """Get a summary of instance resources.

        Returns
        -------
        str
            Summary of instance resources.
        """
        if self.instance_type.has_gpu:
            gpu_str = f"{self.instance_type.gpu_count}x {self.instance_type.gpu_type}"
            return f"{self.instance_type.vcpus} vCPU, {self.instance_type.memory_gb}GB RAM, {gpu_str}"
        else:
            return f"{self.instance_type.vcpus} vCPU, {self.instance_type.memory_gb}GB RAM"

    @property
    def cost_estimate_hourly(self) -> float:
        """Get hourly cost estimate.

        Returns
        -------
        float
            Estimated hourly cost in USD.
        """
        return self.instance_type.hourly_cost_estimate

    def get_ssh_command(self, username: str = "ubuntu", private_key_path: Optional[str] = None) -> str:
        """Generate SSH command for connecting to the instance.

        Parameters
        ----------
        username : str, optional
            SSH username, by default "ubuntu".
        private_key_path : str, optional
            Path to private key file, by default None.

        Returns
        -------
        str
            SSH command string.

        Raises
        ------
        ValueError
            If instance is not ready for SSH or missing required information.
        """
        if not self.is_ready_for_ssh:
            raise ValueError("Instance is not ready for SSH connection")

        cmd_parts = ["ssh"]

        if private_key_path:
            cmd_parts.extend(["-i", private_key_path])
        elif self.key_name:
            cmd_parts.extend(["-i", f"~/.ssh/{self.key_name}.pem"])

        cmd_parts.extend(["-o", "StrictHostKeyChecking=no", f"{username}@{self.public_ip}"])

        return " ".join(cmd_parts)

    def validate(self) -> None:
        """Validate the instance details.

        Raises
        ------
        ValueError
            If any required field is missing or invalid.
        """
        if not self.instance_id:
            raise ValueError("Instance ID is required")

        if not self.instance_id.startswith("i-"):
            raise ValueError("Instance ID must start with 'i-'")

        if not self.ami_id:
            raise ValueError("AMI ID is required")

        if not self.ami_id.startswith("ami-"):
            raise ValueError("AMI ID must start with 'ami-'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance details to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of instance details.
        """
        return {
            "instance_id": self.instance_id,
            "instance_type": self.instance_type.value,
            "status": self.status.value,
            "region": self.region.value,
            "ami_id": self.ami_id,
            "vpc_id": self.vpc_id,
            "subnet_id": self.subnet_id,
            "security_groups": self.security_groups,
            "public_ip": self.public_ip,
            "private_ip": self.private_ip,
            "public_dns": self.public_dns,
            "private_dns": self.private_dns,
            "key_name": self.key_name,
            "name": self.name,
            "launch_time": self.launch_time.isoformat() if self.launch_time else None,
            "tags": self.tags,
            "instance_profile": self.instance_profile,
            "monitoring_enabled": self.monitoring_enabled,
            "resource_summary": self.resource_summary,
            "hourly_cost_estimate": self.cost_estimate_hourly,
        }
