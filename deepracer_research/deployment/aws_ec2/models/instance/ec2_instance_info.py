from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from deepracer_research.deployment.aws_ec2.enum.instance_status import EC2InstanceStatus


@dataclass
class EC2InstanceInfo:
    """Basic EC2 instance information for listing

    Parameters
    ----------
    instance_id : str
        EC2 instance ID.
    instance_type : str
        EC2 instance type.
    status : EC2InstanceStatus
        Current instance status.
    public_ip : str, optional
        Public IP address, by default None.
    private_ip : str, optional
        Private IP address, by default None.
    name : str, optional
        Instance name from Name tag, by default None.
    launch_time : datetime, optional
        Time when instance was launched, by default None.
    """

    instance_id: str
    instance_type: str
    status: EC2InstanceStatus
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    name: Optional[str] = None
    launch_time: Optional[datetime] = None

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
    def is_ready_for_ssh(self) -> bool:
        """Check if instance is ready for SSH connections.

        Returns
        -------
        bool
            True if instance can accept SSH connections.
        """
        return self.status.is_ready_for_ssh and self.public_ip is not None
