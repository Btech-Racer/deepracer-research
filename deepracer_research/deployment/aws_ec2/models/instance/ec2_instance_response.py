from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class EC2InstanceResponse:
    """Response data from EC2 instance creation

    Parameters
    ----------
    instance_id : str
        EC2 instance ID
    ami_id : str
        Amazon Machine Image ID used to launch the instance.
    instance_type : str
        EC2 instance type
    region : str
        AWS region where the instance was launched.
    launch_time : datetime, optional
        Time when the instance was launched, by default None.
    """

    instance_id: str
    ami_id: str
    instance_type: str
    region: str
    launch_time: Optional[datetime] = None

    def validate(self) -> None:
        """Validate the response data.

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

        if not self.instance_type:
            raise ValueError("Instance type is required")

        if not self.region:
            raise ValueError("Region is required")
