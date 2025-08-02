from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VPCInfo:
    """VPC information for EC2 deployments

    Parameters
    ----------
    vpc_id : str
        VPC ID.
    subnet_id : str
        Subnet ID for instance placement.
    security_group_ids : List[str], optional
        List of security group IDs, by default empty list.
    internet_gateway_id : str, optional
        Internet gateway ID, by default None.
    """

    vpc_id: str
    subnet_id: str
    security_group_ids: List[str] = None
    internet_gateway_id: Optional[str] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.security_group_ids is None:
            self.security_group_ids = []

    def validate(self) -> None:
        """Validate VPC information.

        Raises
        ------
        ValueError
            If any required field is missing or invalid.
        """
        if not self.vpc_id:
            raise ValueError("VPC ID is required")

        if not self.vpc_id.startswith("vpc-"):
            raise ValueError("VPC ID must start with 'vpc-'")

        if not self.subnet_id:
            raise ValueError("Subnet ID is required")

        if not self.subnet_id.startswith("subnet-"):
            raise ValueError("Subnet ID must start with 'subnet-'")

        for sg_id in self.security_group_ids:
            if not sg_id.startswith("sg-"):
                raise ValueError(f"Security group ID must start with 'sg-': {sg_id}")
