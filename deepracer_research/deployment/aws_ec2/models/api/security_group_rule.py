from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SecurityGroupRule:
    """Security group rule information

    Parameters
    ----------
    protocol : str
        IP protocol (tcp, udp, icmp, or -1 for all).
    from_port : int, optional
        Start of port range, by default None.
    to_port : int, optional
        End of port range, by default None.
    cidr_blocks : List[str], optional
        List of CIDR blocks, by default empty list.
    source_security_group_id : str, optional
        Source security group ID, by default None.
    description : str, optional
        Rule description, by default None.
    """

    protocol: str
    from_port: Optional[int] = None
    to_port: Optional[int] = None
    cidr_blocks: List[str] = None
    source_security_group_id: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.cidr_blocks is None:
            self.cidr_blocks = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for AWS API.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation for AWS API.
        """
        rule = {"IpProtocol": self.protocol}

        if self.from_port is not None:
            rule["FromPort"] = self.from_port

        if self.to_port is not None:
            rule["ToPort"] = self.to_port

        if self.cidr_blocks:
            rule["IpRanges"] = [{"CidrIp": cidr} for cidr in self.cidr_blocks]

        if self.source_security_group_id:
            rule["UserIdGroupPairs"] = [{"GroupId": self.source_security_group_id}]

        if self.description:
            rule["Description"] = self.description

        return rule
