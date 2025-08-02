from dataclasses import dataclass
from typing import Optional


@dataclass
class EC2DeploymentResult:
    """Result of EC2 deployment operation"""

    success: bool
    instance_id: Optional[str] = None
    hostname: Optional[str] = None
    ssh_ready: bool = False
    deepracer_installed: bool = False
    error_message: Optional[str] = None
