from dataclasses import dataclass
from typing import Optional

from deepracer_research.deployment.thunder_compute.models.instance_models import InstanceDetails


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""

    success: bool
    instance_uuid: str
    instance_details: Optional[InstanceDetails]
    ssh_ready: bool
    deepracer_installed: bool
    error_message: Optional[str] = None
