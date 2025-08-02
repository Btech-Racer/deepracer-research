from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepracer_research.deployment.nvidia_brev.enum.deployment_mode import DeploymentMode
from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType
from deepracer_research.deployment.nvidia_brev.enum.instance_template import InstanceTemplate


@dataclass
class CreateInstanceRequest:
    """Request to create a new NVIDIA Brev instance

    Parameters
    ----------
    name : str
        Instance name
    template : InstanceTemplate
        Software template to use
    gpu_type : GPUType
        GPU type to attach
    deployment_mode : DeploymentMode, optional
        Deployment mode, by default ON_DEMAND
    num_gpus : int, optional
        Number of GPUs, by default 1
    cpu_cores : int, optional
        Number of CPU cores, by default 4
    memory_gb : int, optional
        RAM size in GB, by default None (auto-calculated)
    disk_size_gb : int, optional
        Disk size in GB, by default 100
    region : str, optional
        Preferred region, by default None
    ports : List[int], optional
        Ports to expose, by default [22, 8888]
    environment_variables : Dict[str, str], optional
        Environment variables, by default empty dict
    setup_script : str, optional
        Setup script to run, by default None
    auto_shutdown_hours : int, optional
        Auto shutdown after N hours, by default None
    tags : Dict[str, str], optional
        Instance tags, by default empty dict
    """

    name: str
    template: InstanceTemplate
    gpu_type: GPUType
    deployment_mode: DeploymentMode = DeploymentMode.ON_DEMAND
    num_gpus: int = 1
    cpu_cores: int = 4
    memory_gb: Optional[int] = None
    disk_size_gb: int = 100
    region: Optional[str] = None
    ports: List[int] = field(default_factory=lambda: [22, 8888])
    environment_variables: Dict[str, str] = field(default_factory=dict)
    setup_script: Optional[str] = None
    auto_shutdown_hours: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for API call.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for API
        """
        data = {
            "name": self.name,
            "template": self.template.value,
            "gpu_type": self.gpu_type.value,
            "deployment_mode": self.deployment_mode.value,
            "num_gpus": self.num_gpus,
            "cpu_cores": self.cpu_cores,
            "disk_size_gb": self.disk_size_gb,
            "ports": self.ports,
            "environment_variables": self.environment_variables,
            "tags": self.tags,
        }

        if self.memory_gb is not None:
            data["memory_gb"] = self.memory_gb

        if self.region:
            data["region"] = self.region

        if self.setup_script:
            data["setup_script"] = self.setup_script

        if self.auto_shutdown_hours:
            data["auto_shutdown_hours"] = self.auto_shutdown_hours

        return data

    def validate(self) -> None:
        """Validate the request parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if not self.name or not self.name.strip():
            raise ValueError("Instance name is required")

        if self.num_gpus <= 0:
            raise ValueError("Number of GPUs must be positive")

        if self.cpu_cores <= 0:
            raise ValueError("Number of CPU cores must be positive")

        if self.memory_gb is not None and self.memory_gb <= 0:
            raise ValueError("Memory size must be positive")

        if self.disk_size_gb <= 0:
            raise ValueError("Disk size must be positive")

        if self.auto_shutdown_hours is not None and self.auto_shutdown_hours <= 0:
            raise ValueError("Auto shutdown hours must be positive")

        for port in self.ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Invalid port number: {port}")
