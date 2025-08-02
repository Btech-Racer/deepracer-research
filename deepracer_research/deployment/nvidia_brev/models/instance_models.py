from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional

from deepracer_research.deployment.nvidia_brev.enum.deployment_mode import DeploymentMode
from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType
from deepracer_research.deployment.nvidia_brev.enum.instance_template import InstanceTemplate


class InstanceStatus(StrEnum):
    """Instance status enumeration for NVIDIA Brev instances"""

    PENDING = "pending"

    RUNNING = "running"

    STOPPING = "stopping"

    STOPPED = "stopped"

    TERMINATING = "terminating"

    TERMINATED = "terminated"

    ERROR = "error"

    UNKNOWN = "unknown"

    @property
    def is_active(self) -> bool:
        """Check if instance is in an active state.

        Returns
        -------
        bool
            True if instance is running or pending
        """
        return self in {self.PENDING, self.RUNNING}

    @property
    def is_terminal(self) -> bool:
        """Check if instance is in a terminal state.

        Returns
        -------
        bool
            True if instance is terminated or in error state
        """
        return self in {self.TERMINATED, self.ERROR}

    @property
    def can_connect(self) -> bool:
        """Check if instance can be connected to via SSH.

        Returns
        -------
        bool
            True if instance is running
        """
        return self == self.RUNNING


@dataclass
class InstanceMetrics:
    """Metrics for a running NVIDIA Brev instance

    Parameters
    ----------
    cpu_utilization : float, optional
        CPU utilization percentage (0-100), by default None
    memory_utilization : float, optional
        Memory utilization percentage (0-100), by default None
    gpu_utilization : float, optional
        GPU utilization percentage (0-100), by default None
    gpu_memory_utilization : float, optional
        GPU memory utilization percentage (0-100), by default None
    disk_utilization : float, optional
        Disk utilization percentage (0-100), by default None
    network_in_mbps : float, optional
        Network input in Mbps, by default None
    network_out_mbps : float, optional
        Network output in Mbps, by default None
    uptime_seconds : int, optional
        Instance uptime in seconds, by default None
    cost_per_hour : float, optional
        Current cost per hour in USD, by default None
    total_cost : float, optional
        Total accumulated cost in USD, by default None
    """

    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_utilization: Optional[float] = None
    disk_utilization: Optional[float] = None
    network_in_mbps: Optional[float] = None
    network_out_mbps: Optional[float] = None
    uptime_seconds: Optional[int] = None
    cost_per_hour: Optional[float] = None
    total_cost: Optional[float] = None

    @property
    def uptime_hours(self) -> Optional[float]:
        """Get uptime in hours.

        Returns
        -------
        Optional[float]
            Uptime in hours if available
        """
        if self.uptime_seconds is not None:
            return self.uptime_seconds / 3600
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of metrics
        """
        return {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "disk_utilization": self.disk_utilization,
            "network_in_mbps": self.network_in_mbps,
            "network_out_mbps": self.network_out_mbps,
            "uptime_seconds": self.uptime_seconds,
            "uptime_hours": self.uptime_hours,
            "cost_per_hour": self.cost_per_hour,
            "total_cost": self.total_cost,
        }


@dataclass
class InstanceDetails:
    """Detailed information about an NVIDIA Brev instance

    Parameters
    ----------
    instance_id : str
        Unique instance identifier
    name : str
        Instance name
    status : InstanceStatus
        Current instance status
    gpu_type : GPUType
        GPU type attached to instance
    template : InstanceTemplate
        Software template used
    deployment_mode : DeploymentMode
        Deployment mode (on-demand, spot, etc.)
    num_gpus : int
        Number of GPUs attached
    cpu_cores : int
        Number of CPU cores
    memory_gb : int
        RAM size in GB
    disk_size_gb : int
        Disk size in GB
    public_ip : str, optional
        Public IP address, by default None
    private_ip : str, optional
        Private IP address, by default None
    ssh_host : str, optional
        SSH connection host, by default None
    ssh_port : int, optional
        SSH port, by default 22
    region : str, optional
        Deployment region, by default None
    created_at : datetime, optional
        Instance creation timestamp, by default None
    started_at : datetime, optional
        Instance start timestamp, by default None
    stopped_at : datetime, optional
        Instance stop timestamp, by default None
    ports : List[int], optional
        Exposed ports, by default empty list
    environment_variables : Dict[str, str], optional
        Environment variables, by default empty dict
    tags : Dict[str, str], optional
        Instance tags, by default empty dict
    metrics : InstanceMetrics, optional
        Instance metrics, by default None
    auto_shutdown_at : datetime, optional
        Scheduled auto-shutdown time, by default None
    """

    instance_id: str
    name: str
    status: InstanceStatus
    gpu_type: GPUType
    template: InstanceTemplate
    deployment_mode: DeploymentMode
    num_gpus: int
    cpu_cores: int
    memory_gb: int
    disk_size_gb: int

    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    region: Optional[str] = None

    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    ports: List[int] = None
    environment_variables: Dict[str, str] = None
    tags: Dict[str, str] = None

    metrics: Optional[InstanceMetrics] = None
    auto_shutdown_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization setup."""
        if self.ports is None:
            self.ports = []
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.tags is None:
            self.tags = {}

    @property
    def is_running(self) -> bool:
        """Check if instance is running.

        Returns
        -------
        bool
            True if instance is running
        """
        return self.status == InstanceStatus.RUNNING

    @property
    def can_connect(self) -> bool:
        """Check if instance can be connected to.

        Returns
        -------
        bool
            True if instance can be connected to via SSH
        """
        return self.status.can_connect and bool(self.ssh_host or self.public_ip)

    @property
    def connection_host(self) -> Optional[str]:
        """Get the host to use for SSH connections.

        Returns
        -------
        Optional[str]
            SSH host, preferring ssh_host over public_ip
        """
        return self.ssh_host or self.public_ip

    @property
    def total_memory_gb(self) -> int:
        """Get total memory including GPU memory.

        Returns
        -------
        int
            Total memory in GB (RAM + GPU memory)
        """
        gpu_memory = self.gpu_type.memory_gb * self.num_gpus
        return self.memory_gb + gpu_memory

    def get_cost_estimate(self, hours: Optional[float] = None) -> Dict[str, float]:
        """Get cost estimate for the instance.

        Parameters
        ----------
        hours : Optional[float], optional
            Number of hours to estimate for, by default None (use uptime)

        Returns
        -------
        Dict[str, float]
            Cost estimate information
        """
        if hours is None and self.metrics and self.metrics.uptime_hours:
            hours = self.metrics.uptime_hours
        elif hours is None:
            hours = 1.0

        cost_per_hour = self.metrics.cost_per_hour if self.metrics else 1.0
        total_cost = cost_per_hour * hours

        return {
            "hours": hours,
            "cost_per_hour": cost_per_hour,
            "total_cost": total_cost,
            "deployment_mode": self.deployment_mode.value,
            "cost_multiplier": self.deployment_mode.cost_multiplier,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance details to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of instance details
        """
        data = {
            "instance_id": self.instance_id,
            "name": self.name,
            "status": self.status.value,
            "gpu_type": self.gpu_type.value,
            "template": self.template.value,
            "deployment_mode": self.deployment_mode.value,
            "num_gpus": self.num_gpus,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_size_gb": self.disk_size_gb,
            "public_ip": self.public_ip,
            "private_ip": self.private_ip,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "region": self.region,
            "ports": self.ports,
            "environment_variables": self.environment_variables,
            "tags": self.tags,
            "auto_shutdown_at": self.auto_shutdown_at.isoformat() if self.auto_shutdown_at else None,
        }

        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.stopped_at:
            data["stopped_at"] = self.stopped_at.isoformat()

        if self.metrics:
            data["metrics"] = self.metrics.to_dict()

        return data


@dataclass
class InstanceResponse:
    """Response from NVIDIA Brev API for instance operations

    Parameters
    ----------
    success : bool
        Whether the operation was successful
    instance : InstanceDetails, optional
        Instance details if available, by default None
    message : str, optional
        Response message, by default ""
    error_code : str, optional
        Error code if operation failed, by default None
    request_id : str, optional
        Unique request identifier, by default None
    """

    success: bool
    instance: Optional[InstanceDetails] = None
    message: str = ""
    error_code: Optional[str] = None
    request_id: Optional[str] = None

    @property
    def has_error(self) -> bool:
        """Check if response contains an error.

        Returns
        -------
        bool
            True if response has an error
        """
        return not self.success or bool(self.error_code)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of response
        """
        data = {"success": self.success, "message": self.message, "error_code": self.error_code, "request_id": self.request_id}

        if self.instance:
            data["instance"] = self.instance.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstanceResponse":
        """Create response from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing response data

        Returns
        -------
        InstanceResponse
            Response instance
        """
        instance = None
        if "instance" in data and data["instance"]:
            instance_data = data["instance"]
            instance = InstanceDetails(
                instance_id=instance_data["instance_id"],
                name=instance_data["name"],
                status=InstanceStatus(instance_data["status"]),
                gpu_type=GPUType(instance_data["gpu_type"]),
                template=InstanceTemplate(instance_data["template"]),
                deployment_mode=DeploymentMode(instance_data["deployment_mode"]),
                num_gpus=instance_data["num_gpus"],
                cpu_cores=instance_data["cpu_cores"],
                memory_gb=instance_data["memory_gb"],
                disk_size_gb=instance_data["disk_size_gb"],
                public_ip=instance_data.get("public_ip"),
                private_ip=instance_data.get("private_ip"),
                ssh_host=instance_data.get("ssh_host"),
                ssh_port=instance_data.get("ssh_port", 22),
                region=instance_data.get("region"),
                ports=instance_data.get("ports", []),
                environment_variables=instance_data.get("environment_variables", {}),
                tags=instance_data.get("tags", {}),
                created_at=datetime.fromisoformat(instance_data["created_at"]) if instance_data.get("created_at") else None,
                started_at=datetime.fromisoformat(instance_data["started_at"]) if instance_data.get("started_at") else None,
                stopped_at=datetime.fromisoformat(instance_data["stopped_at"]) if instance_data.get("stopped_at") else None,
                auto_shutdown_at=(
                    datetime.fromisoformat(instance_data["auto_shutdown_at"]) if instance_data.get("auto_shutdown_at") else None
                ),
            )

            if "metrics" in instance_data and instance_data["metrics"]:
                metrics_data = instance_data["metrics"]
                instance.metrics = InstanceMetrics(**metrics_data)

        return cls(
            success=data["success"],
            instance=instance,
            message=data.get("message", ""),
            error_code=data.get("error_code"),
            request_id=data.get("request_id"),
        )
