from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class InstanceStatus(StrEnum):
    """Instance status enumeration"""

    CREATING = "creating"

    RUNNING = "running"

    STOPPED = "stopped"

    DELETING = "deleting"

    FAILED = "failed"

    STARTING = "starting"

    @property
    def is_ready_for_ssh(self) -> bool:
        """Check if instance is ready for SSH connections.

        Returns
        -------
        bool
            True if instance can accept SSH connections.
        """
        return self == self.RUNNING

    @property
    def is_billable(self) -> bool:
        """Check if instance is consuming billable resources.

        Returns
        -------
        bool
            True if instance is consuming compute resources.
        """
        return self in [self.CREATING, self.RUNNING, self.STARTING]


@dataclass
class InstanceResponse:
    """Response data from instance creation

    Parameters
    ----------
    uuid : str
        Unique identifier for the created instance.
    key : str
        Authentication key for instance access.
    identifier : str
        Identifier for the instance.
    """

    uuid: str
    key: str
    identifier: str

    def validate(self) -> None:
        """Validate the response data.

        Raises
        ------
        ValueError
            If any required field is missing or invalid.
        """
        if not self.uuid:
            raise ValueError("Instance UUID is required")

        if not self.key:
            raise ValueError("Instance key is required")

        if not self.identifier:
            raise ValueError("Instance identifier is required")


@dataclass
class InstanceDetails:
    """Detailed instance information

    Parameters
    ----------
    uuid : str
        Unique identifier for the instance.
    identifier : str
        Identifier for the instance.
    status : InstanceStatus
        Current status of the instance.
    cpu_cores : int
        Number of CPU cores allocated to the instance.
    gpu_type : str
        Type of GPU attached to the instance.
    num_gpus : int
        Number of GPUs attached to the instance.
    disk_size_gb : int
        Disk size in gigabytes.
    template : str
        Instance template used for creation.
    created_at : str, optional
        Timestamp when the instance was created, by default None.
    ip_address : str, optional
        Public IP address of the instance, by default None.
    thunder_cli_index : str, optional
        Index used by Thunder CLI to reference this instanceby default None.
    """

    uuid: str
    identifier: str
    status: InstanceStatus
    cpu_cores: int
    gpu_type: str
    num_gpus: int
    disk_size_gb: int
    template: str
    created_at: Optional[str] = None
    ip_address: Optional[str] = None
    thunder_cli_index: Optional[str] = None

    @property
    def is_ready_for_ssh(self) -> bool:
        """Check if instance is ready for SSH connections.

        Returns
        -------
        bool
            True if instance can accept SSH connections.
        """
        return self.status.is_ready_for_ssh

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
            Display name combining identifier and UUID prefix.
        """
        return f"{self.identifier} ({self.uuid[:8]})"

    @property
    def resource_summary(self) -> str:
        """Get a summary of instance resources.

        Returns
        -------
        str
            Summary of instance resources.
        """
        gpu_str = f"{self.num_gpus}x {self.gpu_type.upper()}" if self.num_gpus > 1 else self.gpu_type.upper()
        return f"{self.cpu_cores} CPU, {gpu_str} GPU, {self.disk_size_gb}GB disk"

    def validate(self) -> None:
        """Validate the instance details.

        Raises
        ------
        ValueError
            If any required field is missing or invalid.
        """
        if not self.uuid:
            raise ValueError("Instance UUID is required")

        if not self.identifier:
            raise ValueError("Instance identifier is required")

        if self.cpu_cores <= 0:
            raise ValueError("CPU cores must be positive")

        if self.num_gpus <= 0:
            raise ValueError("Number of GPUs must be positive")

        if self.disk_size_gb <= 0:
            raise ValueError("Disk size must be positive")
