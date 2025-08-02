from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class UpdateInstanceRequest:
    """Request to update an existing NVIDIA Brev instance

    Parameters
    ----------
    instance_id : str
        Instance ID to update
    name : str, optional
        New instance name, by default None
    ports : List[int], optional
        New ports to expose, by default None
    environment_variables : Dict[str, str], optional
        New environment variables, by default None
    auto_shutdown_hours : int, optional
        New auto shutdown hours, by default None
    tags : Dict[str, str], optional
        New tags, by default None
    """

    instance_id: str
    name: Optional[str] = None
    ports: Optional[List[int]] = None
    environment_variables: Optional[Dict[str, str]] = None
    auto_shutdown_hours: Optional[int] = None
    tags: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for API call.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for API
        """
        data = {"instance_id": self.instance_id}

        if self.name is not None:
            data["name"] = self.name

        if self.ports is not None:
            data["ports"] = self.ports

        if self.environment_variables is not None:
            data["environment_variables"] = self.environment_variables

        if self.auto_shutdown_hours is not None:
            data["auto_shutdown_hours"] = self.auto_shutdown_hours

        if self.tags is not None:
            data["tags"] = self.tags

        return data

    def validate(self) -> None:
        """Validate the request parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid
        """
        if not self.instance_id or not self.instance_id.strip():
            raise ValueError("Instance ID is required")

        if self.name is not None and not self.name.strip():
            raise ValueError("Instance name cannot be empty")

        if self.auto_shutdown_hours is not None and self.auto_shutdown_hours <= 0:
            raise ValueError("Auto shutdown hours must be positive")

        if self.ports is not None:
            for port in self.ports:
                if not (1 <= port <= 65535):
                    raise ValueError(f"Invalid port number: {port}")
