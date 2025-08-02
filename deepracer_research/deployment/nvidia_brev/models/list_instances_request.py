from dataclasses import dataclass
from typing import Any, Dict, Optional

from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType
from deepracer_research.deployment.nvidia_brev.enum.instance_template import InstanceTemplate


@dataclass
class ListInstancesRequest:
    """Request to list NVIDIA Brev instances

    Parameters
    ----------
    status : str, optional
        Filter by instance status, by default None
    gpu_type : GPUType, optional
        Filter by GPU type, by default None
    template : InstanceTemplate, optional
        Filter by template, by default None
    tags : Dict[str, str], optional
        Filter by tags, by default None
    limit : int, optional
        Maximum number of instances to return, by default 50
    offset : int, optional
        Number of instances to skip, by default 0
    """

    status: Optional[str] = None
    gpu_type: Optional[GPUType] = None
    template: Optional[InstanceTemplate] = None
    tags: Optional[Dict[str, str]] = None
    limit: int = 50
    offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for API call.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for API
        """
        data = {"limit": self.limit, "offset": self.offset}

        if self.status:
            data["status"] = self.status

        if self.gpu_type:
            data["gpu_type"] = self.gpu_type.value

        if self.template:
            data["template"] = self.template.value

        if self.tags:
            data["tags"] = self.tags

        return data
