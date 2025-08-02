import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from deepracer_research.deployment.nvidia_brev.config.nvidia_brev_config import NvidiaBrevConfig
from deepracer_research.deployment.nvidia_brev.models import (
    ApiResponse,
    CreateInstanceRequest,
    ListInstancesRequest,
    NvidiaBrevError,
    UpdateInstanceRequest,
)
from deepracer_research.deployment.nvidia_brev.models.instance_models import (
    InstanceDetails,
    InstanceMetrics,
    InstanceResponse,
    InstanceStatus,
)


class NvidiaBrevClient:
    """NVIDIA Brev API client for managing GPU instances

    Parameters
    ----------
    config : NvidiaBrevConfig
        Configuration containing API credentials and settings
    """

    def __init__(self, config: NvidiaBrevConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(self.config.headers)
        self.session.verify = config.verify_ssl

        self.config.validate()

    def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None
    ) -> ApiResponse:
        """Make HTTP request to NVIDIA Brev API.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE)
        endpoint : str
            API endpoint path
        data : Optional[Dict[str, Any]], optional
            Request body data, by default None
        params : Optional[Dict[str, Any]], optional
            Query parameters, by default None

        Returns
        -------
        ApiResponse
            API response wrapper

        Raises
        ------
        NvidiaBrevError
            If the API request fails
        """
        url = urljoin(self.config.api_url, endpoint.lstrip("/"))

        try:
            response = self.session.request(method=method, url=url, json=data, params=params, timeout=self.config.timeout)

            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"message": response.text}

            api_response = ApiResponse(
                success=response.ok,
                data=response_data.get("data") if response.ok else None,
                message=response_data.get("message", ""),
                error_code=response_data.get("error_code") if not response.ok else None,
                status_code=response.status_code,
                request_id=response.headers.get("X-Request-ID"),
                metadata={"url": url, "method": method, "elapsed": response.elapsed.total_seconds()},
            )

            if not response.ok:
                api_response.raise_for_status()

            return api_response

        except requests.exceptions.RequestException as e:
            raise NvidiaBrevError(
                message=f"API request failed: {str(e)}",
                status_code=getattr(e.response, "status_code", None) if hasattr(e, "response") else None,
                details={"url": url, "method": method},
            )

    def create_instance(self, request: CreateInstanceRequest) -> InstanceResponse:
        """Create a new NVIDIA Brev instance.

        Parameters
        ----------
        request : CreateInstanceRequest
            Instance creation request

        Returns
        -------
        InstanceResponse
            Response containing instance details
        """
        request.validate()

        response = self._make_request(method="POST", endpoint="/instances", data=request.to_dict())

        if response.data:
            instance = self._parse_instance_data(response.data)
            return InstanceResponse(success=True, instance=instance, message=response.message, request_id=response.request_id)
        else:
            return InstanceResponse(
                success=False, message=response.message or "Failed to create instance", request_id=response.request_id
            )

    def get_instance(self, instance_id: str) -> InstanceResponse:
        """Get details of a specific instance.

        Parameters
        ----------
        instance_id : str
            Instance ID

        Returns
        -------
        InstanceResponse
            Response containing instance details
        """
        response = self._make_request(method="GET", endpoint=f"/instances/{instance_id}")

        if response.data:
            instance = self._parse_instance_data(response.data)
            return InstanceResponse(success=True, instance=instance, message=response.message, request_id=response.request_id)
        else:
            return InstanceResponse(
                success=False, message=response.message or "Instance not found", request_id=response.request_id
            )

    def list_instances(self, request: Optional[ListInstancesRequest] = None) -> List[InstanceDetails]:
        """List all instances with optional filtering.

        Parameters
        ----------
        request : Optional[ListInstancesRequest], optional
            List request with filters, by default None

        Returns
        -------
        List[InstanceDetails]
            List of instance details
        """
        params = request.to_dict() if request else {}

        response = self._make_request(method="GET", endpoint="/instances", params=params)

        instances = []
        if response.data and "instances" in response.data:
            for instance_data in response.data["instances"]:
                instance = self._parse_instance_data(instance_data)
                instances.append(instance)

        return instances

    def update_instance(self, request: UpdateInstanceRequest) -> InstanceResponse:
        """Update an existing instance.

        Parameters
        ----------
        request : UpdateInstanceRequest
            Instance update request

        Returns
        -------
        InstanceResponse
            Response containing updated instance details
        """
        request.validate()

        response = self._make_request(
            method="PUT",
            endpoint=f"/instances/{request.instance_id}",
            data={k: v for k, v in request.to_dict().items() if k != "instance_id"},
        )

        if response.data:
            instance = self._parse_instance_data(response.data)
            return InstanceResponse(success=True, instance=instance, message=response.message, request_id=response.request_id)
        else:
            return InstanceResponse(
                success=False, message=response.message or "Failed to update instance", request_id=response.request_id
            )

    def start_instance(self, instance_id: str) -> InstanceResponse:
        """Start a stopped instance.

        Parameters
        ----------
        instance_id : str
            Instance ID

        Returns
        -------
        InstanceResponse
            Response containing instance details
        """
        response = self._make_request(method="POST", endpoint=f"/instances/{instance_id}/start")

        if response.data:
            instance = self._parse_instance_data(response.data)
            return InstanceResponse(
                success=True, instance=instance, message="Instance start initiated", request_id=response.request_id
            )
        else:
            return InstanceResponse(
                success=False, message=response.message or "Failed to start instance", request_id=response.request_id
            )

    def stop_instance(self, instance_id: str) -> InstanceResponse:
        """Stop a running instance.

        Parameters
        ----------
        instance_id : str
            Instance ID

        Returns
        -------
        InstanceResponse
            Response containing instance details
        """
        response = self._make_request(method="POST", endpoint=f"/instances/{instance_id}/stop")

        if response.data:
            instance = self._parse_instance_data(response.data)
            return InstanceResponse(
                success=True, instance=instance, message="Instance stop initiated", request_id=response.request_id
            )
        else:
            return InstanceResponse(
                success=False, message=response.message or "Failed to stop instance", request_id=response.request_id
            )

    def delete_instance(self, instance_id: str) -> InstanceResponse:
        """Delete an instance.

        Parameters
        ----------
        instance_id : str
            Instance ID

        Returns
        -------
        InstanceResponse
            Response confirming deletion
        """
        response = self._make_request(method="DELETE", endpoint=f"/instances/{instance_id}")

        return InstanceResponse(
            success=response.success,
            message=response.message or ("Instance deletion initiated" if response.success else "Failed to delete instance"),
            request_id=response.request_id,
        )

    def get_instance_metrics(self, instance_id: str) -> Optional[InstanceMetrics]:
        """Get metrics for a running instance.

        Parameters
        ----------
        instance_id : str
            Instance ID

        Returns
        -------
        Optional[InstanceMetrics]
            Instance metrics if available
        """
        try:
            response = self._make_request(method="GET", endpoint=f"/instances/{instance_id}/metrics")

            if response.data:
                return InstanceMetrics(**response.data)

        except NvidiaBrevError:
            pass

        return None

    def wait_for_instance_status(
        self, instance_id: str, target_status: InstanceStatus, timeout_seconds: int = 300, poll_interval: int = 10
    ) -> InstanceDetails:
        """Wait for an instance to reach a specific status.

        Parameters
        ----------
        instance_id : str
            Instance ID
        target_status : InstanceStatus
            Target status to wait for
        timeout_seconds : int, optional
            Maximum time to wait in seconds, by default 300
        poll_interval : int, optional
            Polling interval in seconds, by default 10

        Returns
        -------
        InstanceDetails
            Final instance details

        Raises
        ------
        NvidiaBrevError
            If timeout is reached or instance enters error state
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            response = self.get_instance(instance_id)

            if not response.success or not response.instance:
                raise NvidiaBrevError(f"Failed to get instance status: {response.message}")

            instance = response.instance

            if instance.status == target_status:
                return instance

            if instance.status.is_terminal and instance.status != target_status:
                raise NvidiaBrevError(f"Instance entered terminal state: {instance.status}")

            time.sleep(poll_interval)

        raise NvidiaBrevError(f"Timeout waiting for instance to reach status: {target_status}")

    def _parse_instance_data(self, data: Dict[str, Any]) -> InstanceDetails:
        """Parse API response data into InstanceDetails.

        Parameters
        ----------
        data : Dict[str, Any]
            Instance data from API response

        Returns
        -------
        InstanceDetails
            Parsed instance details
        """
        from datetime import datetime

        from deepracer_research.deployment.nvidia_brev.enum.deployment_mode import DeploymentMode
        from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType
        from deepracer_research.deployment.nvidia_brev.enum.instance_template import InstanceTemplate

        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))

        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))

        stopped_at = None
        if data.get("stopped_at"):
            stopped_at = datetime.fromisoformat(data["stopped_at"].replace("Z", "+00:00"))

        auto_shutdown_at = None
        if data.get("auto_shutdown_at"):
            auto_shutdown_at = datetime.fromisoformat(data["auto_shutdown_at"].replace("Z", "+00:00"))

        metrics = None
        if data.get("metrics"):
            metrics = InstanceMetrics(**data["metrics"])

        return InstanceDetails(
            instance_id=data["instance_id"],
            name=data["name"],
            status=InstanceStatus(data["status"]),
            gpu_type=GPUType(data["gpu_type"]),
            template=InstanceTemplate(data["template"]),
            deployment_mode=DeploymentMode(data["deployment_mode"]),
            num_gpus=data["num_gpus"],
            cpu_cores=data["cpu_cores"],
            memory_gb=data["memory_gb"],
            disk_size_gb=data["disk_size_gb"],
            public_ip=data.get("public_ip"),
            private_ip=data.get("private_ip"),
            ssh_host=data.get("ssh_host"),
            ssh_port=data.get("ssh_port", 22),
            region=data.get("region"),
            created_at=created_at,
            started_at=started_at,
            stopped_at=stopped_at,
            ports=data.get("ports", []),
            environment_variables=data.get("environment_variables", {}),
            tags=data.get("tags", {}),
            metrics=metrics,
            auto_shutdown_at=auto_shutdown_at,
        )

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get available instance templates.

        Returns
        -------
        List[Dict[str, Any]]
            List of available templates
        """
        response = self._make_request(method="GET", endpoint="/templates")

        return response.data.get("templates", []) if response.data else []

    def get_available_gpu_types(self) -> List[Dict[str, Any]]:
        """Get available GPU types.

        Returns
        -------
        List[Dict[str, Any]]
            List of available GPU types
        """
        response = self._make_request(method="GET", endpoint="/gpu-types")

        return response.data.get("gpu_types", []) if response.data else []

    def get_pricing_info(self) -> Dict[str, Any]:
        """Get current pricing information.

        Returns
        -------
        Dict[str, Any]
            Pricing information
        """
        response = self._make_request(method="GET", endpoint="/pricing")

        return response.data if response.data else {}

    def close(self):
        """Close the HTTP session."""
        self.session.close()
