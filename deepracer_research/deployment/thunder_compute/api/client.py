from typing import List

from deepracer_research.deployment.thunder_compute.api.http_client import HTTPClient
from deepracer_research.deployment.thunder_compute.models.api_models import ThunderComputeError
from deepracer_research.deployment.thunder_compute.models.instance_models import (
    InstanceDetails,
    InstanceResponse,
    InstanceStatus,
)
from deepracer_research.utils.logger import debug, error, info, warning


class ThunderComputeClient:
    """Thunder Compute API client for instance management

    Parameters
    ----------
    api_token : str
        Bearer token for Thunder Compute API authentication.
    base_url : str, optional
        Base URL for Thunder Compute API, by default "https://api.thundercompute.com:8443".
    """

    def __init__(self, api_token: str, base_url: str = "https://api.thundercompute.com:8443"):
        """Initialize Thunder Compute client.

        Parameters
        ----------
        api_token : str
            Bearer token for authentication.
        base_url : str, optional
            Base URL for Thunder Compute API, by default "https://api.thundercompute.com:8443".
        """
        self.api_token = api_token
        self.base_url = base_url
        self.http_client = HTTPClient(api_token, base_url)

        info("Thunder Compute client initialized", extra={"base_url": self.base_url})

    def create_instance(
        self, cpu_cores: int, template: str, gpu_type: str, num_gpus: int = 1, disk_size_gb: int = 100, mode: str = "production"
    ) -> InstanceResponse:
        """Create a new Thunder Compute instance.

        Parameters
        ----------
        cpu_cores : int
            Number of CPU cores to allocate.
        template : str
            Instance template (e.g., 'base', 'pytorch').
        gpu_type : str
            GPU type (e.g., 't4', 'v100').
        num_gpus : int, optional
            Number of GPUs to attach, by default 1.
        disk_size_gb : int, optional
            Disk size in gigabytes, by default 100.
        mode : str, optional
            Deployment mode ('prototyping' or 'production'), by default "production".

        Returns
        -------
        InstanceResponse
            Response containing instance UUID, key, and identifier.

        Raises
        ------
        ThunderComputeError
            If instance creation fails.
        """
        endpoint = "/instances/create"

        payload = {
            "cpu_cores": cpu_cores,
            "template": template,
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "disk_size_gb": disk_size_gb,
            "mode": mode,
        }

        info(
            "Creating Thunder Compute instance",
            extra={
                "cpu_cores": cpu_cores,
                "template": template,
                "gpu_type": gpu_type,
                "num_gpus": num_gpus,
                "disk_size_gb": disk_size_gb,
                "mode": mode,
            },
        )

        try:
            response_data = self.http_client.post(endpoint, json_data=payload)

            uuid = response_data.get("uuid") or response_data.get("id") or response_data.get("instance_id")
            key = response_data.get("key") or response_data.get("ssh_key") or "unknown"
            identifier = (
                response_data.get("identifier") or response_data.get("name") or response_data.get("instance_name") or "unknown"
            )

            if not uuid:
                error("No UUID found in response", extra={"response_data": response_data})
                raise ThunderComputeError("No instance UUID found in API response")

            instance_response = InstanceResponse(uuid=uuid, key=key, identifier=identifier)

            instance_response.validate()

            info(
                "Instance created successfully",
                extra={"instance_uuid": instance_response.uuid, "instance_identifier": instance_response.identifier},
            )

            return instance_response

        except KeyError as e:
            error(
                "Invalid response format from instance creation", extra={"missing_key": str(e), "response_data": response_data}
            )
            raise ThunderComputeError(f"Invalid response format: missing {e}")
        except Exception as e:
            error(
                "Instance creation failed",
                extra={
                    "error": str(e),
                    "payload": payload,
                    "response_data": response_data if "response_data" in locals() else "not_available",
                },
            )
            raise ThunderComputeError(f"Instance creation failed: {e}")

    def list_instances(self) -> List[InstanceDetails]:
        """List all instances for the user.

        Returns
        -------
        List[InstanceDetails]
            List of instance details for all user instances.

        Raises
        ------
        ThunderComputeError
            If listing instances fails.
        """
        endpoint = "/instances/list"

        debug("Listing all instances")

        try:
            response_data = self.http_client.get(endpoint)

            debug("Raw list response", extra={"response_data": response_data})

            instances = []
            for key, instance_data in response_data.items():
                if not isinstance(instance_data, dict):
                    continue

                try:
                    instance = InstanceDetails(
                        uuid=instance_data["uuid"],
                        identifier=instance_data.get("name", "unknown"),
                        status=InstanceStatus(instance_data["status"].lower()),
                        cpu_cores=int(instance_data.get("cpuCores", 0)),
                        gpu_type=instance_data.get("gpuType", "unknown"),
                        num_gpus=int(instance_data.get("numGpus", 1)),
                        disk_size_gb=int(instance_data.get("storage", 100)),
                        template=instance_data.get("template", "unknown"),
                        created_at=instance_data.get("createdAt"),
                        ip_address=instance_data.get("ip"),
                        thunder_cli_index=key,
                    )

                    instance.validate()
                    instances.append(instance)

                    debug(
                        "Successfully parsed instance",
                        extra={"uuid": instance.uuid, "status": instance.status.value, "identifier": instance.identifier},
                    )

                except (KeyError, ValueError) as e:
                    warning(
                        "Skipping invalid instance data", extra={"error": str(e), "instance_data": instance_data, "key": key}
                    )
                    continue

            info(f"Listed {len(instances)} instances")
            return instances

        except Exception as e:
            error("Failed to list instances", extra={"error": str(e)})
            raise

    def get_instance_by_id(self, instance_id: str) -> InstanceDetails:
        """Get instance by Thunder CLI index (instance ID).

        Parameters
        ----------
        instance_id : str
            Thunder CLI index of the instance (e.g., "0", "1", etc.).

        Returns
        -------
        InstanceDetails
            Detailed information about the instance.

        Raises
        ------
        ThunderComputeError
            If instance with the given ID is not found.
        """
        debug("Getting instance by ID", extra={"instance_id": instance_id})

        try:
            all_instances = self.list_instances()

            for instance in all_instances:
                if instance.thunder_cli_index == instance_id:
                    debug(
                        "Found instance by ID",
                        extra={"instance_id": instance_id, "uuid": instance.uuid, "status": instance.status.value},
                    )
                    return instance

            error(
                "Instance not found by ID",
                extra={
                    "instance_id": instance_id,
                    "available_instances": [(inst.thunder_cli_index, inst.uuid) for inst in all_instances],
                },
            )
            raise ThunderComputeError(f"Instance with ID '{instance_id}' not found")

        except Exception as e:
            if isinstance(e, ThunderComputeError):
                raise
            error("Failed to get instance by ID", extra={"instance_id": instance_id, "error": str(e)})
            raise ThunderComputeError(f"Failed to get instance by ID: {e}")

    def get_instance(self, instance_uuid: str, wait_for_registration: bool = False, timeout: int = 60) -> InstanceDetails:
        """Get details for a specific instance.

        Uses the list instances endpoint and filters for the specific UUID
        since Thunder Compute doesn't provide individual instance endpoints.

        Parameters
        ----------
        instance_uuid : str
            UUID of the instance to retrieve.
        wait_for_registration : bool, optional
            Whether to wait for instance to appear in list, by default False.
        timeout : int, optional
            Maximum time to wait for registration in seconds, by default 60.

        Returns
        -------
        InstanceDetails
            Detailed information about the instance.

        Raises
        ------
        ThunderComputeError
            If instance not found or request fails.
        """
        import time

        debug(
            "Getting instance details via list endpoint",
            extra={"instance_uuid": instance_uuid, "wait_for_registration": wait_for_registration},
        )

        start_time = time.time()
        attempt = 0

        while True:
            attempt += 1
            try:
                all_instances = self.list_instances()

                for instance in all_instances:
                    if instance.uuid == instance_uuid:
                        debug(
                            "Found instance in list",
                            extra={
                                "instance_uuid": instance_uuid,
                                "status": instance.status.value,
                                "resource_summary": instance.resource_summary,
                                "attempt": attempt,
                            },
                        )
                        return instance

                if not wait_for_registration:
                    error(
                        "Instance not found in user's instances",
                        extra={"instance_uuid": instance_uuid, "available_instances": [inst.uuid for inst in all_instances]},
                    )
                    raise ThunderComputeError(f"Instance {instance_uuid} not found")

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    error(
                        "Instance registration timeout",
                        extra={"instance_uuid": instance_uuid, "timeout": timeout, "elapsed": elapsed, "attempts": attempt},
                    )
                    raise ThunderComputeError(
                        f"Instance {instance_uuid} not registered within {timeout} seconds. Check Thunder Compute console - instance may still be starting."
                    )

                debug(
                    "Instance not registered yet, waiting...",
                    extra={"instance_uuid": instance_uuid, "elapsed": elapsed, "attempt": attempt},
                )
                time.sleep(2)

            except ThunderComputeError:
                raise
            except Exception as e:
                error("Failed to get instance details", extra={"instance_uuid": instance_uuid, "error": str(e)})
                raise ThunderComputeError(f"Failed to get instance: {e}")

    def delete_instance(self, instance_uuid: str) -> bool:
        """Delete an instance.

        Uses the official Thunder Compute API endpoint for instance deletion.

        Parameters
        ----------
        instance_uuid : str
            UUID of the instance to delete.

        Returns
        -------
        bool
            True if deletion was successful.

        Raises
        ------
        ThunderComputeError
            If deletion fails.
        """
        endpoint = f"/instances/{instance_uuid}/delete"

        info("Deleting instance", extra={"instance_uuid": instance_uuid})

        try:
            response_data = self.http_client.post(endpoint)

            info("Instance deleted successfully", extra={"instance_uuid": instance_uuid, "response": response_data})
            return True

        except ThunderComputeError as e:
            if e.status_code == 404:
                info("Instance not found (may have been already deleted)", extra={"instance_uuid": instance_uuid})
                return True
            else:
                error(
                    "Failed to delete instance",
                    extra={"instance_uuid": instance_uuid, "status_code": e.status_code, "error": str(e)},
                )
                raise
        except Exception as e:
            error("Failed to delete instance", extra={"instance_uuid": instance_uuid, "error": str(e)})
            raise ThunderComputeError(f"Instance deletion failed: {e}")

    def start_instance(self, instance_uuid: str) -> bool:
        """Start a stopped instance.

        Uses the official Thunder Compute API endpoint for starting instances.

        Parameters
        ----------
        instance_uuid : str
            UUID of the instance to start.

        Returns
        -------
        bool
            True if start was successful.

        Raises
        ------
        ThunderComputeError
            If start fails.
        """
        endpoint = f"/instances/{instance_uuid}/up"

        info("Starting instance", extra={"instance_uuid": instance_uuid})

        try:
            response_data = self.http_client.post(endpoint)

            info("Instance started successfully", extra={"instance_uuid": instance_uuid, "response": response_data})
            return True

        except Exception as e:
            error("Failed to start instance", extra={"instance_uuid": instance_uuid, "error": str(e)})
            raise ThunderComputeError(f"Instance start failed: {e}")

    def stop_instance(self, instance_uuid: str) -> bool:
        """Stop a running instance.

        Uses the official Thunder Compute API endpoint for stopping instances.

        Parameters
        ----------
        instance_uuid : str
            UUID of the instance to stop.

        Returns
        -------
        bool
            True if stop was successful.

        Raises
        ------
        ThunderComputeError
            If stop fails.
        """
        endpoint = f"/instances/{instance_uuid}/down"

        info("Stopping instance", extra={"instance_uuid": instance_uuid})

        try:
            response_data = self.http_client.post(endpoint)

            info("Instance stopped successfully", extra={"instance_uuid": instance_uuid, "response": response_data})
            return True

        except Exception as e:
            error("Failed to stop instance", extra={"instance_uuid": instance_uuid, "error": str(e)})
            raise ThunderComputeError(f"Instance stop failed: {e}")

    def close(self) -> None:
        """Close the client and clean up resources."""
        self.http_client.close()
        debug("Thunder Compute client closed")

    def __enter__(self) -> "ThunderComputeClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
