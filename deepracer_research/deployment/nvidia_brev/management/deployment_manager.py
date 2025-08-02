import time
from typing import Any, Dict, Optional

from deepracer_research.deployment.nvidia_brev.api.client import NvidiaBrevClient
from deepracer_research.deployment.nvidia_brev.config.deepracer_config import NvidiaBrevDeepRacerConfig
from deepracer_research.deployment.nvidia_brev.installation.deepracer_installer import DeepRacerInstaller
from deepracer_research.deployment.nvidia_brev.management.instance_manager import InstanceManager
from deepracer_research.deployment.nvidia_brev.management.training_manager import TrainingManager
from deepracer_research.deployment.nvidia_brev.models import CreateInstanceRequest
from deepracer_research.deployment.nvidia_brev.models.instance_models import InstanceStatus
from deepracer_research.deployment.nvidia_brev.ssh.ssh_manager import NvidiaBrevSSHManager


class NvidiaBrevDeploymentManager:
    """Main deployment manager for NVIDIA Brev DeepRacer deployments

    Parameters
    ----------
    config : NvidiaBrevDeepRacerConfig
        Complete deployment configuration
    """

    def __init__(self, config: NvidiaBrevDeepRacerConfig):
        self.config = config
        self.client = NvidiaBrevClient(config.brev_config)
        self.instance_manager = None
        self.training_manager = None
        self.current_instance = None

    def deploy(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Deploy complete DeepRacer training environment.

        Parameters
        ----------
        progress_callback : Optional[callable], optional
            Callback for progress updates, by default None

        Returns
        -------
        Dict[str, Any]
            Deployment results
        """
        deployment_start = time.time()
        results = {}

        try:
            if progress_callback:
                progress_callback("Starting NVIDIA Brev deployment...")

            if progress_callback:
                progress_callback("Creating GPU instance...")

            instance_result = self._create_instance()
            results["instance_creation"] = instance_result

            if not instance_result["success"]:
                return self._create_error_result("Instance creation failed", results)

            self.current_instance = instance_result["instance"]

            if progress_callback:
                progress_callback("Waiting for instance to be ready...")

            ready_result = self._wait_for_instance_ready()
            results["instance_ready"] = ready_result

            if not ready_result["success"]:
                return self._create_error_result("Instance failed to become ready", results)

            if progress_callback:
                progress_callback("Establishing SSH connection...")

            ssh_result = self._setup_ssh_connection()
            results["ssh_setup"] = ssh_result

            if not ssh_result["success"]:
                return self._create_error_result("SSH setup failed", results)

            if progress_callback:
                progress_callback("Installing DeepRacer-for-Cloud...")

            install_result = self._install_deepracer(progress_callback)
            results["deepracer_installation"] = install_result

            if not install_result["success"]:
                return self._create_error_result("DeepRacer installation failed", results)

            if progress_callback:
                progress_callback("Initializing training environment...")

            init_result = self._initialize_training()
            results["training_initialization"] = init_result

            if not init_result["success"]:
                return self._create_error_result("Training initialization failed", results)

            if self.config.auto_start_training:
                if progress_callback:
                    progress_callback("Starting training...")

                training_result = self._start_training()
                results["training_start"] = training_result

            deployment_duration = time.time() - deployment_start

            if progress_callback:
                progress_callback("Deployment completed successfully!")

            return {
                "success": True,
                "message": "DeepRacer deployment completed successfully",
                "duration": deployment_duration,
                "instance": self.current_instance.to_dict() if self.current_instance else None,
                "access_info": self._get_access_info(),
                "results": results,
            }

        except Exception as e:
            deployment_duration = time.time() - deployment_start
            return {
                "success": False,
                "message": f"Deployment failed: {str(e)}",
                "duration": deployment_duration,
                "instance": self.current_instance.to_dict() if self.current_instance else None,
                "results": results,
                "error": str(e),
            }

    def _create_instance(self) -> Dict[str, Any]:
        """Create NVIDIA Brev instance."""
        try:
            request = CreateInstanceRequest(
                name=self.config.get_deployment_name(),
                template=self.config.instance_config.template,
                gpu_type=self.config.instance_config.gpu_type,
                deployment_mode=self.config.instance_config.deployment_mode,
                num_gpus=self.config.instance_config.num_gpus,
                cpu_cores=self.config.instance_config.cpu_cores,
                memory_gb=self.config.instance_config.memory_gb,
                disk_size_gb=self.config.instance_config.disk_size_gb,
                region=self.config.instance_config.region,
                ports=self.config.instance_config.ports,
                environment_variables=self.config.get_environment_variables(),
                setup_script=self.config.get_deepracer_setup_script(),
                auto_shutdown_hours=self.config.instance_config.auto_shutdown_hours,
                tags=self.config.tags,
            )

            response = self.client.create_instance(request)

            if response.success and response.instance:
                return {"success": True, "instance": response.instance, "message": response.message}
            else:
                return {"success": False, "message": response.message or "Failed to create instance"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _wait_for_instance_ready(self, timeout: int = 600) -> Dict[str, Any]:
        """Wait for instance to be ready for SSH connections."""
        try:
            if not self.current_instance:
                return {"success": False, "error": "No instance available"}

            ready_instance = self.client.wait_for_instance_status(
                self.current_instance.instance_id, InstanceStatus.RUNNING, timeout_seconds=timeout
            )

            self.current_instance = ready_instance

            return {"success": True, "instance": ready_instance, "message": "Instance is ready"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _setup_ssh_connection(self) -> Dict[str, Any]:
        """Setup SSH connection to the instance."""
        try:
            if not self.current_instance:
                return {"success": False, "error": "No instance available"}

            ssh_manager = NvidiaBrevSSHManager(self.config.ssh_config, self.current_instance)

            ssh_manager.connect()

            system_info = ssh_manager.get_system_info()

            self.instance_manager = InstanceManager(self.client, ssh_manager)
            self.training_manager = TrainingManager(ssh_manager, self.config)

            return {
                "success": True,
                "system_info": system_info,
                "connection_host": self.current_instance.connection_host,
                "ssh_port": self.current_instance.ssh_port,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _install_deepracer(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Install DeepRacer on the instance."""
        try:
            if not self.instance_manager:
                return {"success": False, "error": "Instance manager not initialized"}

            installer = DeepRacerInstaller(self.instance_manager.ssh_manager, self.config)

            install_result = installer.install(progress_callback)

            return install_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _initialize_training(self) -> Dict[str, Any]:
        """Initialize the training environment."""
        try:
            if not self.training_manager:
                return {"success": False, "error": "Training manager not initialized"}

            init_result = self.training_manager.initialize()

            return init_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _start_training(self) -> Dict[str, Any]:
        """Start training if auto-start is enabled."""
        try:
            if not self.training_manager:
                return {"success": False, "error": "Training manager not initialized"}

            training_result = self.training_manager.start_training()

            return training_result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_access_info(self) -> Dict[str, str]:
        """Get access information for the deployment."""
        if not self.current_instance:
            return {}

        host = self.current_instance.connection_host
        ports = self.current_instance.ports

        access_info = {
            "ssh_connection": f"ssh {self.config.ssh_config.username}@{host}",
            "instance_id": self.current_instance.instance_id,
            "gpu_type": self.current_instance.gpu_type.display_name,
            "status": self.current_instance.status.value,
        }

        if 8888 in ports:
            access_info["jupyter"] = f"http://{host}:8888"
        if 6006 in ports:
            access_info["tensorboard"] = f"http://{host}:6006"
        if 8080 in ports:
            access_info["deepracer_ui"] = f"http://{host}:8080"

        return access_info

    def _create_error_result(self, message: str, partial_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create an error result dictionary."""
        return {
            "success": False,
            "message": message,
            "partial_results": partial_results,
            "instance": self.current_instance.to_dict() if self.current_instance else None,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        if not self.current_instance:
            return {"status": "no_instance", "message": "No instance deployed"}

        try:
            response = self.client.get_instance(self.current_instance.instance_id)

            if response.success and response.instance:
                self.current_instance = response.instance

                status_info = {
                    "status": "deployed",
                    "instance": self.current_instance.to_dict(),
                    "access_info": self._get_access_info(),
                }

                if self.training_manager:
                    training_status = self.training_manager.get_status()
                    status_info["training"] = training_status

                return status_info
            else:
                return {"status": "error", "message": response.message or "Failed to get instance status"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def stop_training(self) -> Dict[str, Any]:
        """Stop training."""
        if not self.training_manager:
            return {"success": False, "error": "Training manager not initialized"}

        return self.training_manager.stop_training()

    def destroy(self, force: bool = False) -> Dict[str, Any]:
        """Destroy the deployment."""
        try:
            results = {}

            if self.training_manager:
                stop_result = self.training_manager.stop_training()
                results["training_stop"] = stop_result

            if self.current_instance:
                delete_response = self.client.delete_instance(self.current_instance.instance_id)
                results["instance_deletion"] = {"success": delete_response.success, "message": delete_response.message}

            self.instance_manager = None
            self.training_manager = None
            self.current_instance = None

            return {"success": True, "message": "Deployment destroyed successfully", "results": results}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_instance:
            self.destroy()
