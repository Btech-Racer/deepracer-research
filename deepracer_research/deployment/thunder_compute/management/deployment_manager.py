import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepracer_research.deployment.thunder_compute.api.client import ThunderComputeClient
from deepracer_research.deployment.thunder_compute.config.deepracer_cloud_config import DeepRacerCloudConfig
from deepracer_research.deployment.thunder_compute.config.instance_config import InstanceConfig
from deepracer_research.deployment.thunder_compute.config.ssh_config import SSHConfig
from deepracer_research.deployment.thunder_compute.config.thunder_compute_config import ThunderComputeConfig
from deepracer_research.deployment.thunder_compute.enum.gpu_type import GPUType
from deepracer_research.deployment.thunder_compute.installation.deepracer_installer import DeepRacerCloudInstaller
from deepracer_research.deployment.thunder_compute.installation.errors import DeepRacerInstallationError
from deepracer_research.deployment.thunder_compute.management.deployment_result import DeploymentResult
from deepracer_research.deployment.thunder_compute.management.errors import ThunderDeploymentError
from deepracer_research.deployment.thunder_compute.models.api_models import ThunderComputeError
from deepracer_research.deployment.thunder_compute.models.instance_models import InstanceDetails
from deepracer_research.deployment.thunder_compute.ssh import SSHConnectionError, SSHManager
from deepracer_research.utils.logger import debug, error, info, warning


class ThunderDeploymentManager:
    """Manager for Thunder Compute deployments with DeepRacer integration"""

    def __init__(self, thunder_config: ThunderComputeConfig):
        """Initialize deployment manager.

        Parameters
        ----------
        thunder_config : ThunderComputeConfig
            Thunder Compute API configuration.
        """
        self.thunder_config = thunder_config
        self.client = ThunderComputeClient(thunder_config.api_token, thunder_config.base_url)
        self.active_deployments: Dict[str, Dict[str, Any]] = {}

    def deploy_deepracer_instance(
        self,
        instance_config: InstanceConfig,
        ssh_config: Optional[SSHConfig] = None,
        deepracer_config: Optional[DeepRacerCloudConfig] = None,
        wait_for_ready: bool = True,
    ) -> DeploymentResult:
        """Deploy a complete DeepRacer training instance.

        Parameters
        ----------
        instance_config : InstanceConfig
            Instance configuration.
        ssh_config : SSHConfig, optional
            SSH configuration, by default None (uses defaults).
        deepracer_config : DeepRacerCloudConfig, optional
            DeepRacer configuration, by default None (uses defaults).
        wait_for_ready : bool, optional
            Whether to wait for instance to be ready, by default True.

        Returns
        -------
        DeploymentResult
            Deployment result.

        Raises
        ------
        ThunderDeploymentError
            If deployment fails.
        """
        info("Starting DeepRacer instance deployment")

        instance_uuid = None
        ssh_manager = None

        try:
            info("Creating Thunder Compute instance")
            instance_response = self.client.create_instance(**instance_config.to_dict())
            instance_uuid = instance_response.uuid

            info("Instance created successfully", extra={"instance_uuid": instance_uuid})

            if wait_for_ready:
                instance_details = self._wait_for_instance_running(instance_uuid)
            else:
                instance_details = self.client.get_instance(instance_uuid, wait_for_registration=True, timeout=180)

            info("Setting up SSH connection")
            ssh_manager = SSHManager(instance_uuid, ssh_config, thunder_cli_index=instance_details.thunder_cli_index)

            if wait_for_ready:
                ssh_manager.setup_tnr_connection()
                ssh_manager.wait_for_instance_ready()

            deepracer_installed = False
            if instance_config.install_deepracer_cloud and wait_for_ready:
                info("Installing DeepRacer-for-Cloud")
                installer = DeepRacerCloudInstaller(ssh_manager, deepracer_config)
                installer.install(instance_config)
                deepracer_installed = True

                if instance_config.s3_bucket_name:
                    info("Setting up AWS environment")
                    from deepracer_research.deployment.thunder_compute.ssh.aws_setup import AWSSetupManager

                    aws_setup = AWSSetupManager(ssh_manager)
                    aws_setup.setup_deepracer_environment(instance_config.s3_bucket_name)

            self.active_deployments[instance_uuid] = {
                "instance_config": instance_config,
                "ssh_manager": ssh_manager,
                "installer": DeepRacerCloudInstaller(ssh_manager, deepracer_config) if deepracer_installed else None,
                "created_at": time.time(),
            }

            info("DeepRacer instance deployment completed", extra={"instance_uuid": instance_uuid})

            return DeploymentResult(
                success=True,
                instance_uuid=instance_uuid,
                instance_details=instance_details,
                ssh_ready=wait_for_ready,
                deepracer_installed=deepracer_installed,
            )

        except Exception as e:
            error("Deployment failed", extra={"error": str(e)})

            if instance_uuid:
                try:
                    info("Cleaning up failed deployment", extra={"instance_uuid": instance_uuid})
                    self.client.delete_instance(instance_uuid)
                except:
                    warning("Failed to cleanup instance", extra={"instance_uuid": instance_uuid})

            return DeploymentResult(
                success=False,
                instance_uuid=instance_uuid or "",
                instance_details=None,
                ssh_ready=False,
                deepracer_installed=False,
                error_message=str(e),
            )

    def _wait_for_instance_running(self, instance_uuid: str, timeout: int = 600, check_interval: int = 10) -> InstanceDetails:
        """Wait for instance to be in running state.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.
        timeout : int, optional
            Maximum time to wait in seconds, by default 600.
        check_interval : int, optional
            Time between checks in seconds, by default 10.

        Returns
        -------
        InstanceDetails
            Instance details when running.

        Raises
        ------
        ThunderDeploymentError
            If instance doesn't start within timeout.
        """
        info("Waiting for instance to be running", extra={"instance_uuid": instance_uuid, "timeout": timeout})

        info("Waiting for instance to be registered in API...", extra={"instance_uuid": instance_uuid})
        time.sleep(10)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                instance = self.client.get_instance(instance_uuid, wait_for_registration=True, timeout=180)

                if instance.status.value == "running":
                    info("Instance is now running", extra={"instance_uuid": instance_uuid})
                    return instance
                elif instance.status.value == "failed":
                    raise ThunderDeploymentError(f"Instance {instance_uuid} failed to start")

                debug("Instance status check", extra={"instance_uuid": instance_uuid, "status": instance.status.value})
                time.sleep(check_interval)

            except ThunderComputeError as e:
                if e.status_code == 404:
                    debug(
                        "Instance not found yet, continuing to wait",
                        extra={"instance_uuid": instance_uuid, "elapsed_time": time.time() - start_time},
                    )
                    time.sleep(check_interval)
                    continue
                else:
                    raise

        raise ThunderDeploymentError(
            f"Instance {instance_uuid} not ready within {timeout} seconds. Instance may still be starting or not properly registered."
        )

    def list_instances(self) -> List[InstanceDetails]:
        """List all Thunder Compute instances.

        Returns
        -------
        List[InstanceDetails]
            List of instance details.
        """
        return self.client.list_instances()

    def get_instance_details(self, instance_uuid: str) -> InstanceDetails:
        """Get details for a specific instance.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.

        Returns
        -------
        InstanceDetails
            Instance details.
        """
        return self.client.get_instance(instance_uuid)

    def delete_instance(self, instance_uuid: str) -> bool:
        """Delete an instance and cleanup local tracking.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID to delete.

        Returns
        -------
        bool
            True if deletion successful.
        """
        info("Deleting instance", extra={"instance_uuid": instance_uuid})

        try:
            result = self.client.delete_instance(instance_uuid)

            if instance_uuid in self.active_deployments:
                del self.active_deployments[instance_uuid]

            return result

        except Exception as e:
            error("Failed to delete instance", extra={"instance_uuid": instance_uuid, "error": str(e)})
            return False

    def get_ssh_manager(self, instance_uuid: str) -> Optional[SSHManager]:
        """Get SSH manager for an instance.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.

        Returns
        -------
        SSHManager, optional
            SSH manager if available.
        """
        deployment = self.active_deployments.get(instance_uuid)
        return deployment["ssh_manager"] if deployment else None

    def get_deepracer_installer(self, instance_uuid: str) -> Optional[DeepRacerCloudInstaller]:
        """Get DeepRacer installer for an instance.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.

        Returns
        -------
        DeepRacerCloudInstaller, optional
            DeepRacer installer if available.
        """
        deployment = self.active_deployments.get(instance_uuid)
        return deployment["installer"] if deployment else None

    def start_training(self, instance_uuid: str, training_name: str = "research-training") -> bool:
        """Start DeepRacer training on an instance.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.
        training_name : str, optional
            Name for the training session, by default "research-training".

        Returns
        -------
        bool
            True if training started successfully.

        Raises
        ------
        ThunderDeploymentError
            If training start fails.
        """
        installer = self.get_deepracer_installer(instance_uuid)
        if not installer:
            raise ThunderDeploymentError(f"No DeepRacer installer found for instance {instance_uuid}")

        try:
            return installer.start_training(training_name)
        except DeepRacerInstallationError as e:
            raise ThunderDeploymentError(f"Failed to start training: {e}")

    def stop_training(self, instance_uuid: str) -> bool:
        """Stop DeepRacer training on an instance.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.

        Returns
        -------
        bool
            True if training stopped successfully.
        """
        installer = self.get_deepracer_installer(instance_uuid)
        if not installer:
            warning("No DeepRacer installer found for instance", extra={"instance_uuid": instance_uuid})
            return False

        return installer.stop_training()

    def get_training_logs(self, instance_uuid: str) -> str:
        """Get training logs from an instance.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.

        Returns
        -------
        str
            Training logs as string.
        """
        installer = self.get_deepracer_installer(instance_uuid)
        if not installer:
            return f"No DeepRacer installer found for instance {instance_uuid}"

        return installer.get_training_logs()

    def execute_command(self, instance_uuid: str, command: str) -> str:
        """Execute a command on an instance.

        Parameters
        ----------
        instance_uuid : str
            Instance UUID.
        command : str
            Command to execute.

        Returns
        -------
        str
            Command output.

        Raises
        ------
        ThunderDeploymentError
            If command execution fails.
        """
        ssh_manager = self.get_ssh_manager(instance_uuid)
        if not ssh_manager:
            raise ThunderDeploymentError(f"No SSH manager found for instance {instance_uuid}")

        try:
            result = ssh_manager.execute_command(command)
            return result.stdout if result.success else result.stderr
        except SSHConnectionError as e:
            raise ThunderDeploymentError(f"SSH command failed: {e}")

    def deploy_deepracer_with_bootstrap(
        self,
        model_id: str,
        instance_config: Optional[InstanceConfig] = None,
        s3_bucket_name: Optional[str] = None,
        local_project_root: Optional[Path] = None,
        workers: int = 1,
        object_avoidance_config: Optional[dict] = None,
    ) -> str:
        """Deploy a complete DeepRacer training instance with bootstrap workflow.

        This method creates a new Thunder Compute instance, runs the bootstrap script,
        configures AWS credentials, sets up S3 bucket, uploads model files, and starts training.

        Parameters
        ----------
        model_id : str
            Unique model identifier for the training job
        instance_config : InstanceConfig, optional
            Instance configuration, uses default training config if None
        s3_bucket_name : str, optional
            S3 bucket name, generates one if None
        local_project_root : Path, optional
            Local project root directory, uses current directory if None

        Returns
        -------
        str
            Thunder Compute instance UUID

        Raises
        ------
        ThunderDeploymentError
            If deployment fails at any stage
        """
        info("Starting DeepRacer deployment with bootstrap workflow")
        info(f"Model ID: {model_id}")

        if local_project_root is None:
            local_project_root = Path.cwd()

        if instance_config is None:
            instance_config = InstanceConfig.for_deepracer_training(
                cpu_cores=8, gpu_type=GPUType.A100_XL, disk_size_gb=100, s3_bucket_name=s3_bucket_name
            )

        if s3_bucket_name is None:
            s3_bucket_name = f"deepracer-{model_id}-{int(time.time())}"

        instance_uuid = None
        ssh_manager = None

        try:
            info("Step 1: Creating Thunder Compute instance")
            instance_response = self.client.create_instance(**instance_config.to_dict())
            instance_uuid = instance_response.uuid
            info(f"Instance created: {instance_uuid}")

            info("Step 2: Waiting for instance to be ready")
            instance_details = self._wait_for_instance_running(instance_uuid)

            info("Step 3: Setting up SSH connection")
            ssh_config = SSHConfig(use_tnr_cli=True)
            ssh_manager = SSHManager(instance_uuid, ssh_config, thunder_cli_index=instance_details.thunder_cli_index)

            ssh_manager.setup_tnr_connection()
            ssh_manager.wait_for_instance_ready()

            info("Step 4: Uploading and running bootstrap script")
            self._run_bootstrap_script(ssh_manager, local_project_root)

            info("Step 5: Configuring AWS credentials")
            self._setup_aws_credentials(ssh_manager)

            info("Step 6: Creating S3 bucket and updating environment")
            self._setup_s3_bucket_and_env(ssh_manager, s3_bucket_name, workers, object_avoidance_config)

            info("Step 7: Creating template environment files")
            run_id = f"{model_id}-{int(time.time())}"
            self._create_template_env_files(ssh_manager, model_id, s3_bucket_name, workers, run_id, object_avoidance_config)

            info("Step 7a: Creating local runs directory structure")
            self._create_local_runs_directory(model_id, run_id, local_project_root, workers, object_avoidance_config)

            info("Step 8: Uploading model files")
            self._upload_model_files(ssh_manager, model_id, local_project_root)

            info("Step 9: Starting DeepRacer training")
            self._start_deepracer_training(ssh_manager)

            info(f"✅ DeepRacer deployment completed successfully!")
            info(f"Instance UUID: {instance_uuid}")
            info(f"Model ID: {model_id}")
            info(f"Run ID: {run_id}")
            info(f"S3 Bucket: {s3_bucket_name}")
            info(f"Workers: {workers}")

            return instance_uuid, run_id

        except Exception as e:
            error("DeepRacer deployment failed", extra={"model_id": model_id, "instance_uuid": instance_uuid, "error": str(e)})

            if instance_uuid:
                warning(f"Instance {instance_uuid} was created but deployment failed")
                warning("You may want to clean up the instance manually")

            raise ThunderDeploymentError(f"DeepRacer deployment failed: {e}")

    def _run_bootstrap_script(self, ssh_manager: SSHManager, project_root: Path) -> None:
        """Upload and run the bootstrap script on the instance."""
        bootstrap_script = project_root / "vm" / "bootstrap.sh"

        if not bootstrap_script.exists():
            raise ThunderDeploymentError(f"Bootstrap script not found: {bootstrap_script}")

        info("Uploading bootstrap script")
        ssh_manager.upload_file(str(bootstrap_script), "/tmp/bootstrap.sh")

        info("Making bootstrap script executable")
        result = ssh_manager.execute_command("chmod +x /tmp/bootstrap.sh")
        if not result.success:
            raise ThunderDeploymentError(f"Failed to make bootstrap executable: {result.stderr}")

        info("Running bootstrap script (this may take several minutes)")
        result = ssh_manager.execute_command("/tmp/bootstrap.sh", timeout=1800)
        if not result.success:
            raise ThunderDeploymentError(f"Bootstrap script failed: {result.stderr}")

        info("Bootstrap script completed successfully")

    def _setup_aws_credentials(self, ssh_manager: SSHManager) -> None:
        """Configure AWS credentials using local default profile and set up minio profile."""
        import os

        info("Reading AWS credentials from local default profile")

        try:
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

            if not aws_access_key or not aws_secret_key:
                try:
                    import configparser

                    credentials_path = Path.home() / ".aws" / "credentials"

                    if credentials_path.exists():
                        config = configparser.ConfigParser()
                        config.read(credentials_path)

                        if "default" in config:
                            aws_access_key = config["default"].get("aws_access_key_id")
                            aws_secret_key = config["default"].get("aws_secret_access_key")
                            info("Loaded AWS credentials from ~/.aws/credentials default profile")
                        else:
                            warning("No default profile found in ~/.aws/credentials")
                    else:
                        warning("~/.aws/credentials file not found")

                except Exception as e:
                    warning(f"Failed to read credentials file: {e}")

            if not aws_access_key or not aws_secret_key:
                warning("AWS credentials not found, using placeholder values")
                aws_access_key = "minio"
                aws_secret_key = "miniokey"

        except Exception as e:
            warning(f"Error loading credentials: {e}")
            aws_access_key = "minio"
            aws_secret_key = "miniokey"

        info("Configuring AWS credentials on remote instance")

        commands = [
            "mkdir -p ~/.aws",
            f"cat > ~/.aws/credentials << EOF\n[default]\naws_access_key_id = {aws_access_key}\naws_secret_access_key = {aws_secret_key}\n\n[minio]\naws_access_key_id = {aws_access_key}\naws_secret_access_key = {aws_secret_key}\nEOF",
            f"cat > ~/.aws/config << EOF\n[default]\nregion = us-east-1\n\n[profile minio]\nregion = us-east-1\nEOF",
        ]

        for cmd in commands:
            result = ssh_manager.execute_command(cmd)
            if not result.success:
                raise ThunderDeploymentError(f"Failed to configure AWS credentials: {result.stderr}")

        info("AWS credentials and configuration set up successfully")
        info("Configured profiles: default and minio, both with us-east-1 region")

    def _setup_s3_bucket_and_env(
        self, ssh_manager: SSHManager, bucket_name: str, workers: int = 1, object_avoidance_config: Optional[dict] = None
    ) -> None:
        """Create S3 bucket locally and create system.env from template."""
        info(f"Creating S3 bucket: {bucket_name}")

        try:
            from deepracer_research.deployment.thunder_compute.ssh.aws_setup import create_s3_bucket_locally

            bucket_created = create_s3_bucket_locally(bucket_name)
            if bucket_created:
                info(f"S3 bucket {bucket_name} created successfully")
            else:
                warning(f"S3 bucket {bucket_name} may already exist or creation failed")
        except ImportError:
            info("Creating S3 bucket on remote instance")
            result = ssh_manager.execute_command(f"aws s3 mb s3://{bucket_name} --region us-east-1")
            if not result.success:
                warning(f"S3 bucket creation may have failed: {result.stderr}")
                info("Continuing with deployment - bucket may already exist")

        info("Creating system.env from template")
        system_env_content = self._generate_system_env_template(bucket_name, workers, object_avoidance_config)

        command = f"cat > ~/deepracer-for-cloud/system.env << EOF\n{system_env_content}\nEOF"
        result = ssh_manager.execute_command(command)
        if not result.success:
            raise ThunderDeploymentError(f"Failed to create system.env: {result.stderr}")

        race_type = object_avoidance_config.get("race_type", "TIME_TRIAL") if object_avoidance_config else "TIME_TRIAL"
        info(
            f"Environment configured with S3 bucket: {bucket_name}, workers: {workers}, race type: {race_type}, track: reinvent_base"
        )

    def _create_template_env_files(
        self,
        ssh_manager: SSHManager,
        model_id: str,
        bucket_name: str,
        workers: int = 1,
        run_id: str = None,
        object_avoidance_config: Optional[dict] = None,
    ) -> None:
        """Create template-run.env and worker env files for multi-worker DeepRacer deployment."""
        info(f"Creating template environment files for {workers} worker(s)")

        if run_id is None:
            run_id = f"{model_id}-{int(time.time())}"

        run_env_content = self._generate_run_env_template(model_id, bucket_name, workers, run_id, object_avoidance_config)

        commands = [f"cat > ~/deepracer-for-cloud/defaults/template-run.env << EOF\n{run_env_content}\nEOF"]

        if workers > 1:
            commands.append("mkdir -p ~/deepracer-for-cloud")

            available_tracks = ["reinvent_base", "oval_track", "bowtie_track"]
            worker_colors = ["Blue", "Green", "Orange"]

            for worker_num in range(2, min(workers + 1, 4)):
                worker_track = available_tracks[min(worker_num - 1, len(available_tracks) - 1)]
                worker_color = worker_colors[min(worker_num - 1, len(worker_colors) - 1)]

                worker_env_content = self._generate_worker_env_template(
                    model_id, bucket_name, worker_num, workers, run_id, worker_track, worker_color, object_avoidance_config
                )

                commands.append(f"cat > ~/deepracer-for-cloud/worker-{worker_num}.env << EOF\n{worker_env_content}\nEOF")

        for cmd in commands:
            result = ssh_manager.execute_command(cmd)
            if not result.success:
                raise ThunderDeploymentError(f"Failed to create template environment files: {result.stderr}")

        info(f"Template environment files created successfully for {workers} worker(s)")

    def _load_template(self, template_name: str) -> str:
        """Load template file content."""
        template_path = Path(__file__).parent.parent.parent / "templates" / template_name

        if not template_path.exists():
            raise ThunderDeploymentError(f"Template file not found: {template_path}")

        return template_path.read_text()

    def _render_template(self, template_content: str, context: dict) -> str:
        """Render Jinja2 template with context variables."""
        try:
            from jinja2 import Template

            template = Template(template_content)
            return template.render(**context)
        except ImportError:
            warning("Jinja2 not available, using simple string replacement")
            result = template_content
            for key, value in context.items():
                result = result.replace(f"{{{{ {key} }}}}", str(value))
                result = result.replace(f"{{{{ {key} | default(", str(value) + " # default(")
            return result

    def _generate_system_env_template(
        self, bucket_name: str, workers: int = 1, object_avoidance_config: Optional[dict] = None
    ) -> str:
        """Generate system.env content from template."""
        template_content = self._load_template("template-system.env.j2")

        race_type = "TIME_TRIAL"
        if object_avoidance_config:
            race_type = object_avoidance_config.get("race_type", "TIME_TRIAL")

        context = {
            "bucket_name": bucket_name,
            "track_name": "reinvent_base",
            "race_type": race_type,
            "s3_profile": "default",
            "upload_bucket": bucket_name,
            "upload_profile": "default",
            "workers": str(workers),
            "docker_style": "compose",
            "host_x": "False",
            "sagemaker_image": "crr0004/sagemaker:console",
            "robomaker_image": "crr0004/robomaker:console",
            "coach_image": "crr0004/coach:console",
            "minio_image": "minio/minio:latest",
            "webviewer_port": "8100",
            "cloud_type": "local",
            "aws_region": "us-east-1",
            "mount_logs": "True",
            "cloudwatch_enable": "False",
            "gui_enable": "False",
            "kinesis_enable": "False",
            "kinesis_stream_name": "",
        }

        return self._render_template(template_content, context)

    def _generate_run_env_template(
        self,
        model_id: str,
        bucket_name: str,
        workers: int = 1,
        run_id: str = None,
        object_avoidance_config: Optional[dict] = None,
    ) -> str:
        """Generate template-run.env content based on model configuration."""
        template_content = self._load_template("template-run.env.j2")

        if run_id is None:
            run_id = "0"

        round_robin_distance = "0.05"
        if workers > 1:
            round_robin_distance = str(round(1.0 / (workers * 10), 2))

        race_type = "TIME_TRIAL"
        oa_obstacles = "6"
        oa_min_distance = "2.0"
        oa_randomize = "False"
        oa_bot_car = "False"

        if object_avoidance_config:
            race_type = object_avoidance_config.get("race_type", "TIME_TRIAL")
            if race_type == "OBJECT_AVOIDANCE":
                oa_obstacles = str(object_avoidance_config.get("num_obstacles", 6))
                oa_min_distance = str(object_avoidance_config.get("obstacle_distance", 2.0))
                oa_randomize = str(object_avoidance_config.get("randomize_obstacles", False))
                oa_bot_car = str(object_avoidance_config.get("bot_car_obstacles", False))

        import random

        h2b_lane_change = "False"
        h2b_lower_time = str(random.uniform(2.0, 4.0))
        h2b_upper_time = str(random.uniform(4.0, 6.0))
        h2b_change_distance = str(random.uniform(0.8, 1.2))
        h2b_bot_cars = str(random.randint(2, 4))
        h2b_min_distance = str(random.uniform(1.5, 2.5))
        h2b_randomize = "False"
        h2b_bot_speed = str(random.uniform(0.15, 0.25))
        h2b_bot_penalty = "5.0"

        context = {
            "model_id": model_id,
            "bucket_name": bucket_name,
            "run_id": run_id,
            "track_name": "reinvent_base",
            "race_type": race_type,
            "car_color": "Red",
            "s3_profile": "default",
            "upload_bucket": bucket_name,
            "upload_profile": "default",
            "upload_prefix": model_id,
            "workers": str(workers),
            "multi_config": "True" if workers > 1 else "False",
            "change_start_position": "True",
            "alternate_direction": "False",
            "start_position_offset": "0.00",
            "round_robin_distance": round_robin_distance,
            "min_eval_trials": "5",
            "best_model_metric": "progress",
            "max_steps_per_iteration": "10000",
            "eval_trials": "3",
            "eval_continuous": "True",
            "off_track_penalty": "5.0",
            "collision_penalty": "5.0",
            "save_mp4": "True",
            "gui_enable": "False",
            "kinesis_enable": "False",
            "cloudwatch_enable": "False",
            "mount_logs": "True",
            "cloud_type": "local",
            "aws_region": "us-east-1",
            "oa_obstacles": oa_obstacles,
            "oa_min_distance": oa_min_distance,
            "oa_randomize": oa_randomize,
            "oa_bot_car": oa_bot_car,
            "h2b_lane_change": h2b_lane_change,
            "h2b_lower_time": h2b_lower_time,
            "h2b_upper_time": h2b_upper_time,
            "h2b_change_distance": h2b_change_distance,
            "h2b_bot_cars": h2b_bot_cars,
            "h2b_min_distance": h2b_min_distance,
            "h2b_randomize": h2b_randomize,
            "h2b_bot_speed": h2b_bot_speed,
            "h2b_bot_penalty": h2b_bot_penalty,
        }

        return self._render_template(template_content, context)

    def _generate_worker_env_template(
        self,
        model_id: str,
        bucket_name: str,
        worker_number: int,
        total_workers: int,
        run_id: str,
        worker_track: str = "reinvent_base",
        worker_color: str = "Blue",
        object_avoidance_config: Optional[dict] = None,
    ) -> str:
        """Generate worker env content for specific worker in multi-worker training."""
        template_content = self._load_template("template-worker.env.j2")

        round_robin_distance = str(round(1.0 / (total_workers * 10), 2))

        race_type = "TIME_TRIAL"
        if object_avoidance_config:
            race_type = object_avoidance_config.get("race_type", "TIME_TRIAL")

        import random

        oa_obstacles = "6"
        oa_min_distance = "2.0"
        oa_randomize = "False"
        oa_bot_car = "False"

        if object_avoidance_config and race_type == "OBJECT_AVOIDANCE":
            oa_obstacles = str(object_avoidance_config.get("num_obstacles", 6))
            oa_min_distance = str(object_avoidance_config.get("obstacle_distance", 2.0))
            oa_randomize = str(object_avoidance_config.get("randomize_obstacles", False))
            oa_bot_car = str(object_avoidance_config.get("bot_car_obstacles", False))

        h2b_lane_change = "False"
        h2b_lower_time = str(random.uniform(2.0, 4.0))
        h2b_upper_time = str(random.uniform(4.0, 6.0))
        h2b_change_distance = str(random.uniform(0.8, 1.2))
        h2b_bot_cars = str(random.randint(2, 4))
        h2b_min_distance = str(random.uniform(1.5, 2.5))
        h2b_randomize = "False"
        h2b_bot_speed = str(random.uniform(0.15, 0.25))
        h2b_bot_penalty = "5.0"

        context = {
            "model_id": model_id,
            "bucket_name": bucket_name,
            "worker_number": str(worker_number),
            "workers": str(total_workers),
            "track_name": "reinvent_base",
            "worker_track": worker_track,
            "race_type": race_type,
            "worker_car_color": worker_color,
            "s3_profile": "default",
            "multi_config": "True",
            "change_start_position": "True",
            "round_robin_distance": round_robin_distance,
            "start_position_offset": "0.00",
            "min_eval_trials": "5",
            "gui_enable": "False",
            "kinesis_enable": "False",
            "cloud_type": "local",
            "aws_region": "us-east-1",
            "oa_obstacles": oa_obstacles,
            "oa_min_distance": oa_min_distance,
            "oa_randomize": oa_randomize,
            "oa_bot_car": oa_bot_car,
            "h2b_lane_change": h2b_lane_change,
            "h2b_lower_time": h2b_lower_time,
            "h2b_upper_time": h2b_upper_time,
            "h2b_change_distance": h2b_change_distance,
            "h2b_bot_cars": h2b_bot_cars,
            "h2b_min_distance": h2b_min_distance,
            "h2b_randomize": h2b_randomize,
            "h2b_bot_speed": h2b_bot_speed,
            "h2b_bot_penalty": h2b_bot_penalty,
            "save_mp4": "True",
        }

        return self._render_template(template_content, context)

    def _create_local_runs_directory(
        self, model_id: str, run_id: str, project_root: Path, workers: int, object_avoidance_config: Optional[dict] = None
    ) -> None:
        """Create local runs directory structure for tracking deployments."""
        runs_dir = project_root / "runs" / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)

        info(f"Created local runs directory: {runs_dir}")

        race_type = "TIME_TRIAL"
        if object_avoidance_config:
            race_type = object_avoidance_config.get("race_type", "TIME_TRIAL")

        deployment_info = {
            "model_id": model_id,
            "run_id": run_id,
            "workers": workers,
            "race_type": race_type,
            "deployment_type": "thunder_compute_bootstrap",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "deploying",
        }

        if object_avoidance_config and race_type == "OBJECT_AVOIDANCE":
            deployment_info["object_avoidance"] = {
                "num_obstacles": object_avoidance_config.get("num_obstacles", 3),
                "randomize_obstacles": object_avoidance_config.get("randomize_obstacles", True),
                "obstacle_distance": object_avoidance_config.get("obstacle_distance", 2.0),
                "bot_car_obstacles": object_avoidance_config.get("bot_car_obstacles", False),
            }

        if workers > 1:
            available_tracks = ["reinvent_base", "oval_track", "bowtie_track"]
            worker_colors = ["Blue", "Green", "Orange"]

            deployment_info["worker_configurations"] = []
            for worker_num in range(1, min(workers + 1, 4)):
                worker_track = available_tracks[min(worker_num - 1, len(available_tracks) - 1)]
                worker_color = worker_colors[min(worker_num - 1, len(worker_colors) - 1)] if worker_num > 1 else "Red"

                deployment_info["worker_configurations"].append(
                    {"worker_number": worker_num, "track": worker_track, "car_color": worker_color}
                )

        deployment_file = runs_dir / "deployment.json"
        with open(deployment_file, "w") as f:
            import json

            json.dump(deployment_info, f, indent=2)

        info(f"Deployment configuration saved to: {deployment_file}")

    def _upload_model_files(self, ssh_manager: SSHManager, model_id: str, project_root: Path) -> None:
        """Upload model files to deepracer-for-cloud/custom_files."""
        model_dir = project_root / "models" / model_id

        if not model_dir.exists():
            raise ThunderDeploymentError(f"Model directory not found: {model_dir}")

        info(f"Uploading model files from {model_dir}")

        result = ssh_manager.execute_command("mkdir -p ~/deepracer-for-cloud/custom_files")
        if not result.success:
            raise ThunderDeploymentError(f"Failed to create custom_files directory: {result.stderr}")

        for file in ["hyperparameters.json", "model_metadata.json", "reward_function.py"]:
            local_file = model_dir / file
            if local_file.exists():
                remote_path = f"~/deepracer-for-cloud/custom_files/{file}"
                ssh_manager.upload_file(str(local_file), remote_path)
                info(f"Uploaded: {file}")
            else:
                warning(f"Model file not found: {local_file}")

        info("Model files uploaded successfully")

    def _start_deepracer_training(self, ssh_manager: SSHManager) -> None:
        """Run the DeepRacer training commands."""
        info("Starting DeepRacer training sequence")

        commands = [
            "cd ~/deepracer-for-cloud",
            "source bin/activate.sh",
            "dr-update",
            "dr-update-env",
            "dr-upload-custom-files",
            "dr-start-training last",
        ]

        full_command = " && ".join(commands)

        result = ssh_manager.execute_command(full_command, timeout=600)
        if not result.success:
            raise ThunderDeploymentError(f"DeepRacer training start failed: {result.stderr}")

        info("DeepRacer training started successfully")
        info("Training output:")
        info(result.stdout)

    @classmethod
    def create_training_deployment(
        cls,
        api_token: str,
        cpu_cores: int = 8,
        gpu_type: GPUType = GPUType.T4,
        disk_size_gb: int = 100,
        s3_bucket_name: Optional[str] = None,
    ) -> "ThunderDeploymentManager":
        """Create a deployment manager with training configuration.

        Parameters
        ----------
        api_token : str
            Thunder Compute API token.
        cpu_cores : int, optional
            Number of CPU cores, by default 8.
        gpu_type : GPUType, optional
            GPU type for training, by default GPUType.T4.
        disk_size_gb : int, optional
            Disk size in GB, by default 100.
        s3_bucket_name : str, optional
            S3 bucket for model storage, by default None.

        Returns
        -------
        ThunderDeploymentManager
            Configured deployment manager.
        """
        thunder_config = ThunderComputeConfig(api_token=api_token)
        return cls(thunder_config)

    @classmethod
    def create_evaluation_deployment(
        cls, api_token: str, cpu_cores: int = 4, gpu_type: GPUType = GPUType.T4, disk_size_gb: int = 50
    ) -> "ThunderDeploymentManager":
        """Create a deployment manager with evaluation configuration.

        Parameters
        ----------
        api_token : str
            Thunder Compute API token.
        cpu_cores : int, optional
            Number of CPU cores, by default 4.
        gpu_type : GPUType, optional
            GPU type for evaluation, by default GPUType.T4.
        disk_size_gb : int, optional
            Disk size in GB, by default 50.

        Returns
        -------
        ThunderDeploymentManager
            Configured deployment manager.
        """
        thunder_config = ThunderComputeConfig(api_token=api_token)
        return cls(thunder_config)
