import time
from typing import Any, Dict, List, Optional

from deepracer_research.deployment.aws_ec2.api import AMIFinder, EC2Client, VPCManager
from deepracer_research.deployment.aws_ec2.config import AWSConfig, EC2InstanceConfig, EC2SSHConfig
from deepracer_research.deployment.aws_ec2.installation import EC2DeepRacerInstaller
from deepracer_research.deployment.aws_ec2.management.deployment_result import EC2DeploymentResult
from deepracer_research.deployment.aws_ec2.management.errors import EC2DeploymentError
from deepracer_research.deployment.aws_ec2.models.api import EC2ApiError
from deepracer_research.deployment.aws_ec2.models.instance import EC2InstanceDetails, EC2InstanceInfo
from deepracer_research.deployment.aws_ec2.ssh import EC2SSHManager
from deepracer_research.deployment.deepracer.deep_racer_deployment_config import DeepRacerDeploymentConfig
from deepracer_research.utils import debug, error, info, warning


class EC2DeploymentManager:
    """Manager for AWS EC2 deployments with DeepRacer integration"""

    def __init__(self, aws_config: AWSConfig):
        """Initialize deployment manager.

        Parameters
        ----------
        aws_config : AWSConfig
            AWS configuration for deployments.
        """
        self.aws_config = aws_config
        self.ec2_client = EC2Client(aws_config)
        self.ami_finder = AMIFinder(aws_config)
        self.vpc_manager = VPCManager(aws_config)
        self.active_deployments: Dict[str, Dict[str, Any]] = {}

    def deploy_deepracer_instance(
        self,
        instance_config: EC2InstanceConfig,
        ssh_config: Optional[EC2SSHConfig] = None,
        deepracer_config: Optional[DeepRacerDeploymentConfig] = None,
        wait_for_ready: bool = True,
    ) -> EC2DeploymentResult:
        """Deploy a complete DeepRacer training instance on EC2.

        Parameters
        ----------
        instance_config : EC2InstanceConfig
            EC2 instance configuration.
        ssh_config : EC2SSHConfig, optional
            SSH configuration, by default None (uses defaults).
        deepracer_config : DeepRacerDeploymentConfig, optional
            DeepRacer configuration, by default None.
        wait_for_ready : bool, optional
            Whether to wait for instance to be ready, by default True.

        Returns
        -------
        EC2DeploymentResult
            Deployment result.

        Raises
        ------
        EC2DeploymentError
            If deployment fails.
        """
        info(
            "Starting EC2 DeepRacer instance deployment",
            extra={"instance_type": instance_config.instance_type.value, "region": instance_config.region.value},
        )

        instance_id = None
        ssh_manager = None

        try:
            info("Preparing AWS infrastructure")
            infrastructure = self._prepare_infrastructure(instance_config)

            info("Finding suitable AMI")
            if not instance_config.ami_id:
                instance_config.ami_id = self._find_ami(instance_config)

            info("Creating EC2 instance")
            instance_response = self.ec2_client.create_instance(instance_config.to_dict(), wait_for_running=wait_for_ready)
            instance_id = instance_response.instance_id

            info("Instance created successfully", extra={"instance_id": instance_id})

            if wait_for_ready:
                instance_details = self._wait_for_instance_running(instance_id)
            else:
                instance_details = self.ec2_client.get_instance(instance_id)

            ssh_config = ssh_config or EC2SSHConfig.for_ec2_default()
            if instance_config.key_name and not ssh_config.private_key_path:
                ssh_config.private_key_path = f"~/.ssh/{instance_config.key_name}.pem"

            ssh_manager = EC2SSHManager(instance_id, instance_details.public_ip, ssh_config)

            ssh_ready = False
            if wait_for_ready and instance_details.public_ip:
                info("Waiting for SSH connectivity")
                ssh_ready = ssh_manager.wait_for_instance_ready()
                if not ssh_ready:
                    warning("SSH connection not ready")

            deepracer_installed = False
            if instance_config.install_deepracer_cloud and ssh_ready:
                info("Installing DeepRacer-for-Cloud")
                installer = EC2DeepRacerInstaller(ssh_manager)

                skip_gpu = not instance_config.instance_type.has_gpu

                installer.install_deepracer_cloud(s3_bucket_name=instance_config.s3_bucket_name, skip_gpu_setup=skip_gpu)
                deepracer_installed = True

                if instance_config.s3_bucket_name:
                    info("Configuring AWS environment for DeepRacer")
                    self._setup_aws_environment(ssh_manager, instance_config.s3_bucket_name)

            self.active_deployments[instance_id] = {
                "instance_config": instance_config,
                "ssh_manager": ssh_manager,
                "installer": EC2DeepRacerInstaller(ssh_manager) if deepracer_installed else None,
                "created_at": time.time(),
                "infrastructure": infrastructure,
            }

            info(
                "EC2 DeepRacer instance deployment completed",
                extra={"instance_id": instance_id, "hostname": instance_details.public_ip},
            )

            return EC2DeploymentResult(
                success=True,
                instance_id=instance_id,
                hostname=instance_details.public_ip,
                ssh_ready=ssh_ready,
                deepracer_installed=deepracer_installed,
            )

        except Exception as e:
            error("EC2 deployment failed", extra={"error": str(e)})

            if instance_id:
                try:
                    info("Cleaning up failed deployment", extra={"instance_id": instance_id})
                    self.ec2_client.terminate_instance(instance_id)
                except Exception as cleanup_error:
                    warning("Failed to cleanup instance", extra={"instance_id": instance_id, "error": str(cleanup_error)})

            return EC2DeploymentResult(
                success=False,
                instance_id=instance_id,
                hostname=None,
                ssh_ready=False,
                deepracer_installed=False,
                error_message=str(e),
            )

    def _prepare_infrastructure(self, instance_config: EC2InstanceConfig) -> Dict[str, Any]:
        """Prepare AWS infrastructure for deployment.

        Parameters
        ----------
        instance_config : EC2InstanceConfig
            Instance configuration.

        Returns
        -------
        Dict[str, Any]
            Infrastructure details.
        """
        infrastructure = {}

        if not instance_config.vpc_id:
            vpc_info = self.vpc_manager.get_default_vpc()
            if vpc_info:
                instance_config.vpc_id = vpc_info.vpc_id
                instance_config.subnet_id = vpc_info.subnet_id
                infrastructure["vpc_info"] = vpc_info

        if not instance_config.security_group_ids:
            sg_id = self.vpc_manager.create_deepracer_security_group(instance_config.vpc_id)
            instance_config.security_group_ids = [sg_id]
            infrastructure["security_group_id"] = sg_id

        return infrastructure

    def _find_ami(self, instance_config: EC2InstanceConfig) -> str:
        """Find suitable AMI for the instance.

        Parameters
        ----------
        instance_config : EC2InstanceConfig
            Instance configuration.

        Returns
        -------
        str
            AMI ID.

        Raises
        ------
        EC2DeploymentError
            If no suitable AMI is found.
        """
        ami_id = self.ami_finder.find_deepracer_optimized_ami()

        if not ami_id:
            ami_id = self.ami_finder.find_ubuntu_ami("22.04")

        if not ami_id:
            raise EC2DeploymentError("No suitable AMI found")

        info("Selected AMI", extra={"ami_id": ami_id})
        return ami_id

    def _wait_for_instance_running(self, instance_id: str, timeout: int = 300, check_interval: int = 10) -> EC2InstanceDetails:
        """Wait for EC2 instance to be running.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.
        timeout : int, optional
            Maximum wait time in seconds, by default 300.
        check_interval : int, optional
            Check interval in seconds, by default 10.

        Returns
        -------
        EC2InstanceDetails
            Instance details when running.

        Raises
        ------
        EC2DeploymentError
            If instance doesn't become running within timeout.
        """
        info("Waiting for instance to be running", extra={"instance_id": instance_id, "timeout": timeout})

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                instance = self.ec2_client.get_instance(instance_id)

                if instance.status.value == "running":
                    info("Instance is running", extra={"instance_id": instance_id})
                    return instance

                debug("Instance status", extra={"instance_id": instance_id, "status": instance.status.value})

            except EC2ApiError as e:
                debug("Error checking instance status", extra={"instance_id": instance_id, "error": str(e)})

            time.sleep(check_interval)

        raise EC2DeploymentError(f"Instance {instance_id} did not become running within {timeout} seconds")

    def _setup_aws_environment(self, ssh_manager: EC2SSHManager, s3_bucket_name: str) -> None:
        """Setup AWS environment on the instance.

        Parameters
        ----------
        ssh_manager : EC2SSHManager
            SSH manager for the instance.
        s3_bucket_name : str
            S3 bucket name for DeepRacer.
        """
        info("Setting up AWS environment", extra={"instance_id": ssh_manager.instance_id, "s3_bucket": s3_bucket_name})

        aws_setup_script = f"""
        cd ~/deepracer-for-cloud

        sed -i 's/DR_LOCAL_S3_BUCKET=.*/DR_LOCAL_S3_BUCKET={s3_bucket_name}/' system.env

        aws sts get-caller-identity

        aws s3 mb s3://{s3_bucket_name} --region {self.aws_config.region.value} || true
        """

        result = ssh_manager.execute_script(aws_setup_script, "setup_aws_env.sh")
        if not result.success:
            warning("AWS environment setup had issues", extra={"instance_id": ssh_manager.instance_id, "error": result.stderr})

    def list_instances(self, include_terminated: bool = False) -> List[EC2InstanceInfo]:
        """List EC2 instances.

        Parameters
        ----------
        include_terminated : bool, optional
            Include terminated instances, by default False.

        Returns
        -------
        List[EC2InstanceInfo]
            List of instance information.
        """
        return self.ec2_client.list_instances(include_terminated)

    def get_instance_details(self, instance_id: str) -> EC2InstanceDetails:
        """Get detailed information about an instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        EC2InstanceDetails
            Detailed instance information.
        """
        return self.ec2_client.get_instance(instance_id)

    def delete_instance(self, instance_id: str) -> bool:
        """Delete an EC2 instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        bool
            True if deletion was successful.
        """
        try:
            if instance_id in self.active_deployments:
                deployment = self.active_deployments[instance_id]
                if "ssh_manager" in deployment:
                    deployment["ssh_manager"].disconnect()
                del self.active_deployments[instance_id]

            return self.ec2_client.terminate_instance(instance_id)

        except Exception as e:
            error("Failed to delete instance", extra={"instance_id": instance_id, "error": str(e)})
            return False

    def start_instance(self, instance_id: str) -> bool:
        """Start a stopped EC2 instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        bool
            True if start was successful.
        """
        return self.ec2_client.start_instance(instance_id)

    def stop_instance(self, instance_id: str) -> bool:
        """Stop a running EC2 instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        bool
            True if stop was successful.
        """
        return self.ec2_client.stop_instance(instance_id)

    def start_training(self, instance_id: str, training_name: str = "cli-training") -> bool:
        """Start DeepRacer training on an instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.
        training_name : str, optional
            Training session name, by default "cli-training".

        Returns
        -------
        bool
            True if training started successfully.
        """
        if instance_id not in self.active_deployments:
            raise EC2DeploymentError(f"Instance {instance_id} not found in active deployments")

        deployment = self.active_deployments[instance_id]
        ssh_manager = deployment.get("ssh_manager")

        if not ssh_manager:
            raise EC2DeploymentError(f"No SSH manager available for instance {instance_id}")

        info("Starting DeepRacer training", extra={"instance_id": instance_id, "training_name": training_name})

        training_script = f"""
        cd ~/deepracer-for-cloud
        source bin/activate.sh

        export DR_RUN_ID="{training_name}"

        dr-start-training
        """

        result = ssh_manager.execute_script(training_script, "start_training.sh")
        if result.success:
            info("Training started successfully", extra={"instance_id": instance_id, "training_name": training_name})
            return True
        else:
            error("Failed to start training", extra={"instance_id": instance_id, "error": result.stderr})
            return False

    def stop_training(self, instance_id: str) -> bool:
        """Stop DeepRacer training on an instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        bool
            True if training stopped successfully.
        """
        if instance_id not in self.active_deployments:
            raise EC2DeploymentError(f"Instance {instance_id} not found in active deployments")

        deployment = self.active_deployments[instance_id]
        ssh_manager = deployment.get("ssh_manager")

        if not ssh_manager:
            raise EC2DeploymentError(f"No SSH manager available for instance {instance_id}")

        info("Stopping DeepRacer training", extra={"instance_id": instance_id})

        stop_script = """
        cd ~/deepracer-for-cloud
        source bin/activate.sh

        dr-stop-training
        """

        result = ssh_manager.execute_script(stop_script, "stop_training.sh")
        return result.success

    def get_training_logs(self, instance_id: str) -> str:
        """Get training logs from an instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.

        Returns
        -------
        str
            Training logs.
        """
        if instance_id not in self.active_deployments:
            raise EC2DeploymentError(f"Instance {instance_id} not found in active deployments")

        deployment = self.active_deployments[instance_id]
        ssh_manager = deployment.get("ssh_manager")

        if not ssh_manager:
            raise EC2DeploymentError(f"No SSH manager available for instance {instance_id}")

        logs_script = """
        cd ~/deepracer-for-cloud
        source bin/activate.sh

        dr-logs-sagemaker -a
        """

        result = ssh_manager.execute_script(logs_script, "get_logs.sh")
        return result.output

    def execute_command(self, instance_id: str, command: str) -> str:
        """Execute a command on an instance.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.
        command : str
            Command to execute.

        Returns
        -------
        str
            Command output.
        """
        if instance_id not in self.active_deployments:
            raise EC2DeploymentError(f"Instance {instance_id} not found in active deployments")

        deployment = self.active_deployments[instance_id]
        ssh_manager = deployment.get("ssh_manager")

        if not ssh_manager:
            raise EC2DeploymentError(f"No SSH manager available for instance {instance_id}")

        result = ssh_manager.execute_command(command)
        return result.output
