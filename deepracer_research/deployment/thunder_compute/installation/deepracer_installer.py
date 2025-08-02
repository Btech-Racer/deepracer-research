from textwrap import dedent
from typing import Optional

from deepracer_research.deployment.thunder_compute.config.deepracer_cloud_config import DeepRacerCloudConfig
from deepracer_research.deployment.thunder_compute.config.instance_config import InstanceConfig
from deepracer_research.deployment.thunder_compute.installation.errors import DeepRacerInstallationError
from deepracer_research.deployment.thunder_compute.ssh import SSHManager
from deepracer_research.utils import error, info


class DeepRacerCloudInstaller:
    """Automated installer for DeepRacer-for-Cloud on Thunder Compute instances"""

    def __init__(self, ssh_manager: SSHManager, config: Optional[DeepRacerCloudConfig] = None):
        """Initialize DeepRacer installer.

        Parameters
        ------------
        ssh_manager : SSHManager
            SSH manager for the target instance.
        config : DeepRacerCloudConfig, optional
            DeepRacer Cloud configuration, by default None (uses defaults).
        """
        self.ssh_manager = ssh_manager
        self.config = config or DeepRacerCloudConfig()

        self.config.cloud_mode = "local"

    def install(self, instance_config: InstanceConfig) -> bool:
        """Install DeepRacer-for-Cloud on the instance.

        Parameters
        ------------
        instance_config : InstanceConfig
            Instance configuration with DeepRacer settings.

        Returns
        -------
        bool
            True if installation successful.

        Raises
        ------
        DeepRacerInstallationError
            If installation fails.
        """
        info("Starting DeepRacer-for-Cloud installation")

        try:
            self._install_system_dependencies()

            self._clone_repository()

            self._run_local_preparation()

            self._configure_environment(instance_config)

            self._initialize_deepracer()

            self._verify_installation()

            info("DeepRacer-for-Cloud installation completed successfully")
            return True

        except Exception as e:
            error("DeepRacer installation failed", extra={"error": str(e)})
            raise DeepRacerInstallationError(f"Installation failed: {e}")

    def _install_system_dependencies(self) -> None:
        """Install required system dependencies."""
        info("Installing system dependencies")

        script = dedent(
            """
            #!/bin/bash
            set -e

            # Update package list
            sudo apt-get update

            # Install required packages
            sudo apt-get install -y \\
                git \\
                docker.io \\
                docker-compose \\
                awscli \\
                python3-pip \\
                curl \\
                wget \\
                jq \\
                unzip

            # Start and enable Docker
            sudo systemctl start docker
            sudo systemctl enable docker

            # Add user to docker group
            sudo usermod -aG docker $USER

            # Install Docker Compose (latest version)
            sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose

            echo "System dependencies installed successfully"
        """
        )

        result = self.ssh_manager.execute_script(script, "install_dependencies.sh")
        if not result.success:
            raise DeepRacerInstallationError(f"Failed to install dependencies: {result.stderr}")

    def _clone_repository(self) -> None:
        """Clone DeepRacer-for-Cloud repository."""
        info("Cloning DeepRacer-for-Cloud repository")

        self.ssh_manager.execute_command(f"rm -rf {self.config.install_path}")

        result = self.ssh_manager.execute_command(f"git clone {self.config.git_url} {self.config.install_path}")

        if not result.success:
            raise DeepRacerInstallationError(f"Failed to clone repository: {result.stderr}")

        if self.config.branch != "main":
            result = self.ssh_manager.execute_command(f"cd {self.config.install_path} && git checkout {self.config.branch}")
            if not result.success:
                raise DeepRacerInstallationError(f"Failed to checkout branch: {result.stderr}")

    def _run_local_preparation(self) -> None:
        """Run local preparation instead of cloud preparation."""
        info("Running local preparation for DeepRacer")

        script = dedent(
            f"""
            #!/bin/bash
            set -e

            cd {self.config.install_path}

            # Install Docker Swarm (required for DeepRacer)
            # Check if swarm is already initialized
            if ! docker info | grep -q "Swarm: active"; then
                # Initialize Docker Swarm
                ADVERTISE_ADDR=$(hostname -I | cut -d' ' -f1)
                docker swarm init --advertise-addr $ADVERTISE_ADDR
            fi

            # Run initialization script
            ./bin/init.sh -c local -a {self.config.architecture}

            echo "Local preparation completed"
        """
        )

        result = self.ssh_manager.execute_script(script, "local_preparation.sh")
        if not result.success:
            raise DeepRacerInstallationError(f"Local preparation failed: {result.stderr}")

    def _configure_environment(self, instance_config: InstanceConfig) -> None:
        """Configure DeepRacer environment variables."""
        info("Configuring DeepRacer environment")

        env_vars = {
            "DR_CLOUD": self.config.cloud_mode,
            "DR_LOCAL_S3_PROFILE": "minio",
            "DR_LOCAL_S3_BUCKET": "bucket",
            "DR_UPLOAD_S3_PROFILE": instance_config.aws_profile,
            "DR_UPLOAD_S3_BUCKET": instance_config.s3_bucket_name or "deepracer-models",
            **instance_config.environment_variables,
        }

        config_script = dedent(
            f"""
            #!/bin/bash
            set -e

            cd {self.config.install_path}

            # Activate DeepRacer environment
            source bin/activate.sh

            # Configure Minio credentials (local S3 alternative)
            aws configure set aws_access_key_id minio --profile minio
            aws configure set aws_secret_access_key miniostorage --profile minio
            aws configure set region {self.config.aws_region} --profile minio
            aws configure set output json --profile minio

            # Update system.env with configuration
            cat >> system.env << 'EOF'
        """
        )

        for key, value in env_vars.items():
            if value:
                config_script += f"{key}={value}\n"

        config_script += dedent(
            """
            EOF

            # Update configuration
            dr-update

            echo "Environment configuration completed"
        """
        )

        result = self.ssh_manager.execute_script(config_script, "configure_environment.sh")
        if not result.success:
            raise DeepRacerInstallationError(f"Environment configuration failed: {result.stderr}")

    def _initialize_deepracer(self) -> None:
        """Initialize DeepRacer with default configuration."""
        info("Initializing DeepRacer with default configuration")

        script = dedent(
            f"""
            #!/bin/bash
            set -e

            cd {self.config.install_path}

            # Activate DeepRacer environment
            source bin/activate.sh

            # Copy default configuration files
            cp defaults/hyperparameters.json custom_files/
            cp defaults/model_metadata.json custom_files/
            cp defaults/reward_function.py custom_files/

            # Upload custom files to start minio
            dr-upload-custom-files

            echo "DeepRacer initialization completed"
        """
        )

        result = self.ssh_manager.execute_script(script, "initialize_deepracer.sh")
        if not result.success:
            raise DeepRacerInstallationError(f"DeepRacer initialization failed: {result.stderr}")

    def _verify_installation(self) -> None:
        """Verify DeepRacer installation."""
        info("Verifying DeepRacer installation")

        script = dedent(
            f"""
            #!/bin/bash
            set -e

            cd {self.config.install_path}

            # Activate DeepRacer environment
            source bin/activate.sh

            # Check if Docker containers are running
            echo "Checking Docker containers..."
            docker ps

            # Check DeepRacer commands
            echo "Testing DeepRacer commands..."
            type dr-start-training
            type dr-stop-training
            type dr-logs

            # Check if minio is accessible
            echo "Testing Minio connection..."
            docker logs minio 2>/dev/null | tail -5 || echo "Minio container not found - will start with training"

            echo "DeepRacer installation verification completed"
        """
        )

        result = self.ssh_manager.execute_script(script, "verify_installation.sh")
        if not result.success:
            raise DeepRacerInstallationError(f"Installation verification failed: {result.stderr}")

    def start_training(self, training_name: str = "test-training") -> bool:
        """Start a test training session.

        Parameters
        ------------
        training_name : str, optional
            Name for the training session, by default "test-training".

        Returns
        -------
        bool
            True if training started successfully.

        Raises
        ------
        DeepRacerInstallationError
            If training start fails.
        """
        info("Starting DeepRacer training", extra={"training_name": training_name})

        script = dedent(
            f"""
            #!/bin/bash
            set -e

            cd {self.config.install_path}

            # Activate DeepRacer environment
            source bin/activate.sh

            # Start training
            export DR_RUN_ID="{training_name}"
            dr-start-training

            # Wait a moment and check status
            sleep 10
            dr-logs

            echo "Training started successfully"
        """
        )

        result = self.ssh_manager.execute_script(script, "start_training.sh")
        if not result.success:
            raise DeepRacerInstallationError(f"Failed to start training: {result.stderr}")

        return True

    def stop_training(self) -> bool:
        """Stop current training session.

        Returns
        -------
        bool
            True if training stopped successfully.
        """
        info("Stopping DeepRacer training")

        script = dedent(
            f"""
            #!/bin/bash
            set -e

            cd {self.config.install_path}

            # Activate DeepRacer environment
            source bin/activate.sh

            # Stop training
            dr-stop-training

            echo "Training stopped successfully"
        """
        )

        result = self.ssh_manager.execute_script(script, "stop_training.sh")
        return result.success

    def get_training_logs(self) -> str:
        """Get current training logs.

        Returns
        -------
        str
            Training logs as string.
        """
        info("Retrieving training logs")

        result = self.ssh_manager.execute_command(f"cd {self.config.install_path} && source bin/activate.sh && dr-logs")

        return result.stdout if result.success else result.stderr
