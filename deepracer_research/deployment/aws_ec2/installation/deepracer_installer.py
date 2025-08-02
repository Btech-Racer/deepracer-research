from typing import Optional

from deepracer_research.deployment.aws_ec2.installation.errors import DeepRacerInstallationError
from deepracer_research.deployment.aws_ec2.ssh import EC2SSHManager
from deepracer_research.utils import error, info


class EC2DeepRacerInstaller:
    """Installs DeepRacer-for-Cloud on EC2 instances"""

    def __init__(self, ssh_manager: EC2SSHManager):
        """Initialize DeepRacer installer.

        Parameters
        ----------
        ssh_manager : EC2SSHManager
            SSH manager for the EC2 instance.
        """
        self.ssh_manager = ssh_manager
        self.instance_id = ssh_manager.instance_id

    def install_deepracer_cloud(
        self, s3_bucket_name: Optional[str] = None, aws_profile: str = "default", skip_gpu_setup: bool = False
    ) -> bool:
        """Install DeepRacer-for-Cloud on the EC2 instance using bootstrap script.

        Parameters
        ----------
        s3_bucket_name : str, optional
            S3 bucket name for DeepRacer, by default None.
        aws_profile : str, optional
            AWS profile to use, by default "default".
        skip_gpu_setup : bool, optional
            Skip GPU setup (for CPU-only instances), by default False.

        Returns
        -------
        bool
            True if installation was successful.

        Raises
        ------
        DeepRacerInstallationError
            If installation fails.
        """
        info(
            "Starting DeepRacer-for-Cloud installation using bootstrap script",
            extra={
                "instance_id": self.instance_id,
                "s3_bucket": s3_bucket_name,
                "aws_profile": aws_profile,
                "skip_gpu_setup": skip_gpu_setup,
            },
        )

        try:
            if not self._run_bootstrap_script(skip_gpu_setup):
                raise DeepRacerInstallationError("Failed to run bootstrap script")

            if not self._configure_deepracer_environment(s3_bucket_name, aws_profile):
                raise DeepRacerInstallationError("Failed to configure DeepRacer environment")

            if not self.verify_installation():
                raise DeepRacerInstallationError("Installation verification failed")

            info("DeepRacer-for-Cloud installation completed successfully", extra={"instance_id": self.instance_id})

            return True

        except Exception as e:
            error("DeepRacer installation failed", extra={"instance_id": self.instance_id, "error": str(e)})
            raise DeepRacerInstallationError(f"Installation failed: {e}")

    def _run_bootstrap_script(self, skip_gpu_setup: bool = False) -> bool:
        """Upload and run the VM bootstrap script.

        Parameters
        ----------
        skip_gpu_setup : bool, optional
            Skip GPU setup for CPU-only instances, by default False.

        Returns
        -------
        bool
            True if bootstrap script ran successfully.
        """
        info("Running VM bootstrap script", extra={"instance_id": self.instance_id, "skip_gpu_setup": skip_gpu_setup})

        bootstrap_script_path = "vm/bootstrap.sh"
        try:
            with open(bootstrap_script_path, "r") as f:
                bootstrap_content = f.read()
        except FileNotFoundError:
            error("Bootstrap script not found", extra={"instance_id": self.instance_id, "script_path": bootstrap_script_path})
            return False

        if skip_gpu_setup:
            info("Modifying bootstrap script to skip GPU setup", extra={"instance_id": self.instance_id})

            bootstrap_content = self._modify_bootstrap_for_cpu_only(bootstrap_content)

        result = self.ssh_manager.execute_script(bootstrap_content, "bootstrap.sh")

        if not result.success:
            error(
                "Bootstrap script execution failed",
                extra={"instance_id": self.instance_id, "error": result.stderr, "stdout": result.stdout},
            )
            return False

        info("Bootstrap script completed successfully", extra={"instance_id": self.instance_id})
        return True

    def _modify_bootstrap_for_cpu_only(self, bootstrap_content: str) -> str:
        """Modify bootstrap script to skip GPU-specific components.

        Parameters
        ----------
        bootstrap_content : str
            Original bootstrap script content.

        Returns
        -------
        str
            Modified bootstrap script content.
        """
        modified_content = bootstrap_content.replace(
            'DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq -o Dpkg::Options::="--force-confnew" nvidia-docker2',
            "# Skipping NVIDIA Docker for CPU-only instance\nsudo apt-get update -qq\nsudo apt-get install -y -qq docker.io docker-compose",
        )

        modified_content = modified_content.replace(
            'echo \'{"default-runtime": "nvidia"}\' | sudo tee /etc/docker/daemon.json > /dev/null',
            "# Using default Docker runtime for CPU-only instance",
        )

        modified_content = modified_content.replace("bin/init.sh -a gpu -c local", "bin/init.sh -a cpu -c local")

        return modified_content

    def _configure_deepracer_environment(self, s3_bucket_name: Optional[str], aws_profile: str) -> bool:
        """Configure DeepRacer environment after bootstrap.

        Parameters
        ----------
        s3_bucket_name : str, optional
            S3 bucket name for DeepRacer.
        aws_profile : str, optional
            AWS profile to use.

        Returns
        -------
        bool
            True if configuration was successful.
        """
        info(
            "Configuring DeepRacer environment",
            extra={"instance_id": self.instance_id, "s3_bucket": s3_bucket_name, "aws_profile": aws_profile},
        )

        config_script = f"""
        cd ~/deepracer-for-cloud

        # Update system.env with custom settings
        """

        if s3_bucket_name:
            config_script += f"""
        # Configure S3 bucket
        sed -i 's/DR_LOCAL_S3_BUCKET=.*/DR_LOCAL_S3_BUCKET={s3_bucket_name}/' system.env
        """

        config_script += f"""
        # Configure AWS profile
        sed -i 's/DR_LOCAL_S3_PROFILE=.*/DR_LOCAL_S3_PROFILE={aws_profile}/' system.env
        sed -i 's/DR_UPLOAD_S3_PROFILE=.*/DR_UPLOAD_S3_PROFILE={aws_profile}/' system.env

        # Ensure all scripts are executable
        chmod +x bin/*.sh

        # Test AWS credentials
        aws sts get-caller-identity || echo "Warning: AWS credentials not configured"
        """

        result = self.ssh_manager.execute_script(config_script, "configure_deepracer_env.sh")

        if not result.success:
            error("DeepRacer environment configuration failed", extra={"instance_id": self.instance_id, "error": result.stderr})
            return False

        info("DeepRacer environment configured successfully", extra={"instance_id": self.instance_id})
        return True

    def verify_installation(self) -> bool:
        """Verify DeepRacer installation.

        Returns
        -------
        bool
            True if installation is verified.
        """
        info("Verifying DeepRacer installation", extra={"instance_id": self.instance_id})

        verification_script = """
        # Check if DeepRacer directory exists
        if [ ! -d ~/deepracer-for-cloud ]; then
            echo "ERROR: DeepRacer directory not found"
            exit 1
        fi

        # Check if Docker is working
        if ! docker ps >/dev/null 2>&1; then
            echo "ERROR: Docker is not working"
            exit 1
        fi

        # Check if pyenv and Python 3.12 are installed
        if ! command -v pyenv >/dev/null 2>&1; then
            echo "ERROR: pyenv not found"
            exit 1
        fi

        # Check Python version
        cd ~/deepracer-for-cloud
        source ~/.bashrc
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"

        PYTHON_VERSION=$(python --version 2>&1)
        if [[ ! "$PYTHON_VERSION" =~ "Python 3.12" ]]; then
            echo "ERROR: Python 3.12 not found. Current: $PYTHON_VERSION"
            exit 1
        fi

        # Check if DeepRacer is properly initialized
        if [ ! -f bin/activate.sh ]; then
            echo "ERROR: DeepRacer not properly initialized"
            exit 1
        fi

        # Test activation script
        source bin/activate.sh >/dev/null 2>&1 || {
            echo "ERROR: Failed to activate DeepRacer environment"
            exit 1
        }

        echo "SUCCESS: All verifications passed"
        """

        result = self.ssh_manager.execute_script(verification_script, "verify_installation.sh")

        if result.success and "SUCCESS" in result.stdout:
            info("DeepRacer installation verified successfully", extra={"instance_id": self.instance_id})
            return True
        else:
            error(
                "DeepRacer installation verification failed",
                extra={"instance_id": self.instance_id, "error": result.stderr, "output": result.stdout},
            )
            return False
