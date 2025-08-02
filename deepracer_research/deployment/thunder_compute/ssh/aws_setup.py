import os
import subprocess
from pathlib import Path

from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion
from deepracer_research.deployment.thunder_compute.ssh.manager import SSHManager
from deepracer_research.utils import error, info, warning


class AWSSetupManager:
    """Manager for AWS setup on Thunder Compute instances"""

    def __init__(self, ssh_manager: SSHManager):
        """Initialize AWS setup manager.

        Parameters
        ----------
        ssh_manager : SSHManager
            SSH manager for the Thunder instance.
        """
        self.ssh_manager = ssh_manager

    def create_s3_bucket_if_not_exists(self, bucket_name: str, region: str = "AWSRegion.US_EAST_1") -> bool:
        """Create S3 bucket if it doesn't exist.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket to create.
        region : str, optional
            AWS region for the bucket, by default "AWSRegion.US_EAST_1".

        Returns
        -------
        bool
            True if bucket exists or was created successfully.
        """
        try:
            check_cmd = ["aws", "s3api", "head-bucket", "--bucket", bucket_name]
            result = subprocess.run(check_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                info(f"S3 bucket {bucket_name} already exists")
                return True

            info(f"Creating S3 bucket: {bucket_name}")
            if region == "AWSRegion.US_EAST_1":
                create_cmd = ["aws", "s3api", "create-bucket", "--bucket", bucket_name]
            else:
                create_cmd = [
                    "aws",
                    "s3api",
                    "create-bucket",
                    "--bucket",
                    bucket_name,
                    "--region",
                    region,
                    "--create-bucket-configuration",
                    f"LocationConstraint={region}",
                ]

            result = subprocess.run(create_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                info(f"S3 bucket {bucket_name} created successfully")

                self._set_deepracer_bucket_policy(bucket_name)

                return True
            else:
                error(f"Failed to create S3 bucket: {result.stderr}")
                return False

        except Exception as e:
            error(f"Error managing S3 bucket: {e}")
            return False

    def _set_deepracer_bucket_policy(self, bucket_name: str) -> None:
        """Set appropriate bucket policy for DeepRacer.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket.
        """
        try:
            versioning_cmd = [
                "aws",
                "s3api",
                "put-bucket-versioning",
                "--bucket",
                bucket_name,
                "--versioning-configuration",
                "Status=Enabled",
            ]

            subprocess.run(versioning_cmd, capture_output=True, text=True)
            info(f"Enabled versioning for bucket {bucket_name}")

        except Exception as e:
            warning(f"Could not configure bucket policy: {e}")

    def copy_aws_credentials(self) -> bool:
        """Copy local AWS credentials to Thunder instance.

        Returns
        -------
        bool
            True if credentials were copied successfully.
        """
        try:
            local_aws_dir = Path.home() / ".aws"

            if not local_aws_dir.exists():
                warning("No local ~/.aws directory found")
                return False

            info("Copying AWS credentials to Thunder instance...")

            mkdir_cmd = "mkdir -p ~/.aws"
            result = self.ssh_manager.execute_command(mkdir_cmd)

            if not result.success:
                error("Failed to create .aws directory on remote instance")
                return False

            credentials_file = local_aws_dir / "credentials"
            if credentials_file.exists():
                success = self.ssh_manager.upload_file(str(credentials_file), "~/.aws/credentials")
                if success:
                    info("AWS credentials copied")
                else:
                    error("Failed to copy AWS credentials")
                    return False

            config_file = local_aws_dir / "config"
            if config_file.exists():
                success = self.ssh_manager.upload_file(str(config_file), "~/.aws/config")
                if success:
                    info("AWS config copied")
                else:
                    warning("Failed to copy AWS config (non-critical)")

            chmod_cmd = "chmod 600 ~/.aws/credentials ~/.aws/config 2>/dev/null || true"
            self.ssh_manager.execute_command(chmod_cmd)

            verify_cmd = "aws sts get-caller-identity"
            result = self.ssh_manager.execute_command(verify_cmd)

            if result.success:
                info("AWS credentials verified on remote instance")
                return True
            else:
                warning("AWS credentials copied but verification failed")
                return True

        except Exception as e:
            error(f"Failed to copy AWS credentials: {e}")
            return False

    def setup_deepracer_environment(self, s3_bucket: str) -> bool:
        """Setup complete DeepRacer environment with AWS integration.

        Parameters
        ----------
        s3_bucket : str
            S3 bucket name for DeepRacer models.

        Returns
        -------
        bool
            True if setup was successful.
        """
        try:
            info("Setting up DeepRacer environment...")

            if not self.copy_aws_credentials():
                warning("AWS credentials setup failed - continuing anyway")

            env_vars = {
                "DR_CLOUD": "local",
                "DR_LOCAL_S3_PROFILE": "default",
                "DR_LOCAL_S3_BUCKET": s3_bucket,
                "DR_UPLOAD_S3_PROFILE": "default",
                "DR_UPLOAD_S3_BUCKET": s3_bucket,
                "AWS_DEFAULT_REGION": "AWSRegion.US_EAST_1",
            }

            env_content = "#!/bin/bash\n"
            for key, value in env_vars.items():
                env_content += f'export {key}="{value}"\n'

            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
                f.write(env_content)
                temp_path = f.name

            try:
                success = self.ssh_manager.upload_file(temp_path, "~/deepracer_env.sh")
                if success:
                    bashrc_cmd = 'echo "source ~/deepracer_env.sh" >> ~/.bashrc'
                    self.ssh_manager.execute_command(bashrc_cmd)
                    info("DeepRacer environment configured")
                    return True
                else:
                    error("Failed to upload environment file")
                    return False
            finally:
                os.unlink(temp_path)

        except Exception as e:
            error(f"Failed to setup DeepRacer environment: {e}")
            return False


def create_s3_bucket_locally(bucket_name: str, region: str = AWSRegion.US_EAST_1) -> bool:
    """Create S3 bucket locally before instance creation.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to create.
    region : str, optional
        AWS region for the bucket, by default "AWSRegion.US_EAST_1".

    Returns
    -------
    bool
        True if bucket exists or was created successfully.
    """
    aws_setup = AWSSetupManager(None)
    return aws_setup.create_s3_bucket_if_not_exists(bucket_name, region)
