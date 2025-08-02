from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig
from deepracer_research.deployment.nvidia_brev.config.instance_config import InstanceConfig
from deepracer_research.deployment.nvidia_brev.config.nvidia_brev_config import NvidiaBrevConfig
from deepracer_research.deployment.nvidia_brev.config.ssh_config import SSHConfig


@dataclass
class NvidiaBrevDeepRacerConfig:
    """NVIDIA Brev-specific configuration that extends AWS DeepRacer configuration

    Parameters
    ----------
    aws_deepracer_config : AWSDeepRacerConfig
        Base AWS DeepRacer configuration (model, reward function, etc.)
    brev_config : NvidiaBrevConfig
        NVIDIA Brev API configuration
    instance_config : InstanceConfig
        Instance hardware and software configuration
    ssh_config : SSHConfig, optional
        SSH connection configuration, by default None (auto-generated)
    project_name : str, optional
        Project name for organization, by default None
    aws_s3_bucket : str, optional
        S3 bucket for model storage, by default None
    aws_profile : str, optional
        AWS profile name for credentials, by default "default"
    deepracer_version : str, optional
        DeepRacer-for-Cloud version to install, by default "latest"
    auto_start_training : bool, optional
        Whether to automatically start training after setup, by default False
    backup_enabled : bool, optional
        Whether to enable automatic model backups, by default True
    monitoring_enabled : bool, optional
        Whether to enable training monitoring, by default True
    tags : Dict[str, str], optional
        Tags for resource organization, by default empty dict
    """

    aws_deepracer_config: AWSDeepRacerConfig
    brev_config: NvidiaBrevConfig
    instance_config: InstanceConfig
    ssh_config: Optional[SSHConfig] = None
    project_name: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    aws_profile: str = "default"
    deepracer_version: str = "latest"
    auto_start_training: bool = False
    backup_enabled: bool = True
    monitoring_enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.ssh_config is None:
            self.ssh_config = SSHConfig.for_development()

        if self.project_name is None:
            self.project_name = f"deepracer-{self.aws_deepracer_config.model_name}"
        default_tags = {
            "project": "deepracer",
            "model": self.aws_deepracer_config.model_name,
            "platform": "nvidia-brev",
            "created_by": "deepracer-research",
        }
        self.tags = {**default_tags, **self.tags}

        self.validate()

    @property
    def model_name(self) -> str:
        """Get the model name from the AWS DeepRacer config."""
        return self.aws_deepracer_config.model_name

    @property
    def description(self) -> str:
        """Get the model description from the AWS DeepRacer config."""
        return self.aws_deepracer_config.description

    @property
    def reward_function_code(self) -> str:
        """Get the reward function code from the AWS DeepRacer config."""
        return self.aws_deepracer_config.reward_function_code

    @property
    def track_arn(self) -> str:
        """Get the track ARN from the AWS DeepRacer config."""
        return self.aws_deepracer_config.track_arn

    def validate(self) -> None:
        """Validate the complete configuration.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        self.brev_config.validate()
        self.instance_config.validate()
        if self.ssh_config:
            self.ssh_config.validate()

        if self.aws_s3_bucket and not self.aws_s3_bucket.replace("-", "").replace(".", "").isalnum():
            raise ValueError("S3 bucket name contains invalid characters")

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration
        """
        config = {
            "model_name": self.model_name,
            "description": self.description,
            "project_name": self.project_name,
            "aws_deepracer_config": self.aws_deepracer_config.to_dict(),
            "brev_config": self.brev_config.to_dict(),
            "instance_config": self.instance_config.to_dict(),
            "aws_s3_bucket": self.aws_s3_bucket,
            "aws_profile": self.aws_profile,
            "deepracer_version": self.deepracer_version,
            "auto_start_training": self.auto_start_training,
            "backup_enabled": self.backup_enabled,
            "monitoring_enabled": self.monitoring_enabled,
            "tags": self.tags,
        }

        if self.ssh_config:
            config["ssh_config"] = self.ssh_config.to_dict()

        return config

    def get_deployment_name(self) -> str:
        """Get the deployment name for NVIDIA Brev.

        Returns
        -------
        str
            Deployment name
        """
        return f"{self.project_name}-{self.model_name}"

    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the instance.

        Returns
        -------
        Dict[str, str]
            Environment variables
        """
        dr_params = self.aws_deepracer_config.hyperparameters.to_dict()

        env_vars = {
            "DEEPRACER_MODEL_NAME": self.model_name,
            "DEEPRACER_PROJECT": self.project_name,
            "DEEPRACER_VERSION": self.deepracer_version,
            "AWS_DEFAULT_PROFILE": self.aws_profile,
            "DR_WORLD_NAME": (
                self.aws_deepracer_config.track_arn.split("/")[-1]
                if self.aws_deepracer_config.track_arn
                else "reInvent2019_track"
            ),
            "DR_RACE_TYPE": (
                self.aws_deepracer_config.experimental_scenario.to_deepracer_race_type()
                if hasattr(self.aws_deepracer_config, "experimental_scenario")
                else "TIME_TRIAL"
            ),
            "DR_TRAINING_ALGORITHM": self.aws_deepracer_config.training_algorithm.value.upper(),
            "DR_ACTION_SPACE_TYPE": self.aws_deepracer_config.action_space_type.value.upper(),
            "DR_SENSOR_TYPE": self.aws_deepracer_config.sensor_type.value,
        }

        if self.aws_s3_bucket:
            env_vars["DEEPRACER_S3_BUCKET"] = self.aws_s3_bucket
            env_vars["DR_LOCAL_S3_BUCKET"] = self.aws_s3_bucket
            env_vars["DR_LOCAL_S3_MODEL_PREFIX"] = self.model_name

        for key, value in dr_params.items():
            env_key = f"DR_{key.upper()}"
            env_vars[env_key] = str(value)

        env_vars.update(self.instance_config.environment_variables)

        return env_vars

    def get_deepracer_setup_script(self) -> str:
        """Generate DeepRacer-specific setup script.

        Returns
        -------
        str
            Setup script content
        """
        script_lines = [
            "#!/bin/bash",
            "set -e",
            "",
            f"# DeepRacer NVIDIA Brev Setup Script",
            f"# Model: {self.model_name}",
            f"# Project: {self.project_name}",
            "",
            "echo 'Starting DeepRacer setup on NVIDIA Brev...'",
            "",
            "# Update system",
            "sudo apt-get update -y",
            "",
            "# Install required packages",
            "sudo apt-get install -y curl wget git unzip awscli docker.io docker-compose",
            "",
            "# Add user to docker group",
            "sudo usermod -aG docker $USER",
            "",
            "# Install NVIDIA Container Toolkit",
            "distribution=$(. /etc/os-release;echo $ID$VERSION_ID)",
            "curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -",
            "curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list",
            "sudo apt-get update && sudo apt-get install -y nvidia-docker2",
            "sudo systemctl restart docker",
            "",
        ]

        if self.instance_config.install_deepracer_cloud:
            script_lines.extend(
                [
                    "# Install DeepRacer-for-Cloud",
                    "cd /home/ubuntu",
                    f"git clone https://github.com/aws-deepracer-community/deepracer-for-cloud.git",
                    "cd deepracer-for-cloud",
                    "",
                    "# Configure DeepRacer environment",
                    f"echo 'DR_LOCAL_S3_BUCKET={self.aws_s3_bucket or 'bucket'}' > .env",
                    f"echo 'DR_LOCAL_S3_MODEL_PREFIX={self.model_name}' >> .env",
                    f"echo 'DR_UPLOAD_S3_BUCKET={self.aws_s3_bucket or 'bucket'}' >> .env",
                    f"echo 'DR_UPLOAD_S3_PREFIX={self.model_name}' >> .env",
                    "",
                    "# Add training configuration",
                ]
            )

            env_vars = self.get_environment_variables()
            for key, value in env_vars.items():
                if key.startswith("DR_"):
                    script_lines.append(f"echo '{key}={value}' >> .env")

            script_lines.extend(
                [
                    "",
                    "# Create reward function",
                    "cat > custom_files/reward_function.py << 'EOF'",
                    self.reward_function_code,
                    "EOF",
                    "",
                    "# Initialize DeepRacer",
                    "./bin/init.sh",
                    "",
                ]
            )

        script_lines.extend(
            [
                "# Configure AWS credentials",
                "mkdir -p /home/ubuntu/.aws",
                f"echo '[{self.aws_profile}]' > /home/ubuntu/.aws/config",
                "echo 'region = us-east-1' >> /home/ubuntu/.aws/config",
                "echo 'output = json' >> /home/ubuntu/.aws/config",
                "",
                "# Note: AWS credentials should be set via environment variables:",
                "# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                "",
            ]
        )

        if self.instance_config.custom_setup_script:
            script_lines.extend(["# Custom setup script", self.instance_config.custom_setup_script, ""])

        if self.auto_start_training:
            script_lines.extend(
                [
                    "# Auto-start training",
                    "cd /home/ubuntu/deepracer-for-cloud",
                    "echo 'Starting training automatically...'",
                    "dr-start-training &",
                    "",
                ]
            )

        script_lines.extend(
            [
                "echo 'DeepRacer setup completed!'",
                "echo 'Access training at: http://localhost:8080'",
                "echo 'Access Jupyter at: http://localhost:8888'",
                "echo 'Access TensorBoard at: http://localhost:6006'",
                "echo 'SSH into the instance to monitor training progress'",
                "",
            ]
        )

        return "\n".join(script_lines)

    @classmethod
    def from_aws_deepracer_config(
        cls, aws_deepracer_config: AWSDeepRacerConfig, api_token: str, s3_bucket: Optional[str] = None, gpu_type=None, **kwargs
    ) -> "NvidiaBrevDeepRacerConfig":
        """Create NVIDIA Brev config from existing AWS DeepRacer config.

        Parameters
        ----------
        aws_deepracer_config : AWSDeepRacerConfig
            Existing AWS DeepRacer configuration
        api_token : str
            NVIDIA Brev API token
        s3_bucket : Optional[str], optional
            S3 bucket for model storage, by default None
        gpu_type : GPUType, optional
            GPU type for training, by default A100
        **kwargs
            Additional configuration parameters

        Returns
        -------
        NvidiaBrevDeepRacerConfig
            NVIDIA Brev configuration based on AWS config
        """
        from deepracer_research.deployment.nvidia_brev.enum.deployment_mode import DeploymentMode
        from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType

        if gpu_type is None:
            gpu_type = GPUType.A100

        brev_config = NvidiaBrevConfig(api_token=api_token)
        instance_config = InstanceConfig.for_deepracer_training(
            gpu_type=gpu_type, deployment_mode=DeploymentMode.SPOT, **kwargs
        )

        return cls(
            aws_deepracer_config=aws_deepracer_config,
            brev_config=brev_config,
            instance_config=instance_config,
            aws_s3_bucket=s3_bucket,
            auto_start_training=True,
            **kwargs,
        )

    @classmethod
    def create_quick_training(
        cls,
        model_name: str,
        track_arn: str,
        reward_function_code: str,
        api_token: str,
        s3_bucket: Optional[str] = None,
        **kwargs,
    ) -> "NvidiaBrevDeepRacerConfig":
        """Create a quick training configuration with minimal setup.

        Parameters
        ----------
        model_name : str
            Name for the DeepRacer model
        track_arn : str
            ARN of the track to train on
        reward_function_code : str
            Python code for the reward function
        api_token : str
            NVIDIA Brev API token
        s3_bucket : Optional[str], optional
            S3 bucket for model storage, by default None
        **kwargs
            Additional configuration parameters

        Returns
        -------
        NvidiaBrevDeepRacerConfig
            Quick training configuration
        """
        aws_config = AWSDeepRacerConfig(
            model_name=model_name, track_arn=track_arn, reward_function_code=reward_function_code, **kwargs
        )

        return cls.from_aws_deepracer_config(
            aws_deepracer_config=aws_config, api_token=api_token, s3_bucket=s3_bucket, **kwargs
        )


DeepRacerBrevConfig = NvidiaBrevDeepRacerConfig
