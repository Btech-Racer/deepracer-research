from enum import Enum, unique
from typing import Any, Dict


@unique
class DeploymentTarget(Enum):
    """Unique deployment target enumeration.

    Attributes
    ----------
    LOCAL : str
        Local training environment target
    AWS_DEEPRACER : str
        AWS DeepRacer console training target
    AWS_SAGEMAKER : str
        AWS SageMaker custom training target
    NVIDIA_BREV : str
        NVIDIA Brev cloud GPU training target
    UNIFIED : str
        Multi-target unified deployment
    """

    LOCAL = "local"
    AWS_EC2 = "aws_ec2"
    AWS = "aws"
    AWS_DEEPRACER = "aws_deepracer"
    AWS_SAGEMAKER = "aws_sagemaker"
    NVIDIA_BREV = "nvidia_brev"
    UNIFIED = "unified"

    def get_config_class(self) -> str:
        """Get the corresponding configuration class name for this target.

        Returns
        -------
        str
            The configuration class name for this deployment target
        """
        config_classes = {
            self.LOCAL: "LocalDeploymentConfig",
            self.AWS_DEEPRACER: "AWSDeepRacerConfig",
            self.AWS_SAGEMAKER: "AWSSageMakerConfig",
            self.NVIDIA_BREV: "DeepRacerBrevConfig",
            self.UNIFIED: "UnifiedDeploymentConfig",
        }
        return config_classes[self]

    def get_requirements(self) -> Dict[str, Any]:
        """Get requirements for this deployment target.

        Returns
        -------
        Dict[str, Any]
            Requirements dictionary containing necessary information
        """
        requirements = {
            self.LOCAL: {
                "dependencies": ["docker", "nvidia-docker2"],
                "gpu_required": True,
                "min_gpu_memory_gb": 8,
                "min_disk_space_gb": 50,
                "supported_os": ["linux"],
                "estimated_setup_time_minutes": 30,
            },
            self.AWS_DEEPRACER: {
                "aws_account": True,
                "iam_role": "AWSDeepRacerServiceRole",
                "s3_bucket": True,
                "estimated_cost_per_hour": 3.50,
                "max_training_hours": 24,
                "supported_regions": ["us-east-1", "us-west-2", "eu-west-1"],
            },
            self.AWS_SAGEMAKER: {
                "aws_account": True,
                "iam_role": "SageMakerExecutionRole",
                "s3_bucket": True,
                "custom_algorithm": True,
                "estimated_cost_per_hour": 5.00,
                "supported_instances": ["ml.p3.2xlarge", "ml.g4dn.xlarge"],
            },
            self.NVIDIA_BREV: {
                "brev_account": True,
                "api_token": True,
                "gpu_required": True,
                "min_gpu_memory_gb": 16,
                "estimated_cost_per_hour": 2.40,
                "supported_gpus": ["A100", "H100", "RTX_4090", "A10G"],
                "auto_shutdown": True,
                "ssh_access": True,
            },
            self.UNIFIED: {"multiple_targets": True, "configuration_complexity": "high", "fallback_support": True},
        }
        return requirements[self]

    def is_cloud_target(self) -> bool:
        """Check if this is a cloud-based deployment target.

        Returns
        -------
        bool
            True if target is cloud-based
        """
        cloud_targets = {self.AWS_DEEPRACER, self.AWS_SAGEMAKER, self.NVIDIA_BREV}
        return self in cloud_targets

    def requires_aws_credentials(self) -> bool:
        """Check if this target requires AWS credentials.

        Returns
        -------
        bool
            True if AWS credentials are required
        """
        aws_targets = {self.AWS_DEEPRACER, self.AWS_SAGEMAKER}
        return self in aws_targets

    def supports_gpu_selection(self) -> bool:
        """Check if this target supports GPU type selection.

        Returns
        -------
        bool
            True if GPU selection is supported
        """
        gpu_selection_targets = {self.AWS_SAGEMAKER, self.NVIDIA_BREV, self.LOCAL}
        return self in gpu_selection_targets

    def supports_spot_instances(self) -> bool:
        """Check if this target supports spot/preemptible instances.

        Returns
        -------
        bool
            True if spot instances are supported
        """
        spot_targets = {self.AWS_SAGEMAKER, self.NVIDIA_BREV}
        return self in spot_targets

    def get_typical_cost_range(self) -> Dict[str, float]:
        """Get typical cost range for this deployment target.

        Returns
        -------
        Dict[str, float]
            Cost range with min and max hourly costs in USD
        """
        cost_ranges = {
            self.LOCAL: {"min": 0.0, "max": 0.0},
            self.AWS_DEEPRACER: {"min": 2.50, "max": 4.00},
            self.AWS_SAGEMAKER: {"min": 3.00, "max": 8.00},
            self.NVIDIA_BREV: {"min": 0.50, "max": 4.00},
            self.UNIFIED: {"min": 0.0, "max": 8.00},
        }
        return cost_ranges[self]

    def get_setup_complexity(self) -> str:
        """Get setup complexity level for this target.

        Returns
        -------
        str
            Complexity level: 'low', 'medium', 'high'
        """
        complexity_levels = {
            self.LOCAL: "high",
            self.AWS_DEEPRACER: "low",
            self.AWS_SAGEMAKER: "medium",
            self.NVIDIA_BREV: "low",
            self.UNIFIED: "high",
        }
        return complexity_levels[self]

    @classmethod
    def get_recommended_for_beginners(cls) -> "DeploymentTarget":
        """Get recommended deployment target for beginners.

        Returns
        -------
        DeploymentTarget
            Recommended target for beginners
        """
        return cls.NVIDIA_BREV

    @classmethod
    def get_most_cost_effective(cls) -> "DeploymentTarget":
        """Get most cost-effective deployment target.

        Returns
        -------
        DeploymentTarget
            Most cost-effective target
        """
        return cls.NVIDIA_BREV

    @classmethod
    def from_string(cls, target_str: str) -> "DeploymentTarget":
        """Create a DeploymentTarget from a string value.

        Parameters
        ----------
        target_str : str
            The string representation of the target

        Returns
        -------
        DeploymentTarget
            The corresponding DeploymentTarget enum value

        Raises
        ------
        ValueError
            If the string doesn't match any target
        """
        try:
            return cls(target_str.lower())
        except ValueError:
            valid_targets = [target.value for target in cls]
            raise ValueError(f"Invalid deployment target '{target_str}'. " f"Valid targets are: {valid_targets}")
