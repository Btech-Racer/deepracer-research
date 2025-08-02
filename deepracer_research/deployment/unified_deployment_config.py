from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig
from deepracer_research.deployment.deployment_target import DeploymentTarget
from deepracer_research.deployment.local.config.local_deployment_config import (
    LocalDeploymentConfig,
)
from deepracer_research.deployment.nvidia_brev.config.deepracer_config import NvidiaBrevDeepRacerConfig
from deepracer_research.deployment.sagemaker.sagemaker_deployment_config import (
    SageMakerDeploymentConfig,
)


@dataclass
class UnifiedDeploymentConfig:
    """Unified configuration that supports multiple deployment targets.

    Parameters
    ----------
    model_name : str
        Name for the model
    description : str, optional
        Model description, by default ""
    deployment_target : DeploymentTarget, optional
        Target for deployment, by default DeploymentTarget.LOCAL
    aws_config : Optional[SageMakerDeploymentConfig], optional
        AWS SageMaker deployment configuration, by default None
    local_config : Optional[LocalDeploymentConfig], optional
        Local training configuration, by default None
    deepracer_config : Optional[AWSDeepRacerConfig], optional
        AWS DeepRacer console configuration, by default None
    nvidia_brev_config : Optional[NvidiaBrevDeepRacerConfig], optional
        NVIDIA Brev deployment configuration, by default None
    """

    model_name: str
    description: str = ""
    deployment_target: DeploymentTarget = DeploymentTarget.LOCAL
    aws_config: Optional[SageMakerDeploymentConfig] = None
    local_config: Optional[LocalDeploymentConfig] = None
    deepracer_config: Optional[AWSDeepRacerConfig] = None
    nvidia_brev_config: Optional[NvidiaBrevDeepRacerConfig] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate that the configuration is consistent."""
        configs = [self.aws_config, self.local_config, self.deepracer_config, self.nvidia_brev_config]

        if not any(configs):
            raise ValueError("At least one target-specific configuration must be provided")

        target_config_map = {
            DeploymentTarget.AWS_SAGEMAKER: self.aws_config,
            DeploymentTarget.LOCAL: self.local_config,
            DeploymentTarget.AWS_DEEPRACER: self.deepracer_config,
            DeploymentTarget.NVIDIA_BREV: self.nvidia_brev_config,
        }

        if self.deployment_target != DeploymentTarget.UNIFIED:
            expected_config = target_config_map.get(self.deployment_target)
            if expected_config is None:
                raise ValueError(f"Configuration for {self.deployment_target.value} is required")

    def get_active_config(
        self,
    ) -> Union[SageMakerDeploymentConfig, LocalDeploymentConfig, AWSDeepRacerConfig, NvidiaBrevDeepRacerConfig]:
        """Get the active configuration based on deployment target.

        Returns
        -------
        Union[SageMakerDeploymentConfig, LocalDeploymentConfig, AWSDeepRacerConfig, NvidiaBrevDeepRacerConfig]
            The active configuration for the selected deployment target

        Raises
        ------
        ValueError
            If no configuration is available for the selected target
        """
        target_config_map = {
            DeploymentTarget.AWS_SAGEMAKER: self.aws_config,
            DeploymentTarget.LOCAL: self.local_config,
            DeploymentTarget.AWS_DEEPRACER: self.deepracer_config,
            DeploymentTarget.NVIDIA_BREV: self.nvidia_brev_config,
        }

        if self.deployment_target == DeploymentTarget.UNIFIED:
            for config in target_config_map.values():
                if config is not None:
                    return config
            raise ValueError("No configuration available for unified deployment")

        config = target_config_map.get(self.deployment_target)
        if config is None:
            raise ValueError(f"No configuration available for {self.deployment_target.value}")

        return config

    def switch_target(self, new_target: DeploymentTarget) -> None:
        """Switch to a different deployment target.

        Parameters
        ----------
        new_target : DeploymentTarget
            New deployment target to switch to

        Raises
        ------
        ValueError
            If no configuration is available for the new target
        """
        target_config_map = {
            DeploymentTarget.AWS_SAGEMAKER: self.aws_config,
            DeploymentTarget.LOCAL: self.local_config,
            DeploymentTarget.AWS_DEEPRACER: self.deepracer_config,
            DeploymentTarget.NVIDIA_BREV: self.nvidia_brev_config,
        }

        if new_target != DeploymentTarget.UNIFIED and target_config_map.get(new_target) is None:
            raise ValueError(f"No configuration available for {new_target.value}")

        self.deployment_target = new_target

    def get_available_targets(self) -> list[DeploymentTarget]:
        """Get list of available deployment targets based on configured options.

        Returns
        -------
        list[DeploymentTarget]
            List of available deployment targets
        """
        available = []

        if self.aws_config is not None:
            available.append(DeploymentTarget.AWS_SAGEMAKER)
        if self.local_config is not None:
            available.append(DeploymentTarget.LOCAL)
        if self.deepracer_config is not None:
            available.append(DeploymentTarget.AWS_DEEPRACER)
        if self.nvidia_brev_config is not None:
            available.append(DeploymentTarget.NVIDIA_BREV)

        if len(available) > 1:
            available.append(DeploymentTarget.UNIFIED)

        return available

    def get_cost_estimates(self, hours: int = 1) -> Dict[str, float]:
        """Get cost estimates for all configured deployment targets.

        Parameters
        ----------
        hours : int, optional
            Number of hours to estimate for, by default 1

        Returns
        -------
        Dict[str, float]
            Cost estimates by deployment target
        """
        estimates = {}

        if self.aws_config:
            estimates[DeploymentTarget.AWS_SAGEMAKER.value] = 5.0 * hours

        if self.local_config:
            estimates[DeploymentTarget.LOCAL.value] = 0.0

        if self.deepracer_config:
            estimates[DeploymentTarget.AWS_DEEPRACER.value] = 3.5 * hours

        if self.nvidia_brev_config:
            instance_estimate = self.nvidia_brev_config.instance_config.get_cost_estimate(hours)
            estimates[DeploymentTarget.NVIDIA_BREV.value] = instance_estimate["total_cost"]

        return estimates

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration
        """
        result = {
            "model_name": self.model_name,
            "description": self.description,
            "deployment_target": self.deployment_target.value,
            "available_targets": [t.value for t in self.get_available_targets()],
        }

        if self.aws_config:
            result["aws_config"] = self.aws_config.to_dict()
        if self.local_config:
            result["local_config"] = self.local_config.to_dict()
        if self.deepracer_config:
            result["deepracer_config"] = self.deepracer_config.to_dict()
        if self.nvidia_brev_config:
            result["nvidia_brev_config"] = self.nvidia_brev_config.to_dict()

        return result

    @classmethod
    def create_for_nvidia_brev(
        cls, model_name: str, api_token: str, s3_bucket: Optional[str] = None, description: str = "", **kwargs
    ) -> "UnifiedDeploymentConfig":
        """Create a unified configuration optimized for NVIDIA Brev.

        Parameters
        ----------
        model_name : str
            Name for the DeepRacer model
        api_token : str
            NVIDIA Brev API token
        s3_bucket : Optional[str], optional
            S3 bucket for model storage, by default None
        description : str, optional
            Model description, by default ""
        **kwargs
            Additional configuration parameters

        Returns
        -------
        UnifiedDeploymentConfig
            Unified configuration with NVIDIA Brev setup
        """
        nvidia_brev_config = NvidiaBrevDeepRacerConfig.create_quick_training(
            model_name=model_name, api_token=api_token, s3_bucket=s3_bucket, **kwargs
        )

        return cls(
            model_name=model_name,
            description=description,
            deployment_target=DeploymentTarget.NVIDIA_BREV,
            nvidia_brev_config=nvidia_brev_config,
        )

    @classmethod
    def create_multi_target(cls, model_name: str, description: str = "", **configs) -> "UnifiedDeploymentConfig":
        """Create a multi-target unified configuration.

        Parameters
        ----------
        model_name : str
            Name for the DeepRacer model
        description : str, optional
            Model description, by default ""
        **configs
            Target-specific configurations (aws_config, local_config, etc.)

        Returns
        -------
        UnifiedDeploymentConfig
            Multi-target unified configuration
        """
        return cls(model_name=model_name, description=description, deployment_target=DeploymentTarget.UNIFIED, **configs)
