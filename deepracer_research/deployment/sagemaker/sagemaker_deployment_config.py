from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from deepracer_research.config.aws.aws_hyperparameters import DEFAULT_HYPERPARAMETERS, AWSHyperparameters
from deepracer_research.deployment.sagemaker.instance_type import SageMakerInstanceType
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.rewards.builder import RewardFunctionBuilder


@dataclass
class SageMakerDeploymentConfig:
    """Configuration for model deployment to AWS SageMaker.

    Parameters
    ----------
    model_name : str
        Name for the deployed model
    role_arn : str
        AWS IAM role ARN for training
    description : str, optional
        Model description, by default ""
    training_job_name : str, optional
        SageMaker training job name, by default ""
    algorithm_specification : Dict[str, Any], optional
        Algorithm container configuration, by default predefined config
    hyperparameters : AWSHyperparameters, optional
        Training hyperparameters, by default DEFAULT_HYPERPARAMETERS
    instance_type : SageMakerInstanceType, optional
        SageMaker instance type for training, by default SageMakerInstanceType.ML_C5_2XLARGE
    instance_count : int, optional
        Number of training instances, by default 1
    volume_size_gb : int, optional
        Storage volume size, by default 30
    max_runtime_hours : int, optional
        Maximum training runtime, by default 24
    s3_bucket : str, optional
        S3 bucket for model storage, by default ""
    s3_prefix : str, optional
        S3 prefix for organization, by default ""
    output_s3_key : str, optional
        S3 key for training output, by default ""
    reward_scenario : Optional[ExperimentalScenario], optional
        Experimental scenario for reward function, by default None
    reward_kwargs : Dict[str, Any], optional
        Additional arguments for reward function, by default empty dict
    """

    model_name: str
    role_arn: str
    description: str = ""

    reward_scenario: Optional[ExperimentalScenario] = None
    reward_kwargs: Dict[str, Any] = field(default_factory=dict)

    training_job_name: str = ""
    algorithm_specification: Dict[str, Any] = field(
        default_factory=lambda: {
            "TrainingImageUri": "382416733822.dkr.ecr.us-east-1.amazonaws.com/sagemaker-rl-tensorflow:coach0.11.0-tf-1.13.1-cpu-py3",
            "TrainingInputMode": "File",
        }
    )
    hyperparameters: AWSHyperparameters = field(default_factory=lambda: DEFAULT_HYPERPARAMETERS)

    instance_type: SageMakerInstanceType = SageMakerInstanceType.ML_C5_2XLARGE
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_hours: int = 24

    s3_bucket: str = ""
    s3_prefix: str = ""
    output_s3_key: str = ""

    def validate_configuration(self) -> bool:
        """Validate the deployment configuration.

        Returns
        -------
        bool
            True if configuration is valid, False otherwise
        """
        required_fields = [self.model_name, self.role_arn]
        return all(field.strip() for field in required_fields)

    def get_s3_output_path(self) -> str:
        """Get the complete S3 output path.

        Returns
        -------
        str
            Complete S3 path for model output
        """
        if not self.s3_bucket:
            return ""

        parts = [f"s3://{self.s3_bucket}"]
        if self.s3_prefix:
            parts.append(self.s3_prefix.strip("/"))
        if self.output_s3_key:
            parts.append(self.output_s3_key.strip("/"))

        return "/".join(parts)

    def update_hyperparameter(self, key: str, value: Union[str, int, float]) -> None:
        """Update a specific hyperparameter.

        Parameters
        ----------
        key : str
            Hyperparameter key to update
        value : Union[str, int, float]
            New value for the hyperparameter

        Returns
        -------
        None
        """
        if hasattr(self.hyperparameters, key):
            setattr(self.hyperparameters, key, value)
        else:
            raise ValueError(f"Hyperparameter '{key}' not found in AWSHyperparameters")

    def get_hyperparameters_dict(self) -> Dict[str, Union[int, float, str]]:
        """Get hyperparameters as a dictionary for AWS API compatibility.

        Returns
        -------
        Dict[str, Union[int, float, str]]
            Hyperparameters as a dictionary with string values for AWS APIs
        """
        return {key: str(value) for key, value in self.hyperparameters.to_dict().items()}

    def get_resource_config(self) -> Dict[str, Any]:
        """Get the resource configuration for training.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing instance_type, instance_count,
            volume_size_gb, and max_runtime_hours
        """
        return {
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "volume_size_gb": self.volume_size_gb,
            "max_runtime_hours": self.max_runtime_hours,
        }

    def get_reward_function_code(self) -> str:
        """Get the reward function code using the RewardFunctionBuilder.

        Returns
        -------
        str
            The complete reward function code ready for AWS deployment
        """
        if self.reward_function_code:
            return self.reward_function_code

        if self.scenario:
            builder = RewardFunctionBuilder.create_for_scenario(self.scenario, **self.reward_parameters)

            builder = builder.with_optimization("advanced")

            return builder.build_function_code()

        if self.reward_function:
            return self.reward_function.get_function_code()

        return self._get_default_reward_function_code()

    def set_reward_scenario(self, scenario: ExperimentalScenario, **kwargs) -> None:
        """Set the reward scenario for AWS deployment.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The experimental scenario for reward function
        **kwargs
            Additional arguments for the reward function

        Returns
        -------
        None
        """
        self.reward_scenario = scenario
        self.reward_kwargs = kwargs

    def to_local_config(self):
        """Convert AWS SageMaker config to local deployment config.

        Returns
        -------
        LocalDeploymentConfig
            Equivalent local deployment configuration
        """
        from deepracer_research.deployment.local import LocalComputeDevice, LocalDeploymentConfig, LocalTrainingBackend

        device = LocalComputeDevice.CPU

        if "p3" in self.instance_type.value or "g4dn" in self.instance_type.value:
            device = LocalComputeDevice.GPU

        local_config = LocalDeploymentConfig(
            model_name=self.model_name,
            description=self.description,
            reward_scenario=self.reward_scenario,
            reward_kwargs=self.reward_kwargs.copy(),
            hyperparameters=self.hyperparameters,
            backend=LocalTrainingBackend.STABLE_BASELINES3,
            device=device,
            enable_gpu=(device == LocalComputeDevice.GPU),
        )

        return local_config


def create_aws_sagemaker_config(
    model_name: str,
    role_arn: str,
    description: str = "",
    reward_scenario: Optional[ExperimentalScenario] = None,
    s3_bucket: str = "",
    instance_type: SageMakerInstanceType = SageMakerInstanceType.ML_C5_2XLARGE,
    **kwargs,
) -> SageMakerDeploymentConfig:
    """Create an AWS SageMaker deployment configuration with sensible defaults.

    Parameters
    ----------
    model_name : str
        Name for the deployed model
    role_arn : str
        AWS IAM role ARN for training
    description : str, optional
        Model description, by default ""
    reward_scenario : Optional[ExperimentalScenario], optional
        Experimental scenario for reward function, by default None
    s3_bucket : str, optional
        S3 bucket for model storage, by default ""
    instance_type : SageMakerInstanceType, optional
        SageMaker instance type, by default SageMakerInstanceType.ML_C5_2XLARGE
    **kwargs
        Additional configuration options

    Returns
    -------
    SageMakerDeploymentConfig
        Configured AWS SageMaker deployment configuration
    """
    config = SageMakerDeploymentConfig(
        model_name=model_name,
        role_arn=role_arn,
        description=description,
        s3_bucket=s3_bucket,
        instance_type=instance_type,
        **kwargs,
    )

    if reward_scenario:
        config.set_reward_scenario(reward_scenario)

    return config


AWSDeploymentConfig = SageMakerDeploymentConfig
create_aws_deployment_config = create_aws_sagemaker_config
