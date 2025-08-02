from deepracer_research.deployment.nvidia_brev.config import InstanceConfig, NvidiaBrevConfig, NvidiaBrevDeepRacerConfig
from deepracer_research.deployment.nvidia_brev.enum import DeploymentMode, GPUType, InstanceTemplate


def create_quick_training_config(
    model_name: str, api_token: str, s3_bucket: str = None, gpu_type: GPUType = GPUType.A100, **kwargs
) -> NvidiaBrevDeepRacerConfig:
    """Create a quick training configuration with sensible defaults.

    This is a convenience function that creates a complete DeepRacer configuration
    optimized for training on NVIDIA Brev with minimal setup required.

    Parameters
    ----------
    model_name : str
        Name for the DeepRacer model
    api_token : str
        NVIDIA Brev API token
    s3_bucket : str, optional
        S3 bucket for model storage, by default None
    gpu_type : GPUType, optional
        GPU type for training, by default A100
    **kwargs
        Additional configuration parameters

    Returns
    -------
    NvidiaBrevDeepRacerConfig
        Complete configuration ready for deployment
    """
    return NvidiaBrevDeepRacerConfig.create_quick_training(
        model_name=model_name,
        track_arn="arn:aws:deepracer:us-east-1::track/reInvent2019_track",
        reward_function_code="def reward_function(params): return 1.0",
        api_token=api_token,
        s3_bucket=s3_bucket,
        gpu_type=gpu_type,
        **kwargs,
    )


def create_development_config(
    model_name: str, api_token: str, gpu_type: GPUType = GPUType.RTX_4080, **kwargs
) -> NvidiaBrevDeepRacerConfig:
    """Create a development configuration optimized for experimentation.

    This configuration is optimized for development and experimentation with
    lower costs and shorter auto-shutdown times.

    Parameters
    ----------
    model_name : str
        Name for the DeepRacer model
    api_token : str
        NVIDIA Brev API token
    gpu_type : GPUType, optional
        GPU type for development, by default RTX_4080
    **kwargs
        Additional configuration parameters

    Returns
    -------
    NvidiaBrevDeepRacerConfig
        Development-optimized configuration
    """
    brev_config = NvidiaBrevConfig(api_token=api_token)
    instance_config = InstanceConfig.for_development(gpu_type=gpu_type, **kwargs)

    from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig

    aws_config = AWSDeepRacerConfig(
        model_name=model_name,
        track_arn="arn:aws:deepracer:us-east-1::track/reInvent2019_track",
        reward_function_code="def reward_function(params): return 1.0",
    )

    return NvidiaBrevDeepRacerConfig.from_aws_deepracer_config(
        aws_deepracer_config=aws_config,
        api_token=api_token,
        gpu_type=gpu_type,
        deployment_mode=DeploymentMode.ON_DEMAND,
        auto_start_training=False,
        **kwargs,
    )


def get_available_gpu_types() -> list[GPUType]:
    """Get list of available GPU types for NVIDIA Brev.

    Returns
    -------
    list[GPUType]
        List of available GPU types
    """
    return list(GPUType)


def get_suitable_gpu_types_for_training() -> list[GPUType]:
    """Get GPU types suitable for DeepRacer training.

    Returns
    -------
    list[GPUType]
        List of GPU types with sufficient memory for training
    """
    return [gpu for gpu in GPUType if gpu.is_suitable_for_training]


def get_cost_estimate(gpu_type: GPUType, deployment_mode: DeploymentMode, hours: int = 1) -> dict[str, float]:
    """Get cost estimate for running a configuration.

    Parameters
    ----------
    gpu_type : GPUType
        GPU type to estimate for
    deployment_mode : DeploymentMode
        Deployment mode affecting pricing
    hours : int, optional
        Number of hours to run, by default 1

    Returns
    -------
    dict[str, float]
        Cost estimate breakdown
    """
    config = InstanceConfig(
        cpu_cores=gpu_type.get_recommended_cpu_cores(),
        template=InstanceTemplate.DEEPRACER_READY,
        gpu_type=gpu_type,
        deployment_mode=deployment_mode,
    )

    return config.get_cost_estimate(hours)
