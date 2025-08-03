from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepracer_research.config.aws.types.action_space_type import ActionSpaceType
from deepracer_research.config.aws.types.sensor_type import SensorType
from deepracer_research.config.track.track_type import TrackType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.deployment.deepracer.hyperparameters import DeepRacerHyperparameters
from deepracer_research.deployment.local.enum.local_compute_device import LocalComputeDevice
from deepracer_research.deployment.local.enum.local_training_backend import LocalTrainingBackend
from deepracer_research.experiments import ExperimentalScenario
from deepracer_research.utils.logger import debug, error, info, warning


@dataclass
class LocalDeploymentConfig:
    """Configuration for local DeepRacer model training using DeepRacer for Cloud

    Parameters
    ----------
    model_name : str
        Name for the DeepRacer model
    track_name : TrackType
        Track to train on
    reward_function_code : str
        Python code for the reward function
    output_directory : Path, optional
        Directory to save training outputs, by default ./models
    action_space_type : ActionSpaceType, optional
        Type of action space ('discrete' or 'continuous'), by default 'discrete'
    training_algorithm : TrainingAlgorithm, optional
        Training algorithm to use, by default PPO
    hyperparameters : DeepRacerHyperparameters, optional
        Training hyperparameters, by default DeepRacerHyperparameters()
    max_job_duration_seconds : int, optional
        Maximum training duration in seconds, by default 7200 (2 hours)
    sensor_type : SensorType, optional
        Sensor configuration, by default FRONT_FACING_CAMERA
    experimental_scenario : ExperimentalScenario, optional
        Experimental scenario for training, by default TIME_TRIAL
    backend : LocalTrainingBackend, optional
        Local training backend, by default DEEPRACER_FOR_CLOUD
    device : LocalComputeDevice, optional
        Compute device to use, by default AUTO
    """

    model_name: str
    track_name: TrackType
    reward_function_code: str
    output_directory: Path = field(default_factory=lambda: Path("./models"))

    action_space_type: ActionSpaceType = ActionSpaceType.DISCRETE
    steering_angle_granularity: int = 5
    max_steering_angle: float = 30.0
    speed_granularity: int = 3
    max_speed: float = 4.0
    min_speed: float = 1.0

    training_algorithm: TrainingAlgorithm = TrainingAlgorithm.PPO
    hyperparameters: DeepRacerHyperparameters = field(default_factory=DeepRacerHyperparameters)
    max_job_duration_seconds: int = 7200

    sensor_type: SensorType = SensorType.FRONT_FACING_CAMERA
    experimental_scenario: ExperimentalScenario = ExperimentalScenario.TIME_TRIAL

    backend: LocalTrainingBackend = LocalTrainingBackend.DEEPRACER_FOR_CLOUD
    device: LocalComputeDevice = LocalComputeDevice.AUTO
    num_workers: int = 4

    docker_compose_file: Optional[Path] = None
    use_gpu: bool = True
    memory_limit_gb: Optional[float] = 8.0
    cpu_limit: Optional[int] = None

    use_tensorboard: bool = True
    tensorboard_log_dir: Optional[Path] = None
    checkpoint_frequency: int = 1000
    log_frequency: int = 100

    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    reward_threshold: Optional[float] = None

    description: str = ""
    version: str = "1.0.0"
    tags: Dict[str, str] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing with validation and DeepRacer for Cloud setup."""
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if self.use_tensorboard and self.tensorboard_log_dir is None:
            self.tensorboard_log_dir = self.output_directory / "tensorboard"
            self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        if self.device == LocalComputeDevice.GPU and not self._is_gpu_available():
            warning("GPU not available, falling back to CPU")
            self.device = LocalComputeDevice.CPU
            self.use_gpu = False

        if self.action_space_type == ActionSpaceType.DISCRETE:
            if self.steering_angle_granularity < 3:
                warning("Steering angle granularity too low for discrete action space, setting to 3")
                self.steering_angle_granularity = 3

        if self.backend == LocalTrainingBackend.DEEPRACER_FOR_CLOUD:
            self._setup_deepracer_for_cloud()

    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for training."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf

                return len(tf.config.list_physical_devices("GPU")) > 0
            except ImportError:
                return False

    def _setup_deepracer_for_cloud(self):
        """Setup DeepRacer for Cloud specific configuration."""
        if self.docker_compose_file is None:
            self.docker_compose_file = Path("./docker-compose.yml")

        (self.output_directory / "logs").mkdir(parents=True, exist_ok=True)
        (self.output_directory / "models").mkdir(parents=True, exist_ok=True)
        (self.output_directory / "metrics").mkdir(parents=True, exist_ok=True)

    def get_action_space_config(self) -> Dict[str, Any]:
        """Get action space configuration compatible with DeepRacer for Cloud.

        Returns
        -------
        Dict[str, Any]
            Action space configuration for DeepRacer training
        """
        if self.action_space_type == ActionSpaceType.DISCRETE:
            return {
                "action_space_type": "discrete",
                "steering_angle_granularity": self.steering_angle_granularity,
                "max_steering_angle": self.max_steering_angle,
                "speed_granularity": self.speed_granularity,
                "max_speed": self.max_speed,
                "min_speed": self.min_speed,
            }
        else:
            return {
                "action_space_type": "continuous",
                "max_steering_angle": self.max_steering_angle,
                "max_speed": self.max_speed,
                "min_speed": self.min_speed,
            }

    def get_hyperparameters_config(self) -> Dict[str, str]:
        """Get hyperparameters configuration in DeepRacer format.

        Uses the same format as DeepRacerHyperparameters.to_deepracer_format()
        to ensure consistency with AWS DeepRacer API.

        Returns
        -------
        Dict[str, str]
            Hyperparameters configuration with all values as strings
        """
        return self.hyperparameters.to_deepracer_format()

    def get_deepracer_config(self) -> Dict[str, Any]:
        """Get complete DeepRacer configuration compatible with DeepRacer for Cloud.

        Returns
        -------
        Dict[str, Any]
            Complete DeepRacer training configuration
        """
        config = {
            "job_type": "training",
            "model_name": self.model_name,
            "max_job_duration_seconds": self.max_job_duration_seconds,
            "training_algorithm": self.training_algorithm.value,
            "track_name": self.track_name.value,
            "sensor": self.sensor_type.value,
            "scenario": self.experimental_scenario.value,
            "action_space": self.get_action_space_config(),
            "hyperparameters": self.get_hyperparameters_config(),
            "reward_function": {"code": self.reward_function_code},
            "output_directory": str(self.output_directory),
            "use_gpu": self.use_gpu,
            "memory_limit_gb": self.memory_limit_gb,
            "logging": {
                "tensorboard": self.use_tensorboard,
                "tensorboard_log_dir": str(self.tensorboard_log_dir) if self.tensorboard_log_dir else None,
                "checkpoint_frequency": self.checkpoint_frequency,
                "log_frequency": self.log_frequency,
            },
            "training_limits": {
                "max_episodes": self.max_episodes,
                "max_steps_per_episode": self.max_steps_per_episode,
                "reward_threshold": self.reward_threshold,
            },
            "metadata": {"description": self.description, "version": self.version, "tags": self.tags},
        }

        if self.cpu_limit is not None:
            config["cpu_limit"] = self.cpu_limit

        if self.backend == LocalTrainingBackend.DEEPRACER_FOR_CLOUD:
            config["docker"] = {
                "compose_file": str(self.docker_compose_file) if self.docker_compose_file else None,
                "num_workers": self.num_workers,
            }

        if self.custom_config:
            config.update(self.custom_config)

        return config

    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits for local training.

        Returns
        -------
        Dict[str, Any]
            Resource limits configuration
        """
        limits = {"device": self.device.value, "num_workers": self.num_workers, "use_gpu": self.use_gpu}

        if self.memory_limit_gb:
            limits["memory_gb"] = self.memory_limit_gb

        if self.cpu_limit:
            limits["cpu_count"] = self.cpu_limit

        return limits

    def validate_configuration(self) -> List[str]:
        """Validate the local deployment configuration.

        Returns
        -------
        List[str]
            List of validation errors, empty if configuration is valid
        """
        errors = []

        if not self.model_name.strip():
            errors.append("Model name cannot be empty")

        if not self.reward_function_code.strip():
            errors.append("Reward function code cannot be empty")

        if self.action_space_type == ActionSpaceType.DISCRETE:
            if self.steering_angle_granularity < 3:
                errors.append("Steering angle granularity must be at least 3 for discrete action space")
            if self.speed_granularity < 1:
                errors.append("Speed granularity must be at least 1")

        if self.max_steering_angle <= 0:
            errors.append("Maximum steering angle must be positive")

        if self.max_speed <= self.min_speed:
            errors.append("Maximum speed must be greater than minimum speed")

        if self.min_speed <= 0:
            errors.append("Minimum speed must be positive")

        if self.max_job_duration_seconds <= 0:
            errors.append("Maximum job duration must be positive")

        if self.max_episodes <= 0:
            errors.append("Maximum episodes must be positive")

        if self.max_steps_per_episode <= 0:
            errors.append("Maximum steps per episode must be positive")

        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            errors.append("Memory limit must be positive if specified")

        if self.cpu_limit is not None and self.cpu_limit <= 0:
            errors.append("CPU limit must be positive if specified")

        if self.backend == LocalTrainingBackend.DEEPRACER_FOR_CLOUD:
            if self.docker_compose_file and not self.docker_compose_file.exists():
                errors.append(f"Docker compose file does not exist: {self.docker_compose_file}")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns
        -------
        bool
            True if configuration is valid, False otherwise
        """
        return len(self.validate_configuration()) == 0

    def validate_deepracer_compatibility(self) -> List[str]:
        """Validate configuration for DeepRacer for Cloud compatibility.

        Returns
        -------
        List[str]
            List of compatibility issues, empty if fully compatible
        """
        issues = []

        try:
            hyperparams = self.get_hyperparameters_config()
            if not isinstance(hyperparams, dict):
                issues.append("Hyperparameters must be in dictionary format")
            else:
                for key, value in hyperparams.items():
                    if not isinstance(value, str):
                        issues.append(f"Hyperparameter '{key}' must be string, got {type(value)}")
        except Exception as e:
            issues.append(f"Failed to get hyperparameters config: {e}")

        if not self.reward_function_code.strip():
            issues.append("Reward function code cannot be empty")
        elif "def reward_function" not in self.reward_function_code:
            issues.append("Reward function must contain 'def reward_function' definition")

        if self.backend == LocalTrainingBackend.DEEPRACER_FOR_CLOUD:
            if self.docker_compose_file and not self.docker_compose_file.exists():
                issues.append(f"Docker compose file not found: {self.docker_compose_file}")

        return issues

    def is_deepracer_compatible(self) -> bool:
        """Check if configuration is compatible with DeepRacer for Cloud.

        Returns
        -------
        bool
            True if compatible, False otherwise
        """
        return len(self.validate_deepracer_compatibility()) == 0

    def create_deepracer_for_cloud_files(self, base_path: Path) -> Dict[str, Path]:
        """Create DeepRacer for Cloud configuration files.

        Parameters
        ----------
        base_path : Path
            Base directory to create files in

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping file type to created file path
        """
        base_path.mkdir(parents=True, exist_ok=True)
        created_files = {}

        info(f"Creating DeepRacer for Cloud configuration files in {base_path}")

        try:
            reward_file = base_path / "reward_function.py"
            reward_file.write_text(self.reward_function_code)
            created_files["reward_function"] = reward_file
            debug(f"Created reward function file: {reward_file}")

            hyperparams_file = base_path / "hyperparameters.json"
            import json

            hyperparams_config = self.get_hyperparameters_config()
            hyperparams_file.write_text(json.dumps(hyperparams_config, indent=2))
            created_files["hyperparameters"] = hyperparams_file
            debug(f"Created hyperparameters file: {hyperparams_file}")

            action_space_file = base_path / "action_space.json"
            action_space_config = self.get_action_space_config()
            action_space_file.write_text(json.dumps(action_space_config, indent=2))
            created_files["action_space"] = action_space_file
            debug(f"Created action space file: {action_space_file}")

            metadata_file = base_path / "model_metadata.json"
            metadata = {
                "model_name": self.model_name,
                "version": self.version,
                "description": self.description,
                "track_name": self.track_name.value,
                "sensor_type": self.sensor_type.value,
                "training_algorithm": self.training_algorithm.value,
                "experimental_scenario": self.experimental_scenario.value,
                "backend": self.backend.value,
                "tags": self.tags,
            }
            metadata_file.write_text(json.dumps(metadata, indent=2))
            created_files["metadata"] = metadata_file
            debug(f"Created metadata file: {metadata_file}")

            training_config_file = base_path / "training_params.json"
            training_config = self.get_deepracer_config()
            training_config_file.write_text(json.dumps(training_config, indent=2))
            created_files["training_config"] = training_config_file
            debug(f"Created training config file: {training_config_file}")

            if self.backend == LocalTrainingBackend.DEEPRACER_FOR_CLOUD:
                docker_config_file = base_path / "docker_config.json"
                docker_config = {
                    "compose_file": str(self.docker_compose_file) if self.docker_compose_file else None,
                    "use_gpu": self.use_gpu,
                    "memory_limit_gb": self.memory_limit_gb,
                    "cpu_limit": self.cpu_limit,
                    "num_workers": self.num_workers,
                }
                docker_config_file.write_text(json.dumps(docker_config, indent=2))
                created_files["docker_config"] = docker_config_file
                debug(f"Created Docker config file: {docker_config_file}")

            info(f"Successfully created {len(created_files)} configuration files")
            return created_files

        except Exception as e:
            error(f"Failed to create configuration files: {e}")
            raise

    def setup_training_environment(self) -> Dict[str, Path]:
        """Setup the complete training environment for DeepRacer for Cloud.

        Creates all necessary directories and configuration files,
        similar to AWS DeepRacer model setup but for local deployment.

        Returns
        -------
        Dict[str, Path]
            Dictionary of created files and directories
        """
        info(f"Setting up training environment for model: {self.model_name}")

        model_dir = self.output_directory / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        config_files = self.create_deepracer_for_cloud_files(model_dir)

        directories = {}

        logs_dir = model_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        directories["logs"] = logs_dir
        debug(f"Created logs directory: {logs_dir}")

        models_dir = model_dir / "models"
        models_dir.mkdir(exist_ok=True)
        directories["models"] = models_dir
        debug(f"Created models directory: {models_dir}")

        metrics_dir = model_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        directories["metrics"] = metrics_dir
        debug(f"Created metrics directory: {metrics_dir}")

        if self.use_tensorboard:
            tensorboard_dir = model_dir / "tensorboard"
            tensorboard_dir.mkdir(exist_ok=True)
            directories["tensorboard"] = tensorboard_dir
            debug(f"Created TensorBoard directory: {tensorboard_dir}")

        info(
            f"Training environment setup complete. Created {len(config_files)} config files and {len(directories)} directories"
        )

        return {"model_directory": model_dir, "config_files": config_files, "directories": directories}

    def get_model_config_for_deployment(self) -> Dict[str, Any]:
        """Get model configuration formatted for deployment.

        Returns configuration in AWS DeepRacer console compatible format,
        without checkpoint references since this is for new model creation.

        Returns
        -------
        Dict[str, Any]
            Model configuration for deployment
        """
        return {
            "ModelName": self.model_name,
            "Description": self.description,
            "TrainingJobArn": f"local://training-job/{self.model_name}",
            "TrainingParameters": {
                "TrainingAlgorithm": self.training_algorithm.value,
                "MaxJobDurationInSeconds": self.max_job_duration_seconds,
                "HyperParameters": self.get_hyperparameters_config(),
                "ActionSpace": self.get_action_space_config(),
                "SensorConfiguration": {"SensorType": self.sensor_type.value},
                "TrackName": self.track_name.value,
                "RewardFunction": {"Code": self.reward_function_code},
                "ExperimentalScenario": self.experimental_scenario.value,
            },
            "LocalConfiguration": {
                "Backend": self.backend.value,
                "Device": self.device.value,
                "UseGPU": self.use_gpu,
                "MemoryLimitGB": self.memory_limit_gb,
                "CPULimit": self.cpu_limit,
                "NumWorkers": self.num_workers,
            },
            "Metadata": {"Version": self.version, "Tags": self.tags, "CreatedAt": "2025-07-31"},
        }

    def get_model_save_path(self, episode: Optional[int] = None) -> Path:
        """Get the path to save the model.

        Parameters
        ----------
        episode : Optional[int], optional
            Episode number for checkpoint naming, by default None

        Returns
        -------
        Path
            Path to save the model
        """
        if episode is not None:
            return self.output_directory / f"{self.model_name}_episode_{episode}.zip"
        else:
            return self.output_directory / f"{self.model_name}_final.zip"

    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the training environment setup.

        Returns
        -------
        Dict[str, Any]
            Environment configuration information
        """
        return {
            "track": {"name": self.track_name.value, "type": "simulation"},
            "action_space": {
                "type": self.action_space_type.value,
                "description": self.action_space_type.get_description(),
                "complexity": "discrete" if self.action_space_type == ActionSpaceType.DISCRETE else "continuous",
            },
            "observation_space": {
                "type": self.sensor_type.value,
                "description": f"Sensor configuration: {self.sensor_type.value}",
                "complexity": "camera_based",
            },
        }

    @classmethod
    def create_for_scenario(
        cls, model_name: str, output_directory: Path, scenario: ExperimentalScenario, **kwargs
    ) -> "LocalDeploymentConfig":
        """Create a configuration optimized for a specific scenario.

        Parameters
        ----------
        model_name : str
            Name for the model
        output_directory : Path
            Output directory for training
        scenario : ExperimentalScenario
            The experimental scenario
        **kwargs
            Additional configuration overrides

        Returns
        -------
        LocalDeploymentConfig
            Optimized configuration for the scenario
        """
        track_name = TrackType.get_scenario_tracks(scenario.value)
        device = LocalComputeDevice.AUTO

        scenario_defaults = {
            ExperimentalScenario.SPEED_OPTIMIZATION: {
                "backend": LocalTrainingBackend.DEEPRACER_FOR_CLOUD,
                "device": device,
                "track_name": track_name,
                "action_space_type": ActionSpaceType.CONTINUOUS,
                "sensor_type": SensorType.FRONT_FACING_CAMERA,
                "max_episodes": 15000,
                "reward_threshold": 300.0,
                "custom_config": {"lr": 5e-4, "entropy_coeff": 0.02},
            },
            ExperimentalScenario.CENTERLINE_FOLLOWING: {
                "backend": LocalTrainingBackend.DEEPRACER_FOR_CLOUD,
                "device": device,
                "track_name": track_name,
                "action_space_type": ActionSpaceType.DISCRETE,
                "sensor_type": SensorType.FRONT_FACING_CAMERA,
                "max_episodes": 8000,
                "reward_threshold": 200.0,
                "custom_config": {"learning_rate": 3e-4, "ent_coef": 0.01},
            },
            ExperimentalScenario.TIME_TRIAL: {
                "backend": LocalTrainingBackend.DEEPRACER_FOR_CLOUD,
                "device": device,
                "track_name": track_name,
                "action_space_type": ActionSpaceType.CONTINUOUS,
                "sensor_type": SensorType.FRONT_FACING_CAMERA,
                "max_episodes": 20000,
                "reward_threshold": 500.0,
                "custom_config": {"lr": 1e-3, "train_batch_size": 8000},
            },
        }

        defaults = scenario_defaults.get(scenario, scenario_defaults[ExperimentalScenario.TIME_TRIAL])

        config_params = {
            "model_name": model_name,
            "reward_function_code": kwargs.get("reward_function_code", "def reward_function(params): return 1.0"),
            "output_directory": output_directory,
            **defaults,
            **kwargs,
        }

        return cls(**config_params)

    @classmethod
    def create_for_research(cls, model_name: str, output_directory: Path, **kwargs) -> "LocalDeploymentConfig":
        """Create a configuration optimized for research scenarios.

        Parameters
        ----------
        model_name : str
            Name for the model
        output_directory : Path
            Output directory for training
        **kwargs
            Additional configuration overrides

        Returns
        -------
        LocalDeploymentConfig
            Research-optimized configuration
        """
        defaults = {
            "backend": LocalTrainingBackend.DEEPRACER_FOR_CLOUD,
            "device": LocalComputeDevice.AUTO,
            "sensor_type": SensorType.FRONT_FACING_CAMERA,
            "action_space_type": ActionSpaceType.CONTINUOUS,
            "use_tensorboard": True,
            "checkpoint_frequency": 500,
            "log_frequency": 50,
        }

        return cls(
            model_name=model_name,
            output_directory=output_directory,
            reward_function_code="def reward_function(params): return 1.0",
            track_name=TrackType.REINVENT_BASE,
            **{**defaults, **kwargs},
        )


def create_local_deployment_config(
    model_name: str,
    description: str = "",
    reward_scenario: Optional[ExperimentalScenario] = None,
    output_directory: Optional[Path] = None,
    **kwargs,
) -> LocalDeploymentConfig:
    """Create a local deployment configuration with sensible defaults.

    Parameters
    ----------
    model_name : str
        Name for the model
    description : str, optional
        Model description, by default ""
    reward_scenario : Optional[ExperimentalScenario], optional
        Experimental scenario for optimization, by default None
    output_directory : Optional[Path], optional
        Output directory for training, by default None (uses ./models/{model_name})
    **kwargs
        Additional configuration options

    Returns
    -------
    LocalDeploymentConfig
        Configured local deployment configuration
    """
    if output_directory is None:
        output_directory = Path(f"./models/{model_name}")

    if reward_scenario:
        config = LocalDeploymentConfig.create_for_scenario(
            model_name=model_name, output_directory=output_directory, scenario=reward_scenario, **kwargs
        )
    else:
        config = LocalDeploymentConfig.create_for_research(model_name=model_name, output_directory=output_directory, **kwargs)

    if description:
        config.tags["description"] = description

    return config
