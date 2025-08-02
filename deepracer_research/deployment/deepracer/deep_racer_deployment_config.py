from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepracer_research.config.aws.types.sensor_type import SensorType
from deepracer_research.config.track.track_type import TrackType, get_track_arn_by_name
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.experiments import ExperimentalScenario
from deepracer_research.rewards.builder import RewardFunctionBuildConfig, RewardFunctionBuilder


@dataclass
class DeepRacerDeploymentConfig:
    """Simplified deployment configuration for AWS DeepRacer.

    Parameters
    ----------
    model_name : str
        Name for the model
    track_name : TrackType
        Track to train on
    training_algorithm : TrainingAlgorithm
        Training algorithm to use
    training_time_minutes : int
        Training duration in minutes
    region : str
        AWS region
    max_speed : float
        Maximum speed for action space
    max_steering_angle : float
        Maximum steering angle
    sensor_configuration : List[SensorType]
        List of sensors to use
    """

    model_name: str
    track_name: TrackType = TrackType.PENBAY_PRO
    training_algorithm: TrainingAlgorithm = TrainingAlgorithm.PPO
    training_time_minutes: int = 30
    region: str = "us-east-1"
    max_speed: float = 3.0
    max_steering_angle: float = 30.0
    sensor_configuration: List[SensorType] = field(default_factory=lambda: [SensorType.FRONT_FACING_CAMERA])

    def to_training_job_config(self):
        """Convert to training job configuration for the training manager."""
        from deepracer_research.training.config.training_job_config import TrainingJobConfig

        return TrainingJobConfig(
            model_name=self.model_name,
            racing_track=self.track_name.value,
            training_algorithm=self.training_algorithm.value,
            training_time_minutes=self.training_time_minutes,
            hyperparameters={"max_speed": self.max_speed, "max_steering_angle": self.max_steering_angle},
        )

    def _generate_default_reward_function(self) -> str:
        """Generate default reward function using the reward builder.

        Returns
        -------
        str
            Generated reward function code
        """
        try:
            config = RewardFunctionBuildConfig(
                scenario=ExperimentalScenario.CENTERLINE_FOLLOWING,
                parameters={
                    "track_width_multiplier": 1.0,
                    "speed_threshold": self.max_speed * 0.8,
                    "steering_threshold": self.max_steering_angle * 0.5,
                },
                include_comments=True,
                minify_code=False,
            )

            builder = RewardFunctionBuilder()
            return builder.build_reward_function(config)
        except Exception:
            from deepracer_research.rewards import basic_centerline_reward

            return basic_centerline_reward()

    def to_deepracer_training_job_config(self) -> Dict[str, Any]:
        """Convert to AWS DeepRacer training job configuration.

        Returns
        -------
        Dict[str, Any]
            Complete configuration for DeepRacer API
        """
        try:
            track_arn = get_track_arn_by_name(self.track_name.value, self.region)
        except ValueError:
            track_arn = get_track_arn_by_name("reinvent2018", self.region)

        reward_function_code = self._generate_default_reward_function()

        config = {
            "modelName": self.model_name,
            "modelDescription": f"Model created via CLI for {self.track_name.value} track",
            "trainingAlgorithm": self.training_algorithm.value.upper(),
            "environment": {"trackArn": track_arn},
            "rewardFunction": {"code": reward_function_code},
            "trainingJobConfig": {"maxRuntimeInSeconds": self.training_time_minutes * 60, "volumeSizeInGB": 30},
            "actionSpace": {"actionSpaceType": "discrete", "actions": self._generate_action_space()},
            "sensorConfiguration": [sensor.value for sensor in self.sensor_configuration],
            "hyperParameters": {
                "batch_size": "64",
                "learning_rate": "0.0003",
                "entropy_coefficient": "0.01",
                "discount_factor": "0.999",
                "loss_type": "huber",
                "num_episodes_between_training": "20",
                "num_epochs": "10",
            },
        }

        return config

    def _generate_action_space(self) -> List[Dict[str, float]]:
        """Generate a discrete action space.

        Returns
        -------
        List[Dict[str, float]]
            List of action combinations
        """
        actions = []

        speeds = [1.0, 2.0, self.max_speed]
        steering_angles = [-self.max_steering_angle, -15.0, 0.0, 15.0, self.max_steering_angle]

        for speed in speeds:
            for steering in steering_angles:
                actions.append({"speed": speed, "steering_angle": steering})

        return actions

    def validate_configuration(self) -> bool:
        """Validate the deployment configuration.

        Returns
        -------
        bool
            True if configuration is valid
        """
        if not self.model_name.strip():
            return False

        if self.training_time_minutes <= 0:
            return False

        if self.max_speed <= 0:
            return False

        if self.max_steering_angle <= 0:
            return False

        if not self.sensor_configuration:
            return False

        return True

    @classmethod
    def create_for_scenario(
        cls, model_name: str, scenario: ExperimentalScenario, track_name: Optional[TrackType] = None, **kwargs
    ) -> "DeepRacerDeploymentConfig":
        """Create configuration optimized for a specific scenario.

        Parameters
        ----------
        model_name : str
            Name for the model
        scenario : ExperimentalScenario
            The experimental scenario
        track_name : Optional[TrackType]
            Track to use (will be selected automatically if None)
        **kwargs
            Additional configuration overrides

        Returns
        -------
        DeepRacerDeploymentConfig
            Optimized configuration for the scenario
        """
        if track_name is None:
            track_name = TrackType.get_scenario_tracks(scenario.value)

        scenario_defaults = {
            ExperimentalScenario.SPEED_OPTIMIZATION: {
                "training_algorithm": TrainingAlgorithm.SAC,
                "max_speed": 6.0,
                "training_time_minutes": 60,
            },
            ExperimentalScenario.CENTERLINE_FOLLOWING: {
                "training_algorithm": TrainingAlgorithm.PPO,
                "max_speed": 3.0,
                "training_time_minutes": 45,
            },
            ExperimentalScenario.TIME_TRIAL: {
                "training_algorithm": TrainingAlgorithm.SAC,
                "max_speed": 8.0,
                "training_time_minutes": 90,
            },
        }

        defaults = scenario_defaults.get(scenario, {})

        config_params = {"model_name": model_name, "track_name": track_name, **defaults, **kwargs}

        return cls(**config_params)
