import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from deepracer_research.config.aws.types.action_space_type import ActionSpaceType
from deepracer_research.config.aws.types.sensor_type import SensorType
from deepracer_research.config.network.architecture_type import ArchitectureType
from deepracer_research.config.training.loss_type import LossType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.deployment.deepracer.enum.instance_type import DeepRacerInstanceType
from deepracer_research.deployment.deepracer.hyperparameters import DeepRacerHyperparameters
from deepracer_research.experiments import ExperimentalScenario


@dataclass
class AWSDeepRacerConfig:
    """Configuration for AWS DeepRacer console training

    Parameters
    ----------
    model_name : str
        Name for the DeepRacer model
    track_arn : str
        ARN of the track to train on
    reward_function_code : str
        Python code for the reward function
    action_space_type : ActionSpaceType, optional
        Type of action space ('discrete' or 'continuous'), by default 'discrete'
    training_algorithm : TrainingAlgorithm, optional
        Training algorithm to use, by default PPO
    architecture_type : ArchitectureType, optional
        Neural network architecture type, by default ATTENTION_CNN
    hyperparameters : DeepRacerHyperparameters, optional
        Training hyperparameters, by default DeepRacerHyperparameters()
    max_job_duration_seconds : int, optional
        Maximum training duration in seconds, by default 7200 (2 hours)
    sensor_type : Union[SensorType, List[SensorType]], optional
        Sensor configuration(s), by default FRONT_FACING_CAMERA
        Can be a single sensor or list of sensors
    race_type : DeepRacerRaceType, optional
        Race type for training, by default TIME_TRIAL
    instance_type : DeepRacerInstanceType, optional
        Instance type for training resources, by default ML_C5_2XLARGE
    instance_count : int, optional
        Number of instances for training, by default 1
    volume_size_gb : int, optional
        Volume size in GB for training, by default 30
    tags : Dict[str, str], optional
        Tags for the training job, by default empty dict
    """

    model_name: str
    track_arn: str
    reward_function_code: str

    action_space_type: ActionSpaceType = ActionSpaceType.DISCRETE
    steering_angle_granularity: int = 5
    max_steering_angle: float = 30.0
    speed_granularity: int = 3
    max_speed: float = 4.0
    min_speed: float = 0.5

    training_algorithm: TrainingAlgorithm = TrainingAlgorithm.CLIPPED_PPO
    architecture_type: ArchitectureType = ArchitectureType.ATTENTION_CNN
    hyperparameters: DeepRacerHyperparameters = field(default_factory=DeepRacerHyperparameters)
    max_job_duration_seconds: int = 7200

    sensor_type: Union[SensorType, List[SensorType]] = SensorType.FRONT_FACING_CAMERA
    experimental_scenario: ExperimentalScenario = ExperimentalScenario.TIME_TRIAL
    instance_type: DeepRacerInstanceType = DeepRacerInstanceType.ML_C5_2XLARGE
    instance_count: int = 1
    volume_size_gb: int = 30

    description: str = ""
    version: str = "1.0.0"
    tags: Dict[str, str] = field(default_factory=dict)

    def get_sensor_list(self) -> List[SensorType]:
        """Get the sensor configuration as a list.

        Returns
        -------
        List[SensorType]
            List of sensors, ensuring it's always a list
        """
        if isinstance(self.sensor_type, list):
            return self.sensor_type
        else:
            return [self.sensor_type]

    def get_primary_sensor(self) -> SensorType:
        """Get the primary (first) sensor from the configuration.

        Returns
        -------
        SensorType
            The primary sensor
        """
        sensor_list = self.get_sensor_list()
        return sensor_list[0]

    def get_action_space_config(self) -> Dict[str, Any]:
        """Get the action space configuration for DeepRacer (file format).

        Returns
        -------
        Dict[str, Any]
            Action space configuration
        """
        if self.action_space_type == ActionSpaceType.DISCRETE:
            actions = []

            steering_angles = []
            if self.steering_angle_granularity >= 1:
                steering_angles.append(0.0)

            if self.steering_angle_granularity >= 3:
                slight_angle = self.max_steering_angle * 0.25
                steering_angles.extend([-slight_angle, slight_angle])

            if self.steering_angle_granularity >= 5:
                medium_angle = self.max_steering_angle * 0.5
                steering_angles.extend([-medium_angle, medium_angle])

            if self.steering_angle_granularity >= 7:
                sharp_angle = self.max_steering_angle * 0.75
                steering_angles.extend([-sharp_angle, sharp_angle])

            if self.steering_angle_granularity >= 9:
                steering_angles.extend([-self.max_steering_angle, self.max_steering_angle])

            steering_angles = sorted(list(set(steering_angles)))

            speeds = []
            for speed_idx in range(self.speed_granularity):
                if self.speed_granularity == 1:
                    speed = (self.min_speed + self.max_speed) / 2
                else:
                    speed = self.min_speed + (self.max_speed - self.min_speed) * speed_idx / (self.speed_granularity - 1)
                speeds.append(round(speed, 2))

            for i, angle in enumerate(steering_angles):
                abs_angle = abs(angle)
                angle_factor = abs_angle / self.max_steering_angle if self.max_steering_angle > 0 else 0

                if abs_angle == 0:
                    speed = speeds[-1]
                elif angle_factor <= 0.3:
                    speed_idx = max(len(speeds) - 2, 0)
                    speed = speeds[speed_idx]
                elif angle_factor <= 0.6:
                    speed_idx = len(speeds) // 2
                    speed = speeds[speed_idx]
                else:
                    speed = speeds[0]

                actions.append({"speed": speed, "steering_angle": round(angle, 2)})

            return {"actionSpaceType": ActionSpaceType.DISCRETE.value, "actions": actions}
        else:
            return {
                "actionSpaceType": ActionSpaceType.CONTINUOUS.value,
                "speedRange": {"min": self.min_speed, "max": self.max_speed},
                "steeringAngleRange": {"min": -self.max_steering_angle, "max": self.max_steering_angle},
            }

    def get_api_action_space_config(self) -> Dict[str, Any]:
        """Get the action space configuration for DeepRacer API.

        Returns
        -------
        Dict[str, Any]
            Action space configuration for API calls
        """
        if self.action_space_type == ActionSpaceType.DISCRETE:
            pass

    def get_api_action_space_config(self) -> Dict[str, Any]:
        """Get the action space configuration for DeepRacer API.

        Returns
        -------
        Dict[str, Any]
            Action space configuration for API calls
        """
        if self.action_space_type == ActionSpaceType.DISCRETE:
            actions = []

            steering_angles = []
            if self.steering_angle_granularity >= 1:
                steering_angles.append(0.0)

            if self.steering_angle_granularity >= 3:
                slight_angle = self.max_steering_angle * 0.25
                steering_angles.extend([-slight_angle, slight_angle])

            if self.steering_angle_granularity >= 5:
                medium_angle = self.max_steering_angle * 0.5
                steering_angles.extend([-medium_angle, medium_angle])

            if self.steering_angle_granularity >= 7:
                sharp_angle = self.max_steering_angle * 0.75
                steering_angles.extend([-sharp_angle, sharp_angle])

            if self.steering_angle_granularity >= 9:
                steering_angles.extend([-self.max_steering_angle, self.max_steering_angle])

            steering_angles = sorted(list(set(steering_angles)))

            speeds = []
            for speed_idx in range(self.speed_granularity):
                if self.speed_granularity == 1:
                    speed = (self.min_speed + self.max_speed) / 2
                else:
                    speed = self.min_speed + (self.max_speed - self.min_speed) * speed_idx / (self.speed_granularity - 1)
                speeds.append(round(speed, 2))

            for i, angle in enumerate(steering_angles):
                abs_angle = abs(angle)
                angle_factor = abs_angle / self.max_steering_angle if self.max_steering_angle > 0 else 0

                if abs_angle == 0:
                    speed = speeds[-1]
                elif angle_factor <= 0.3:
                    speed_idx = max(len(speeds) - 2, 0)
                    speed = speeds[speed_idx]
                elif angle_factor <= 0.6:
                    speed_idx = len(speeds) // 2
                    speed = speeds[speed_idx]
                else:
                    speed = speeds[0]

                actions.append({"Speed": speed, "SteeringAngle": round(angle, 2)})

            return {"actionSpaceType": ActionSpaceType.DISCRETE.value, "actions": actions}
        else:
            return {
                "actionSpaceType": ActionSpaceType.CONTINUOUS.value,
                "speedRange": {"min": self.min_speed, "max": self.max_speed},
                "steeringAngleRange": {"min": -self.max_steering_angle, "max": self.max_steering_angle},
            }

    def to_deepracer_training_job_config(self, reward_function_s3_uri: str, role_arn: str = "") -> Dict[str, Any]:
        """Convert to AWS DeepRacer training job configuration.

        Parameters
        ----------
        reward_function_s3_uri : str
            S3 URI of the uploaded reward function
        role_arn : str, optional
            ARN of the IAM role for the training job

        Returns
        -------
        Dict[str, Any]
            Complete training job configuration for DeepRacer API
        """
        config = {
            "ModelName": re.sub(r"[^a-zA-Z0-9\-]", "-", self.model_name)[:64],
            "ModelFramework": "TENSOR_FLOW",
            "AgentAlgorithm": self.training_algorithm.value.upper(),
            "AgentNetwork": "SIX_LAYER_DOUBLE_HEAD_OUTPUT",
            "RoleArn": role_arn,
            "TrainingConfig": {
                "RewardFunctionS3Source": reward_function_s3_uri,
                "TrackConfig": {"TrackArn": self.track_arn},
                "RaceType": self.experimental_scenario.to_deepracer_race_type(),
                "ResourceConfig": {"InstanceType": self.instance_type.value, "InstanceCount": self.instance_count},
                "TerminationConditions": {"MaxTimeInMinutes": self.max_job_duration_seconds // 60},
                "Hyperparameters": self.hyperparameters.to_deepracer_format(),
            },
        }

        action_space_config = self.get_api_action_space_config()
        if action_space_config:
            if action_space_config.get("actionSpaceType") == ActionSpaceType.DISCRETE.value:
                config["CustomActionSpace"] = {
                    "ActionSpaceType": ActionSpaceType.DISCRETE.value.upper(),
                    "DiscreteActionSpace": action_space_config["actions"],
                }
            else:
                config["CustomActionSpace"] = {
                    "ActionSpaceType": ActionSpaceType.CONTINUOUS.value.upper(),
                    "ContinuousActionSpace": {
                        "SpeedRange": action_space_config["speedRange"],
                        "SteeringAngleRange": action_space_config["steeringAngleRange"],
                    },
                }

        if self.description:
            config["ModelDescription"] = self.description

        if self.tags:
            config["Tags"] = [{"Key": k, "Value": v} for k, v in self.tags.items()]

        return config

    def to_deepracer_config_file(self) -> Dict[str, Any]:
        """Convert to AWS DeepRacer configuration file format for manual deployment.

        Returns
        -------
        Dict[str, Any]
            Configuration suitable for saving to file for manual deployment
        """
        if self.action_space_type == ActionSpaceType.DISCRETE:
            actions = []
            speeds = [self.min_speed, (self.min_speed + self.max_speed) / 2, self.max_speed]

            steering_angles = []
            for i in range(self.steering_angle_granularity):
                if i == 0:
                    steering_angles.append(0.0)
                else:
                    angle = -self.max_steering_angle + (2 * self.max_steering_angle * i) / (self.steering_angle_granularity - 1)
                    steering_angles.append(angle)

            for speed in speeds:
                for steering_angle in steering_angles:
                    actions.append({"speed": round(speed, 1), "steering_angle": round(steering_angle, 1)})

            action_space_config = {"actionSpaceType": self.action_space_type.value, "actions": actions}
        else:
            action_space_config = {
                "actionSpaceType": self.action_space_type.value,
                "continuousActionSpace": {
                    "steering_angle": {"high": self.max_steering_angle, "low": -self.max_steering_angle},
                    "speed": {"high": self.max_speed, "low": self.min_speed},
                },
            }

        hyperparams = self.hyperparameters.to_dict()
        hyperparams_str = {k: str(v) for k, v in hyperparams.items()}

        config = {
            "modelName": re.sub(r"[^a-zA-Z0-9\-]", "-", self.model_name)[:64],
            "modelDescription": self.description or f"DeepRacer model: {self.model_name}",
            "roleArn": "",
            "version": self.version,
            "trainingAlgorithm": self.training_algorithm.value.lower(),
            "actionSpace": action_space_config,
            "sensorConfiguration": [sensor.value for sensor in self.get_sensor_list()],
            "hyperParameters": hyperparams_str,
            "environment": {
                "trackArn": self.track_arn,
                "raceType": self.experimental_scenario.to_deepracer_race_type(),
                "scenario": self.experimental_scenario.value,
            },
            "rewardFunction": {"code": self.reward_function_code},
            "trainingConfig": {
                "maxRuntimeInSeconds": self.max_job_duration_seconds,
                "volumeSizeInGB": self.volume_size_gb,
                "instanceType": self.instance_type.value,
                "instanceCount": self.instance_count,
            },
        }

        if self.tags:
            config["tags"] = [{"key": k, "value": v} for k, v in self.tags.items()]

        return config

    def validate_configuration(self) -> bool:
        """Validate the DeepRacer configuration.

        Returns
        -------
        bool
            True if configuration is valid, False otherwise
        """
        required_fields = [self.model_name, self.track_arn, self.reward_function_code]
        if not all(field.strip() for field in required_fields):
            return False

        if self.max_speed <= self.min_speed:
            return False
        if self.min_speed < 0.1 or self.max_speed > 4.0:
            return False

        if self.max_steering_angle <= 0 or self.max_steering_angle > 30.0:
            return False

        if self.steering_angle_granularity not in [3, 5, 7]:
            return False
        if self.speed_granularity < 1 or self.speed_granularity > 10:
            return False

        if not self.validate_algorithm_action_space_compatibility(
            self.training_algorithm, self.action_space_type, self.experimental_scenario
        ):
            return False

        return True

    @staticmethod
    def get_compatible_action_space_type(
        algorithm: TrainingAlgorithm, scenario: ExperimentalScenario = None
    ) -> ActionSpaceType:
        """Get the compatible action space type for a given training algorithm and scenario.

        Parameters
        ----------
        algorithm : TrainingAlgorithm
            The training algorithm
        scenario : ExperimentalScenario, optional
            The experimental scenario

        Returns
        -------
        ActionSpaceType
            The compatible action space type for the algorithm and scenario
        """
        if algorithm == TrainingAlgorithm.SAC:
            return ActionSpaceType.CONTINUOUS

        if scenario in [ExperimentalScenario.OBJECT_AVOIDANCE, ExperimentalScenario.HEAD_TO_HEAD]:
            if algorithm in [TrainingAlgorithm.PPO, TrainingAlgorithm.CLIPPED_PPO]:
                return ActionSpaceType.CONTINUOUS
            else:
                return ActionSpaceType.CONTINUOUS

        return ActionSpaceType.DISCRETE

    @staticmethod
    def validate_algorithm_action_space_compatibility(
        algorithm: TrainingAlgorithm, action_space_type: ActionSpaceType, scenario: ExperimentalScenario = None
    ) -> bool:
        """Validate if an algorithm is compatible with an action space type and scenario.

        Parameters
        ----------
        algorithm : TrainingAlgorithm
            The training algorithm
        action_space_type : ActionSpaceType
            The action space type
        scenario : ExperimentalScenario, optional
            The experimental scenario

        Returns
        -------
        bool
            True if the algorithm is compatible with the action space type and scenario
        """
        if algorithm == TrainingAlgorithm.SAC and action_space_type == ActionSpaceType.DISCRETE:
            return False

        if scenario in [ExperimentalScenario.OBJECT_AVOIDANCE, ExperimentalScenario.HEAD_TO_HEAD]:
            if algorithm in [TrainingAlgorithm.PPO, TrainingAlgorithm.CLIPPED_PPO]:
                return True
            elif action_space_type == ActionSpaceType.DISCRETE:
                return False

        return True

    @classmethod
    def create_for_scenario(
        cls,
        model_name: str,
        track_arn: str,
        reward_function_code: str,
        experimental_scenario: ExperimentalScenario = ExperimentalScenario.TIME_TRIAL,
        **kwargs,
    ) -> "AWSDeepRacerConfig":
        """Create a configuration optimized for a specific scenario.

        Parameters
        ----------
        model_name : str
            Name for the model
        track_arn : str
            ARN of the track
        reward_function_code : str
            Reward function code
        experimental_scenario : ExperimentalScenario, optional
            The experimental scenario, by default TIME_TRIAL
        **kwargs
            Additional configuration overrides

        Returns
        -------
        AWSDeepRacerConfig
            Optimized configuration for the scenario
        """
        scenario_defaults = {
            ExperimentalScenario.SPEED_OPTIMIZATION: {
                "max_speed": 4.0,
                "min_speed": 1.5,
                "max_steering_angle": 25.0,
                "speed_granularity": 4,
                "steering_angle_granularity": 5,
                "action_space_type": ActionSpaceType.CONTINUOUS,
                "training_algorithm": TrainingAlgorithm.SAC,
                "hyperparameters": DeepRacerHyperparameters(
                    batch_size=256,
                    learning_rate=0.0008,
                    sac_alpha=0.12,
                    discount_factor=0.992,
                    loss_type=LossType.HUBER,
                    num_episodes_between_training=10,
                    entropy_coefficient=0.008,
                    lr_decay_rate=0.999,
                    num_epochs=6,
                    stack_size=3,
                    epsilon=0.12,
                    beta_entropy=0.008,
                ),
            },
            ExperimentalScenario.CENTERLINE_FOLLOWING: {
                "max_speed": 3.0,
                "min_speed": 0.8,
                "max_steering_angle": 20.0,
                "speed_granularity": 4,
                "steering_angle_granularity": 9,
                "action_space_type": ActionSpaceType.DISCRETE,
                "training_algorithm": TrainingAlgorithm.CLIPPED_PPO,
                "hyperparameters": DeepRacerHyperparameters(
                    batch_size=128,
                    num_epochs=10,
                    learning_rate=0.0003,
                    entropy_coefficient=0.01,
                    epsilon=0.2,
                    discount_factor=0.998,
                    loss_type=LossType.HUBER,
                    num_episodes_between_training=20,
                    lr_decay_rate=0.9995,
                    stack_size=4,
                    beta_entropy=0.01,
                    beta=0.02,
                ),
            },
            ExperimentalScenario.TIME_TRIAL: {
                "max_speed": 4.0,
                "min_speed": 1.0,
                "max_steering_angle": 28.0,
                "speed_granularity": 4,
                "steering_angle_granularity": 5,
                "action_space_type": ActionSpaceType.CONTINUOUS,
                "training_algorithm": TrainingAlgorithm.SAC,
                "hyperparameters": DeepRacerHyperparameters(
                    batch_size=128,
                    learning_rate=0.0005,
                    sac_alpha=0.15,
                    discount_factor=0.995,
                    loss_type=LossType.HUBER,
                    num_episodes_between_training=12,
                    entropy_coefficient=0.01,
                    lr_decay_rate=0.9995,
                    num_epochs=5,
                    stack_size=4,
                ),
            },
            ExperimentalScenario.OBJECT_AVOIDANCE: {
                "max_speed": 2.5,
                "min_speed": 0.5,
                "max_steering_angle": 30.0,
                "speed_granularity": 5,
                "steering_angle_granularity": 7,
                "action_space_type": ActionSpaceType.CONTINUOUS,
                "training_algorithm": TrainingAlgorithm.SAC,
                "hyperparameters": DeepRacerHyperparameters(
                    batch_size=64,
                    learning_rate=0.0002,
                    sac_alpha=0.3,
                    discount_factor=0.98,
                    loss_type=LossType.HUBER,
                    num_episodes_between_training=8,
                    entropy_coefficient=0.03,
                    lr_decay_rate=0.999,
                    num_epochs=6,
                    stack_size=5,
                    epsilon=0.25,
                ),
            },
            ExperimentalScenario.HEAD_TO_HEAD: {
                "max_speed": 3.8,
                "min_speed": 0.8,
                "max_steering_angle": 30.0,
                "speed_granularity": 4,
                "steering_angle_granularity": 5,
                "action_space_type": ActionSpaceType.CONTINUOUS,
                "training_algorithm": TrainingAlgorithm.SAC,
                "hyperparameters": DeepRacerHyperparameters(
                    batch_size=128,
                    learning_rate=0.0003,
                    sac_alpha=0.2,
                    discount_factor=0.99,
                    loss_type=LossType.HUBER,
                    num_episodes_between_training=10,
                    entropy_coefficient=0.015,
                    lr_decay_rate=0.9998,
                    num_epochs=4,
                    stack_size=4,
                    epsilon=0.18,
                    beta_entropy=0.02,
                ),
            },
        }

        defaults = scenario_defaults.get(experimental_scenario, {})

        config_params = {
            "model_name": model_name,
            "track_arn": track_arn,
            "reward_function_code": reward_function_code,
            "experimental_scenario": experimental_scenario,
            **defaults,
            **kwargs,
        }

        return cls(**config_params)
