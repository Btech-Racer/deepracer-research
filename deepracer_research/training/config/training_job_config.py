from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepracer_research.config.aws.aws_hyperparameters import DEFAULT_HYPERPARAMETERS, AWSHyperparameters
from deepracer_research.config.aws.types.sensor_type import SensorType
from deepracer_research.config.network import NeuralNetworkType
from deepracer_research.config.track.track_type import TrackType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.rewards.builder import RewardFunctionBuilder


@dataclass
class TrainingJobConfig:
    """Configuration for AWS DeepRacer training jobs."""

    model_name: str
    role_arn: str
    racing_track: str = TrackType.REINVENT_2019_TRACK
    training_algorithm: str = TrainingAlgorithm.CLIPPED_PPO
    action_space: List[Dict[str, float]] = field(default_factory=list)
    neural_network: str = NeuralNetworkType.SHALLOW
    sensors: List[SensorType] = field(default_factory=lambda: [SensorType.FRONT_FACING_CAMERA])
    hyperparameters: Optional[AWSHyperparameters] = None
    reward_function: str = ""
    reward_scenario: ExperimentalScenario = ExperimentalScenario.CENTERLINE_FOLLOWING
    reward_parameters: Dict[str, Any] = field(default_factory=dict)
    training_time_minutes: int = 180
    stop_conditions: Dict[str, Any] = field(default_factory=dict)
    tags: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default values after object creation."""
        if not self.action_space:
            self.action_space = self._default_action_space()
        if self.hyperparameters is None:
            self.hyperparameters = DEFAULT_HYPERPARAMETERS
        if not self.stop_conditions:
            self.stop_conditions = {"MaxTrainingTimeInSeconds": self.training_time_minutes * 60}
        if not self.reward_function:
            self.reward_function = self._build_reward_function()

    def _default_action_space(self) -> List[Dict[str, float]]:
        """Generate default action space for AWS DeepRacer."""
        actions = []
        speeds = [1.0, 2.0, 3.0]
        steering_angles = [-30.0, -15.0, 0.0, 15.0, 30.0]

        for speed in speeds:
            for steering in steering_angles:
                actions.append({"speed": speed, "steering_angle": steering})
        return actions

    def _build_reward_function(self) -> str:
        """Build reward function using the reward function builder.

        Returns
        -------
        str
            Generated reward function code for the specified scenario
        """
        builder = RewardFunctionBuilder(scenario=self.reward_scenario)

        if self.reward_parameters:
            builder = builder.with_parameters(self.reward_parameters)

        return builder.build_function_code()

    def _default_hyperparameters_dict(self) -> Dict[str, str]:
        """Generate default hyperparameters in AWS string format for backward compatibility.

        Returns
        -------
        Dict[str, str]
            Hyperparameters formatted as strings for AWS API
        """
        hyperparams = self.hyperparameters or DEFAULT_HYPERPARAMETERS

        return {
            "batch_size": str(hyperparams.batch_size),
            "beta_entropy": str(hyperparams.beta_entropy),
            "discount_factor": str(hyperparams.discount_factor),
            "e_greedy_value": str(hyperparams.e_greedy_value),
            "epsilon_steps": str(hyperparams.epsilon_steps),
            "exploration_type": hyperparams.exploration_type,
            "loss_type": hyperparams.loss_type.value,
            "lr": str(hyperparams.learning_rate),
            "num_episodes_between_training": str(hyperparams.num_episodes_between_training),
            "num_epochs": str(hyperparams.num_epochs),
            "stack_size": str(hyperparams.stack_size),
            "term_cond_avg_score": str(hyperparams.term_condition_avg_score),
            "term_cond_max_episodes": str(hyperparams.term_condition_max_episodes),
        }

    def _default_reward_function(self) -> str:
        """Generate default reward function using builder (deprecated - use _build_reward_function).

        Returns
        -------
        str
            Default reward function code
        """
        return self._build_reward_function()

    def to_aws_request(self) -> Dict[str, Any]:
        """Convert configuration to AWS DeepRacer API request format.

        Returns
        -------
        Dict[str, Any]
            Configuration formatted for AWS DeepRacer API
        """
        return {
            "ModelName": self.model_name,
            "RoleArn": self.role_arn,
            "TrainingMode": "RL",
            "RacingTrack": self.racing_track,
            "RewardFunction": self.reward_function,
            "TrainingAlgorithm": self.training_algorithm,
            "ActionSpace": self.action_space,
            "NeuralNetwork": self.neural_network,
            "Sensors": [sensor.value for sensor in self.sensors],
            "Hyperparameters": self._default_hyperparameters_dict(),
            "StoppingConditions": self.stop_conditions,
            "Tags": self.tags,
        }

    def update_reward_scenario(self, scenario: ExperimentalScenario, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Update the reward function scenario and regenerate the function.

        Parameters
        ----------
        scenario : ExperimentalScenario
            New experimental scenario
        parameters : Optional[Dict[str, Any]], optional
            New reward function parameters, by default None
        """
        self.reward_scenario = scenario
        if parameters is not None:
            self.reward_parameters = parameters

        self.reward_function = self._build_reward_function()

    def update_hyperparameters(self, hyperparameters: AWSHyperparameters) -> None:
        """Update training hyperparameters.

        Parameters
        ----------
        hyperparameters : AWSHyperparameters
            New hyperparameters configuration
        """
        if not hyperparameters.validate():
            raise ValueError("Invalid hyperparameters provided")

        self.hyperparameters = hyperparameters

    def get_hyperparameters(self) -> AWSHyperparameters:
        """Get the current hyperparameters.

        Returns
        -------
        AWSHyperparameters
            Current hyperparameters configuration
        """
        return self.hyperparameters or DEFAULT_HYPERPARAMETERS
