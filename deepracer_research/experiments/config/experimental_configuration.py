import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from deepracer_research.config import ArchitectureType
from deepracer_research.config.aws import ActionSpaceConfiguration
from deepracer_research.config.track.track_type import TrackType
from deepracer_research.experiments.config.hyperparameter_configuration import HyperparameterConfiguration
from deepracer_research.experiments.config.sensor_configuration import SensorConfiguration
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.experiments.enums.sensor_modality import SensorModality
from deepracer_research.rewards.reward_function_type import RewardFunctionType


@dataclass
class ExperimentalConfiguration:
    """Comprehensive experimental configuration for systematic evaluation."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "deepracer_experiment"
    description: str = ""

    scenario: ExperimentalScenario = ExperimentalScenario.CENTERLINE_FOLLOWING
    sensor_config: SensorConfiguration = field(default_factory=lambda: SensorConfiguration(SensorModality.MONOCULAR_CAMERA))
    network_architecture: ArchitectureType = ArchitectureType.ATTENTION_CNN
    action_space_config: ActionSpaceConfiguration = field(default_factory=ActionSpaceConfiguration)
    hyperparameters: HyperparameterConfiguration = field(default_factory=HyperparameterConfiguration)

    training_time_minutes: int = 180
    racing_track: TrackType = TrackType.REINVENT_BASE
    reward_function_name: RewardFunctionType = RewardFunctionType.CENTERLINE_FOLLOWING
    stop_conditions: Dict[str, Any] = field(default_factory=lambda: {"max_episodes": 1000})

    created_at: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived fields after object creation."""
        if not self.tags:
            self.tags = {
                "scenario": self.scenario.value,
                "sensor_modality": self.sensor_config.modality.value,
                "architecture": self.network_architecture.value,
                "experiment_id": self.experiment_id,
            }

    def to_aws_training_config(self, role_arn: str) -> Dict[str, Any]:
        """Convert to AWS DeepRacer training job configuration."""
        model_name = f"{self.name}-{self.experiment_id}"

        config = {
            "ModelName": model_name,
            "RoleArn": role_arn,
            "TrainingMode": "RL",
            "RacingTrack": self._get_track_for_scenario(),
            "RewardFunction": self._get_reward_function_code(),
            "TrainingAlgorithm": "clippedPPO",
            "ActionSpace": self.action_space_config.generate_action_space(),
            "NeuralNetwork": self._get_aws_network_type(),
            "Sensors": self.sensor_config.to_aws_format(),
            "Hyperparameters": self.hyperparameters.to_aws_format(),
            "StoppingConditions": {"MaxTrainingTimeInSeconds": self.training_time_minutes * 60, **self.stop_conditions},
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()]
            + [{"Key": "Research", "Value": "Thesis"}, {"Key": "CreatedAt", "Value": self.created_at.isoformat()}],
        }

        return config

    def _get_track_for_scenario(self) -> str:
        """Select appropriate track based on experimental scenario."""
        track_mapping = {
            ExperimentalScenario.CENTERLINE_FOLLOWING: TrackType.REINVENT_BASE,
            ExperimentalScenario.STATIC_OBJECT_AVOIDANCE: TrackType.OVAL_TRACK,
            ExperimentalScenario.DYNAMIC_OBJECT_AVOIDANCE: TrackType.BOWTIE_TRACK,
            ExperimentalScenario.MULTI_AGENT_RACING: TrackType.HEAD_TO_HEAD_TRACK,
            ExperimentalScenario.SPEED_OPTIMIZATION: TrackType.SPEED_TRACK,
            ExperimentalScenario.TIME_TRIAL: TrackType.TIME_TRIAL_TRACK,
            ExperimentalScenario.HEAD_TO_HEAD: TrackType.HEAD_TO_HEAD_TRACK,
        }
        track = track_mapping.get(self.scenario, self.racing_track)
        return track.value if isinstance(track, TrackType) else track

    def _get_aws_network_type(self) -> str:
        """Convert architecture type to AWS DeepRacer network format."""
        aws_networks = {
            ArchitectureType.ATTENTION_CNN: "DEEP_CONVOLUTIONAL_NETWORK",
            ArchitectureType.RESIDUAL_NETWORK: "RESNET_ARCHITECTURE",
            ArchitectureType.EFFICIENT_NET: "MOBILENET_ARCHITECTURE",
            ArchitectureType.TEMPORAL_CNN: "DEEP_CONVOLUTIONAL_NETWORK",
            ArchitectureType.TRANSFORMER_VISION: "DEEP_CONVOLUTIONAL_NETWORK",
            ArchitectureType.MULTI_MODAL_FUSION: "DEEP_CONVOLUTIONAL_NETWORK",
            ArchitectureType.LSTM_CNN: "DEEP_CONVOLUTIONAL_NETWORK",
        }
        return aws_networks.get(self.network_architecture, "DEEP_CONVOLUTIONAL_NETWORK")

    def _get_reward_function_code(self) -> str:
        """Get reward function code for this scenario using template system."""
        from deepracer_research.rewards.template_loader import render_reward_function

        return render_reward_function(
            scenario=self.scenario, custom_parameters=self._get_reward_parameters(), experiment_id=self.experiment_id
        )

    def _get_reward_parameters(self) -> Dict[str, Any]:
        """Get custom reward function parameters based on experiment configuration.

        Returns
        -------
        Dict[str, Any]
            Custom parameters for reward function template
        """
        base_params = {}

        if self.scenario == ExperimentalScenario.SPEED_OPTIMIZATION:
            base_params.update(
                {
                    "speed_thresholds": {"min_speed": 1.5, "target_speed": 3.5, "max_speed": 4.5},
                    "rewards": {"speed_bonus": 2.5, "centerline_bonus": 1.2},
                }
            )
        elif self.scenario == ExperimentalScenario.STATIC_OBJECT_AVOIDANCE:
            base_params.update(
                {
                    "object_detection": {"safe_distance": 0.6, "warning_distance": 1.2},
                    "rewards": {"object_avoidance": 3.5, "safe_navigation": 2.0},
                }
            )
        elif self.scenario == ExperimentalScenario.CENTERLINE_FOLLOWING:
            base_params.update({"track_width_markers": {"excellent": 0.08, "good": 0.2, "acceptable": 0.4}})

        return base_params
