from typing import Any, Dict, Optional, Union

from deepracer_research.config.aws.action_space_config import ActionSpaceConfig
from deepracer_research.config.aws.aws_hyperparameters import AWSHyperparameters
from deepracer_research.config.aws.aws_model_metadata import AWSModelMetadata
from deepracer_research.config.aws.sensor_config import SensorConfig
from deepracer_research.config.aws.types.action_space_type import ActionSpaceType
from deepracer_research.config.aws.types.sensor_type import SensorType
from deepracer_research.config.network.neural_network_type import NeuralNetworkType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.models.build.aws_deepracer_model import AWSDeepRacerModel
from deepracer_research.models.build.aws_model_config import AWSModelConfig
from deepracer_research.rewards.builder import RewardFunctionBuilder
from deepracer_research.utils.logger import info


class AWSModelBuilder:
    """Builder class for converting research models to AWS DeepRacer format"""

    def __init__(self):
        """Initialize the AWS model builder.

        Returns
        -------
        None
        """
        self._config = AWSModelConfig(model_name="research_model", description="Converted research model")
        self._reward_scenario: Optional[ExperimentalScenario] = None
        self._reward_kwargs: Dict[str, Any] = {}
        self._model_files: Dict[str, Union[str, bytes]] = {}
        self._custom_reward_code: Optional[str] = None

    def with_name(self, name: str) -> "AWSModelBuilder":
        """Set the model name.

        Parameters
        ----------
        name : str
            The name for the AWS model

        Returns
        -------
        AWSModelBuilder
            Self for method chaining
        """
        self._config.model_name = name
        return self

    def with_description(self, description: str) -> "AWSModelBuilder":
        """Set the model description.

        Parameters
        ----------
        description : str
            The description for the AWS model

        Returns
        -------
        AWSModelBuilder
            Self for method chaining
        """
        self._config.description = description
        return self

    def with_version(self, version: str) -> "AWSModelBuilder":
        """Set the model version.

        Parameters
        ----------
        version : str
            The version string for the AWS model

        Returns
        -------
        AWSModelBuilder
            Self for method chaining
        """
        self._config.version = version
        return self

    def with_action_space(self, action_space_type: ActionSpaceType) -> "AWSModelBuilder":
        """Set the action space type.

        Parameters
        ----------
        action_space_type : ActionSpaceType
            The action space type ('continuous' or 'discrete')

        Returns
        -------
        AWSModelBuilder
            Self for method chaining

        Raises
        ------
        ValueError
            If action space type is invalid
        """
        if action_space_type not in [ActionSpaceType.CONTINUOUS, ActionSpaceType.DISCRETE]:
            raise ValueError("Action space type must be 'continuous' or 'discrete'")
        self._config.action_space_type = action_space_type
        if self._config.action_space_config:
            self._config.action_space_config.type = action_space_type
        else:
            self._config.action_space_config = ActionSpaceConfig(type=action_space_type)
        return self

    def with_sensor_config(self, sensor_config: Union[SensorConfig, Dict[str, Any]]) -> "AWSModelBuilder":
        """Set the sensor configuration.

        Parameters
        ----------
        sensor_config : Union[SensorConfig, Dict[str, Any]]
            Sensor configuration object or dictionary of sensor settings

        Returns
        -------
        AWSModelBuilder
            Self for method chaining

        Raises
        ------
        TypeError
            If sensor_config is not a SensorConfig instance or dictionary
        """
        if isinstance(sensor_config, dict):
            for key, value in sensor_config.items():
                if hasattr(self._config.sensor_config, key):
                    setattr(self._config.sensor_config, key, value)
        elif isinstance(sensor_config, SensorConfig):
            self._config.sensor_config = sensor_config
        else:
            raise TypeError("sensor_config must be SensorConfig instance or dictionary")
        return self

    def with_hyperparameters(self, hyperparameters: Union[AWSHyperparameters, Dict[str, Any]]) -> "AWSModelBuilder":
        """Set or update hyperparameters.

        Parameters
        ----------
        hyperparameters : Union[AWSHyperparameters, Dict[str, Any]]
            Hyperparameters object or dictionary of hyperparameter settings

        Returns
        -------
        AWSModelBuilder
            Self for method chaining

        Raises
        ------
        TypeError
            If hyperparameters is not an AWSHyperparameters instance or dictionary
        """
        if isinstance(hyperparameters, dict):
            for key, value in hyperparameters.items():
                if hasattr(self._config.hyperparameters, key):
                    setattr(self._config.hyperparameters, key, value)
        elif isinstance(hyperparameters, AWSHyperparameters):
            self._config.hyperparameters = hyperparameters
        else:
            raise TypeError("hyperparameters must be AWSHyperparameters instance or dictionary")
        return self

    def with_metadata(self, metadata: Union[AWSModelMetadata, Dict[str, Any]]) -> "AWSModelBuilder":
        """Set or update metadata.

        Parameters
        ----------
        metadata : Union[AWSModelMetadata, Dict[str, Any]]
            Metadata object or dictionary of metadata

        Returns
        -------
        AWSModelBuilder
            Self for method chaining

        Raises
        ------
        TypeError
            If metadata is not an AWSModelMetadata instance or dictionary
        """
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if hasattr(self._config.metadata, key):
                    setattr(self._config.metadata, key, value)
        elif isinstance(metadata, AWSModelMetadata):
            self._config.metadata = metadata
        else:
            raise TypeError("metadata must be AWSModelMetadata instance or dictionary")
        return self

    def with_action_space_config(self, action_space_config: Union[ActionSpaceConfig, Dict[str, Any]]) -> "AWSModelBuilder":
        """Set the action space configuration.

        Parameters
        ----------
        action_space_config : Union[ActionSpaceConfig, Dict[str, Any]]
            Action space configuration object or dictionary

        Returns
        -------
        AWSModelBuilder
            Self for method chaining

        Raises
        ------
        TypeError
            If action_space_config is not an ActionSpaceConfig instance or dictionary
        """
        if isinstance(action_space_config, dict):
            if self._config.action_space_config is None:
                self._config.action_space_config = ActionSpaceConfig()
            for key, value in action_space_config.items():
                if hasattr(self._config.action_space_config, key):
                    setattr(self._config.action_space_config, key, value)
        elif isinstance(action_space_config, ActionSpaceConfig):
            self._config.action_space_config = action_space_config
            self._config.action_space_type = action_space_config.type
        else:
            raise TypeError("action_space_config must be ActionSpaceConfig instance or dictionary")
        return self

    def with_reward_scenario(self, scenario: ExperimentalScenario, **kwargs) -> "AWSModelBuilder":
        """Set the reward function scenario.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The experimental scenario for reward function
        **kwargs
            Additional arguments for the reward function

        Returns
        -------
        AWSModelBuilder
            Self for method chaining
        """
        self._reward_scenario = scenario
        self._reward_kwargs = kwargs
        return self

    def with_custom_reward_code(self, reward_code: str) -> "AWSModelBuilder":
        """Set custom reward function code.

        Parameters
        ----------
        reward_code : str
            Custom AWS-compatible reward function code

        Returns
        -------
        AWSModelBuilder
            Self for method chaining
        """
        self._custom_reward_code = reward_code
        return self

    def with_model_file(self, filename: str, content: Union[str, bytes]) -> "AWSModelBuilder":
        """Add a model file to the AWS package.

        Parameters
        ----------
        filename : str
            The filename for the model file
        content : Union[str, bytes]
            The file content

        Returns
        -------
        AWSModelBuilder
            Self for method chaining
        """
        self._model_files[filename] = content
        return self

    def _generate_reward_function_code(self) -> str:
        """Generate AWS-compatible reward function code using RewardFunctionBuilder.

        Returns
        -------
        str
            AWS DeepRacer compatible reward function code
        """
        if self._custom_reward_code:
            return self._custom_reward_code

        if not self._reward_scenario:
            return self._get_default_reward_function_code()

        builder = RewardFunctionBuilder.create_for_scenario(self._reward_scenario, **self._reward_kwargs)

        return builder.with_optimization("advanced").build_function_code()

    def _get_default_reward_function_code(self) -> str:
        """Get default reward function code using the reward function builder.

        Returns
        -------
        str
            Default AWS DeepRacer reward function code using centerline following scenario
        """
        builder = RewardFunctionBuilder.create_for_scenario(ExperimentalScenario.CENTERLINE_FOLLOWING)

        return builder.with_optimization("basic").build_function_code()

    def _generate_model_metadata(self) -> Dict[str, Any]:
        """Generate model metadata for AWS deployment.

        Returns
        -------
        Dict[str, Any]
            Complete model metadata in new AWS format (version 5)
        """
        metadata = {
            "action_space": self._convert_action_space_to_aws_format(),
            "sensor": SensorType.get_recommended_for_time_trials().get_sensor_list(),
            "neural_network": NeuralNetworkType.get_recommended().value,
            "version": "5",
            "training_algorithm": TrainingAlgorithm.CLIPPED_PPO.value,
            "action_space_type": ActionSpaceType.get_recommended().value,
            "preprocess_type": "GREY_SCALE",
            "regional_parameters": [0, 0, 0, 0],
        }

        return metadata

    def _convert_action_space_to_aws_format(self) -> Dict[str, Any]:
        """Convert action space configuration to new AWS format.

        Returns
        -------
        Dict[str, Any]
            Action space configuration in new AWS format
        """
        max_steering = 30.0
        min_speed = 0.5
        max_speed = 1.0

        if self._config.action_space_config:
            if hasattr(self._config.action_space_config, "steering_range") and self._config.action_space_config.steering_range:
                max_steering = abs(self._config.action_space_config.steering_range.get("max", 30.0))
            if hasattr(self._config.action_space_config, "speed_range") and self._config.action_space_config.speed_range:
                min_speed = self._config.action_space_config.speed_range.get("min", 0.5)
                max_speed = self._config.action_space_config.speed_range.get("max", 1.0)

        return {
            "steering_angle": {"high": float(max_steering), "low": float(-max_steering)},
            "speed": {"high": float(max_speed), "low": float(min_speed)},
        }

    def _validate_configuration(self) -> None:
        """Validate the current configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the configuration is invalid
        """
        if not self._config.model_name:
            raise ValueError("Model name is required")

        if not self._config.description:
            raise ValueError("Model description is required")

        if self._config.action_space_type not in [ActionSpaceType.CONTINUOUS.value, ActionSpaceType.DISCRETE.value]:
            raise ValueError(
                f"Invalid action space type: {self._config.action_space_type}. Must be one of: {[t.value for t in ActionSpaceType]}"
            )

        if self._config.sensor_config and hasattr(self._config.sensor_config, "validate"):
            if not self._config.sensor_config.validate():
                raise ValueError("Invalid sensor configuration - at least one camera must be enabled")

        if self._config.hyperparameters and hasattr(self._config.hyperparameters, "validate"):
            if not self._config.hyperparameters.validate():
                raise ValueError("Invalid hyperparameters configuration")

        if self._config.action_space_config and hasattr(self._config.action_space_config, "validate"):
            if not self._config.action_space_config.validate():
                raise ValueError("Invalid action space configuration")

        info("Configuration validation passed")

    def build(self) -> "AWSDeepRacerModel":
        """Build the AWS DeepRacer model.

        Returns
        -------
        AWSDeepRacerModel
            The built AWS DeepRacer model

        Raises
        ------
        ValueError
            If the configuration is invalid
        """
        self._validate_configuration()

        reward_code = self._generate_reward_function_code()
        metadata = self._generate_model_metadata()

        return AWSDeepRacerModel(
            config=self._config, reward_function_code=reward_code, metadata=metadata, model_files=self._model_files.copy()
        )
