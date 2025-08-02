from dataclasses import dataclass
from typing import Optional

from deepracer_research.config.aws_config import (
    ActionSpaceConfig,
    ActionSpaceType,
    AWSHyperparameters,
    AWSModelMetadata,
    SensorConfig,
)


@dataclass
class AWSModelConfig:
    """Configuration for AWS DeepRacer model conversion

    Parameters
    ----------
    model_name : str
        Name of the model
    description : str
        Description of the model
    version : str, optional
        Model version, by default "1.0.0"
    action_space_type : ActionSpaceType, optional
        Type of action space, by default "continuous"
    sensor_config : Optional[SensorConfig], optional
        Sensor configuration, by default None
    hyperparameters : Optional[AWSHyperparameters], optional
        Training hyperparameters, by default None
    metadata : Optional[AWSModelMetadata], optional
        Model metadata, by default None
    action_space_config : Optional[ActionSpaceConfig], optional
        Action space configuration, by default None
    """

    model_name: str
    description: str
    version: str = "1.0.0"
    action_space_type: ActionSpaceType = ActionSpaceType.get_recommended()
    sensor_config: Optional[SensorConfig] = None
    hyperparameters: Optional[AWSHyperparameters] = None
    metadata: Optional[AWSModelMetadata] = None
    action_space_config: Optional[ActionSpaceConfig] = None

    def __post_init__(self):
        """Initialize default configurations after object creation.

        Returns
        -------
        None
        """
        if self.sensor_config is None:
            self.sensor_config = SensorConfig()

        if self.hyperparameters is None:
            self.hyperparameters = AWSHyperparameters()

        if self.metadata is None:
            self.metadata = AWSModelMetadata()

        if self.action_space_config is None:
            self.action_space_config = ActionSpaceConfig(type=self.action_space_type)

    def validate_configuration(self) -> bool:
        """Validate the AWS model configuration.

        Returns
        -------
        bool
            True if configuration is valid, False otherwise
        """
        if not self.model_name.strip():
            return False

        if not self.description.strip():
            return False

        if self.sensor_config is None:
            return False

        if self.hyperparameters is None:
            return False

        if self.metadata is None:
            return False

        if self.action_space_config is None:
            return False

        return True

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration
        """
        return {
            "model_name": self.model_name,
            "description": self.description,
            "version": self.version,
            "action_space_type": self.action_space_type,
            "sensor_config": self.sensor_config.__dict__ if self.sensor_config else None,
            "hyperparameters": self.hyperparameters.__dict__ if self.hyperparameters else None,
            "metadata": self.metadata.__dict__ if self.metadata else None,
            "action_space_config": self.action_space_config.__dict__ if self.action_space_config else None,
        }
