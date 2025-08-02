from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from deepracer_research.config.network.architecture_type import ArchitectureType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.models.deployment_status import DeploymentStatus, DeploymentType
from deepracer_research.rewards.reward_function_type import RewardFunctionType


@dataclass
class ModelMetadata:
    """Comprehensive model metadata for research tracking

    Parameters
    ----------
    model_name : str
        Model name
    model_id : str, optional
        Unique model identifier, by default ""
    version : str, optional
        Model version string, by default "1.0.0"
    created_date : datetime, optional
        Model creation timestamp, by default current datetime
    last_modified : datetime, optional
        Last modification timestamp, by default current datetime
    algorithm : TrainingAlgorithm, optional
        Training algorithm (PPO, SAC), by default TrainingAlgorithm.PPO
    neural_architecture : ArchitectureType, optional
        Network architecture type, by default ArchitectureType.ATTENTION_CNN
    reward_function : RewardFunctionType, optional
        Reward function type, by default RewardFunctionType.DEFAULT
    hyperparameters : Dict[str, Any], optional
        Training hyperparameters, by default empty dict
    training_episodes : int, optional
        Number of training episodes, by default 0
    training_duration_hours : float, optional
        Total training time, by default 0.0
    completion_rate : float, optional
        Track completion rate percentage, by default 0.0
    best_lap_time : float, optional
        Best lap time achieved, by default inf
    average_speed : float, optional
        Average speed during evaluation, by default 0.0
    convergence_episode : int, optional
        Episode where training converged, by default -1
    model_files : List[str], optional
        List of model file paths, by default empty list
    model_size_mb : float, optional
        Model size in megabytes, by default 0.0
    checksum : str, optional
        Model file checksum for integrity, by default ""
    experiment_id : str, optional
        Associated experiment identifier, by default ""
    scenario : str, optional
        Training scenario name, by default ""
    track_name : str, optional
        Track used for training, by default ""
    notes : str, optional
        Additional notes and comments, by default ""
    tags : List[str], optional
        Model tags for categorization, by default empty list
    s3_location : str, optional
        AWS S3 storage location, by default ""
    deepracer_model_arn : str, optional
        AWS DeepRacer model ARN, by default ""
    deployment_status : DeploymentStatus, optional
        Current deployment status, by default DeploymentStatus.LOCAL
    """

    model_name: str
    model_id: str = ""
    version: str = "1.0.0"
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)

    algorithm: TrainingAlgorithm = TrainingAlgorithm.PPO
    neural_architecture: ArchitectureType = ArchitectureType.ATTENTION_CNN
    reward_function: RewardFunctionType = RewardFunctionType.DEFAULT
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_episodes: int = 0
    training_duration_hours: float = 0.0

    completion_rate: float = 0.0
    best_lap_time: float = float("inf")
    average_speed: float = 0.0
    convergence_episode: int = -1

    model_files: List[str] = field(default_factory=list)
    model_size_mb: float = 0.0
    checksum: str = ""

    experiment_id: str = ""
    scenario: str = ""
    track_name: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    s3_location: str = ""
    deepracer_model_arn: str = ""
    deployment_status: DeploymentStatus = DeploymentStatus.LOCAL

    def update_last_modified(self) -> None:
        """Update the last modified timestamp to current time.

        Returns
        -------
        None
        """
        self.last_modified = datetime.now()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the model if not already present.

        Parameters
        ----------
        tag : str
            Tag to add to the model

        Returns
        -------
        None
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self.update_last_modified()

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the model.

        Parameters
        ----------
        tag : str
            Tag to remove from the model

        Returns
        -------
        bool
            True if tag was removed, False if tag was not found
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self.update_last_modified()
            return True
        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of model performance metrics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing performance metrics including completion_rate,
            best_lap_time, average_speed, and convergence_episode
        """
        return {
            "completion_rate": self.completion_rate,
            "best_lap_time": self.best_lap_time,
            "average_speed": self.average_speed,
            "convergence_episode": self.convergence_episode,
            "training_episodes": self.training_episodes,
            "training_duration_hours": self.training_duration_hours,
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training configuration and parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing training configuration including algorithm,
            architecture, reward function, and hyperparameters
        """
        return {
            "algorithm": self.algorithm.value,
            "neural_architecture": self.neural_architecture.value,
            "reward_function": self.reward_function.value,
            "hyperparameters": self.hyperparameters.copy(),
            "training_episodes": self.training_episodes,
            "training_duration_hours": self.training_duration_hours,
        }

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment-related information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing deployment status, locations, and identifiers
        """
        return {
            "deployment_status": self.deployment_status.value,
            "deployment_status_description": self.deployment_status.get_description(),
            "is_cloud_deployed": self.deployment_status.is_cloud_based(),
            "is_active": self.deployment_status.is_active(),
            "s3_location": self.s3_location,
            "deepracer_model_arn": self.deepracer_model_arn,
        }

    @property
    def deployment_type(self) -> DeploymentType:
        """Get the deployment type based on deployment status.

        Returns
        -------
        DeploymentType
            The deployment type (LOCAL or CLOUD) derived from deployment status
        """
        return DeploymentType.from_status(self.deployment_status)

    def is_ready_for_deployment(self) -> bool:
        """Check if the model is ready for cloud deployment.

        Returns
        -------
        bool
            True if the model has sufficient training and is in a deployable state
        """
        return (
            self.training_episodes > 0
            and self.completion_rate > 0.0
            and len(self.model_files) > 0
            and self.deployment_status == DeploymentStatus.LOCAL
        )

    def is_template_based_reward(self) -> bool:
        """Check if the reward function uses YAML templates.

        Returns
        -------
        bool
            True if the reward function is template-based, False otherwise
        """
        return self.reward_function.is_template_based()

    def get_model_info_dict(self) -> Dict[str, Any]:
        """Get complete model information as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Complete model metadata as a dictionary with enum values converted to strings
        """
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "version": self.version,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "algorithm": self.algorithm.value,
            "neural_architecture": self.neural_architecture.value,
            "reward_function": self.reward_function.value,
            "hyperparameters": self.hyperparameters,
            "training_episodes": self.training_episodes,
            "training_duration_hours": self.training_duration_hours,
            "completion_rate": self.completion_rate,
            "best_lap_time": self.best_lap_time,
            "average_speed": self.average_speed,
            "convergence_episode": self.convergence_episode,
            "model_files": self.model_files,
            "model_size_mb": self.model_size_mb,
            "checksum": self.checksum,
            "experiment_id": self.experiment_id,
            "scenario": self.scenario,
            "track_name": self.track_name,
            "notes": self.notes,
            "tags": self.tags,
            "s3_location": self.s3_location,
            "deepracer_model_arn": self.deepracer_model_arn,
            "deployment_status": self.deployment_status.value,
        }
