from enum import Enum, unique


@unique
class DeploymentStatus(str, Enum):
    """Deployment status for AWS DeepRacer models"""

    LOCAL = "local"
    UPLOADED = "uploaded"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"
    PENDING = "pending"
    VALIDATING = "validating"
    RETIRED = "retired"

    @classmethod
    def get_active_statuses(cls) -> list:
        """Get deployment statuses representing active/usable models.

        Returns
        -------
        list
            List of DeploymentStatus values for active models
        """
        return [cls.LOCAL, cls.DEPLOYED]

    @classmethod
    def get_cloud_statuses(cls) -> list:
        """Get deployment statuses representing cloud-based models.

        Returns
        -------
        list
            List of DeploymentStatus values for cloud-deployed models
        """
        return [cls.UPLOADED, cls.DEPLOYED, cls.ARCHIVED, cls.VALIDATING]

    @classmethod
    def get_transitional_statuses(cls) -> list:
        """Get deployment statuses representing temporary states.

        Returns
        -------
        list
            List of DeploymentStatus values for transitional states
        """
        return [cls.PENDING, cls.VALIDATING]

    @classmethod
    def get_error_statuses(cls) -> list:
        """Get deployment statuses representing error states.

        Returns
        -------
        list
            List of DeploymentStatus values for error conditions
        """
        return [cls.FAILED]

    def is_active(self) -> bool:
        """Check if this status represents an active/usable model.

        Returns
        -------
        bool
            True if the model is active and usable, False otherwise
        """
        return self in self.get_active_statuses()

    def is_cloud_based(self) -> bool:
        """Check if this status represents a cloud-deployed model.

        Returns
        -------
        bool
            True if the model is in cloud storage/deployment, False otherwise
        """
        return self in self.get_cloud_statuses()

    def is_transitional(self) -> bool:
        """Check if this status represents a temporary state.

        Returns
        -------
        bool
            True if the status is transitional, False otherwise
        """
        return self in self.get_transitional_statuses()

    def is_error_state(self) -> bool:
        """Check if this status represents an error condition.

        Returns
        -------
        bool
            True if the status indicates an error, False otherwise
        """
        return self in self.get_error_statuses()

    def get_description(self) -> str:
        """Get a  description of the deployment status.

        Returns
        -------
        str
            Description of the deployment status
        """
        descriptions = {
            self.LOCAL: "Available only in local development environment",
            self.UPLOADED: "Uploaded to cloud storage but not yet deployed",
            self.DEPLOYED: "Successfully deployed and available for racing",
            self.ARCHIVED: "Archived for long-term storage",
            self.FAILED: "Deployment or upload failed",
            self.PENDING: "Deployment operation in progress",
            self.VALIDATING: "Model validation in progress",
            self.RETIRED: "Retired from active use",
        }
        return descriptions.get(self, "Unknown deployment status")


@unique
class DeploymentType(str, Enum):
    """Deployment type for AWS DeepRacer models."""

    LOCAL = "local"
    CLOUD = "cloud"

    @classmethod
    def from_status(cls, status: DeploymentStatus) -> "DeploymentType":
        """Determine deployment type from deployment status.

        Parameters
        ----------
        status : DeploymentStatus
            The deployment status to analyze

        Returns
        -------
        DeploymentType
            The corresponding deployment type
        """
        if status == DeploymentStatus.LOCAL:
            return cls.LOCAL
        elif status in DeploymentStatus.get_cloud_statuses():
            return cls.CLOUD
        else:
            return cls.LOCAL

    def get_compatible_statuses(self) -> list:
        """Get deployment statuses compatible with this deployment type.

        Returns
        -------
        list
            List of DeploymentStatus values compatible with this type
        """
        if self == DeploymentType.LOCAL:
            return [DeploymentStatus.LOCAL]
        else:
            return DeploymentStatus.get_cloud_statuses()

    def is_local(self) -> bool:
        """Check if this is a local deployment type.

        Returns
        -------
        bool
            True if local deployment, False otherwise
        """
        return self == DeploymentType.LOCAL

    def is_cloud(self) -> bool:
        """Check if this is a cloud deployment type.

        Returns
        -------
        bool
            True if cloud deployment, False otherwise
        """
        return self == DeploymentType.CLOUD

    def get_description(self) -> str:
        """Get a  description of the deployment type.

        Returns
        -------
        str
            Description of the deployment type
        """
        descriptions = {self.LOCAL: "Local development environment deployment", self.CLOUD: "AWS cloud-based deployment"}
        return descriptions.get(self, "Unknown deployment type")

    def get_supported_features(self) -> list:
        """Get features supported by this deployment type.

        Returns
        -------
        list
            List of supported features for this deployment type
        """
        if self == DeploymentType.LOCAL:
            return ["local_training", "local_evaluation", "development_testing", "model_debugging", "fast_iteration"]
        else:
            return [
                "scalable_training",
                "production_deployment",
                "competition_ready",
                "managed_infrastructure",
                "s3_storage",
                "sagemaker_integration",
            ]
