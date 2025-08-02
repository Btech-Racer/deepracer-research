from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DeepRacerCloudConfig:
    """Configuration for DeepRacer-for-Cloud installation

    Parameters
    ----------
    git_url : str, optional
        Git repository URL for DeepRacer-for-Cloud, by default official repo.
    branch : str, optional
        Git branch to checkout, by default "main".
    cloud_mode : str, optional
        DeepRacer cloud mode (local, aws, azure), by default "local".
    architecture : str, optional
        Target architecture (gpu, cpu), by default "gpu".
    aws_region : str, optional
        AWS region for S3 operations, by default "us-east-1".
    s3_bucket : str, optional
        S3 bucket for model storage, by default None.
    install_path : str, optional
        Installation path on the instance, by default "/home/ubuntu/deepracer-for-cloud".
    """

    git_url: str = "https://github.com/aws-deepracer-community/deepracer-for-cloud.git"
    branch: str = "main"
    cloud_mode: str = "local"
    architecture: str = "gpu"

    aws_region: str = "us-east-1"
    s3_bucket: Optional[str] = None

    install_path: str = "/home/ubuntu/deepracer-for-cloud"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {
            "git_url": self.git_url,
            "branch": self.branch,
            "cloud_mode": self.cloud_mode,
            "architecture": self.architecture,
            "aws_region": self.aws_region,
            "s3_bucket": self.s3_bucket,
            "install_path": self.install_path,
        }

    def validate(self) -> None:
        """Validate the configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if not self.git_url:
            raise ValueError("Git URL is required")

        if not self.git_url.startswith(("http://", "https://", "git@")):
            raise ValueError("Git URL must be a valid URL")

        if self.cloud_mode not in ["local", "aws", "azure"]:
            raise ValueError("Cloud mode must be 'local', 'aws', or 'azure'")

        if self.architecture not in ["gpu", "cpu"]:
            raise ValueError("Architecture must be 'gpu' or 'cpu'")

        if not self.install_path:
            raise ValueError("Install path is required")

        if not self.install_path.startswith("/"):
            raise ValueError("Install path must be absolute")

    @classmethod
    def for_local_training(cls, architecture: str = "gpu", s3_bucket: Optional[str] = None) -> "DeepRacerCloudConfig":
        """Create configuration for local training mode.

        Parameters
        ----------
        architecture : str, optional
            Target architecture (gpu or cpu), by default "gpu".
        s3_bucket : str, optional
            S3 bucket for model storage, by default None.

        Returns
        -------
        DeepRacerCloudConfig
            Configuration optimized for local training.
        """
        return cls(cloud_mode="local", architecture=architecture, s3_bucket=s3_bucket)

    @classmethod
    def for_aws_training(
        cls, s3_bucket: str, aws_region: str = "us-east-1", architecture: str = "gpu"
    ) -> "DeepRacerCloudConfig":
        """Create configuration for AWS training mode.

        Parameters
        ----------
        s3_bucket : str
            S3 bucket for model storage.
        aws_region : str, optional
            AWS region, by default "us-east-1".
        architecture : str, optional
            Target architecture (gpu or cpu), by default "gpu".

        Returns
        -------
        DeepRacerCloudConfig
            Configuration optimized for AWS training.
        """
        return cls(cloud_mode="aws", architecture=architecture, aws_region=aws_region, s3_bucket=s3_bucket)

    @classmethod
    def for_development(cls) -> "DeepRacerCloudConfig":
        """Create configuration for development and testing.

        Returns
        -------
        DeepRacerCloudConfig
            Configuration optimized for development work.
        """
        return cls(cloud_mode="local", architecture="cpu", branch="main")  # CPU for faster setup during development

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepRacerCloudConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration parameters.

        Returns
        -------
        DeepRacerCloudConfig
            Configuration instance created from dictionary.
        """
        return cls(
            git_url=data.get("git_url", "https://github.com/aws-deepracer-community/deepracer-for-cloud.git"),
            branch=data.get("branch", "main"),
            cloud_mode=data.get("cloud_mode", "local"),
            architecture=data.get("architecture", "gpu"),
            aws_region=data.get("aws_region", "us-east-1"),
            s3_bucket=data.get("s3_bucket"),
            install_path=data.get("install_path", "/home/ubuntu/deepracer-for-cloud"),
        )
