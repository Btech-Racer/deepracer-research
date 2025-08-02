import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion


@dataclass
class AWSConfig:
    """AWS configuration for EC2 deployments

    Parameters
    ----------
    region : AWSRegion, optional
        AWS region for deployments, by default AWSRegion.US_EAST_1.
    profile_name : str, optional
        AWS profile name to use, by default "default".
    access_key_id : str, optional
        AWS access key ID, by default None (uses environment or profile).
    secret_access_key : str, optional
        AWS secret access key, by default None (uses environment or profile).
    session_token : str, optional
        AWS session token for temporary credentials, by default None.
    """

    region: AWSRegion = AWSRegion.US_EAST_1
    profile_name: str = "default"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {
            "region": self.region.value,
            "profile_name": self.profile_name,
            "access_key_id": self.access_key_id,
            "secret_access_key": self.secret_access_key,
            "session_token": self.session_token,
        }

    def get_boto3_session_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for creating boto3 session.

        Returns
        -------
        Dict[str, Any]
            Dictionary of kwargs for boto3.Session().
        """
        kwargs = {"region_name": self.region.value}

        if self.profile_name != "default":
            kwargs["profile_name"] = self.profile_name

        if self.access_key_id:
            kwargs["aws_access_key_id"] = self.access_key_id

        if self.secret_access_key:
            kwargs["aws_secret_access_key"] = self.secret_access_key

        if self.session_token:
            kwargs["aws_session_token"] = self.session_token

        return kwargs

    def validate(self) -> None:
        """Validate the configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if not self.profile_name:
            raise ValueError("Profile name cannot be empty")

        if self.access_key_id and not self.secret_access_key:
            raise ValueError("Secret access key required when access key ID is provided")

        if self.secret_access_key and not self.access_key_id:
            raise ValueError("Access key ID required when secret access key is provided")

    @classmethod
    def from_environment(cls, region: Optional[AWSRegion] = None) -> "AWSConfig":
        """Create configuration from environment variables.

        Parameters
        ----------
        region : AWSRegion, optional
            AWS region override, by default uses AWS_DEFAULT_REGION or US_EAST_1.

        Returns
        -------
        AWSConfig
            Configuration instance created from environment.
        """
        env_region = os.getenv("AWS_DEFAULT_REGION", AWSRegion.US_EAST_1.value)
        if region is None:
            try:
                region = AWSRegion(env_region)
            except ValueError:
                region = AWSRegion.US_EAST_1

        return cls(
            region=region,
            profile_name=os.getenv("AWS_PROFILE", "default"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AWSConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration parameters.

        Returns
        -------
        AWSConfig
            Configuration instance created from dictionary.
        """
        region_str = data.get("region", AWSRegion.US_EAST_1.value)
        try:
            region = AWSRegion(region_str)
        except ValueError:
            region = AWSRegion.US_EAST_1

        return cls(
            region=region,
            profile_name=data.get("profile_name", "default"),
            access_key_id=data.get("access_key_id"),
            secret_access_key=data.get("secret_access_key"),
            session_token=data.get("session_token"),
        )
