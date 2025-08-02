from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ThunderComputeConfig:
    """Thunder Compute API configuration

    Parameters
    ----------
    api_token : str
        Bearer token for Thunder Compute API authentication.
    base_url : str, optional
        Base URL for Thunder Compute API, by default "https://api.thundercompute.com:8443".
    """

    api_token: str
    base_url: str = "https://api.thundercompute.com:8443"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {"api_token": self.api_token, "base_url": self.base_url}

    def validate(self) -> None:
        """Validate the configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if not self.api_token:
            raise ValueError("API token is required")

        if not self.api_token.strip():
            raise ValueError("API token cannot be empty or whitespace")

        if not self.base_url:
            raise ValueError("Base URL is required")

        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThunderComputeConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration parameters.

        Returns
        -------
        ThunderComputeConfig
            Configuration instance created from dictionary.
        """
        return cls(api_token=data["api_token"], base_url=data.get("base_url", "https://api.thundercompute.com:8443"))
