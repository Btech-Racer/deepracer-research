from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class NvidiaBrevConfig:
    """NVIDIA Brev API configuration

    Parameters
    ----------
    api_token : str
        Bearer token for NVIDIA Brev API authentication.
    organization_id : str, optional
        Organization ID for team/enterprise accounts, by default None.
    base_url : str, optional
        Base URL for NVIDIA Brev API, by default "https://api.brev.dev".
    api_version : str, optional
        API version to use, by default "v1".
    timeout : int, optional
        Request timeout in seconds, by default 30.
    verify_ssl : bool, optional
        Whether to verify SSL certificates, by default True.
    """

    api_token: str
    organization_id: Optional[str] = None
    base_url: str = "https://api.brev.dev"
    api_version: str = "v1"
    timeout: int = 30
    verify_ssl: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {
            "api_token": self.api_token,
            "organization_id": self.organization_id,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl,
        }

    @property
    def api_url(self) -> str:
        """Get the full API URL.

        Returns
        -------
        str
            Full API URL including version
        """
        return f"{self.base_url}/{self.api_version}"

    @property
    def headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests.

        Returns
        -------
        Dict[str, str]
            HTTP headers including authorization
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "deepracer-research/1.0",
        }

        if self.organization_id:
            headers["X-Organization-ID"] = self.organization_id

        return headers

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

        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")

        if self.organization_id is not None and not self.organization_id.strip():
            raise ValueError("Organization ID cannot be empty if provided")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvidiaBrevConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration parameters.

        Returns
        -------
        NvidiaBrevConfig
            Configuration instance created from dictionary.
        """
        return cls(
            api_token=data["api_token"],
            organization_id=data.get("organization_id"),
            base_url=data.get("base_url", "https://api.brev.dev"),
            api_version=data.get("api_version", "v1"),
            timeout=data.get("timeout", 30),
            verify_ssl=data.get("verify_ssl", True),
        )

    @classmethod
    def from_environment(cls, org_id: Optional[str] = None) -> "NvidiaBrevConfig":
        """Create configuration from environment variables


        Parameters
        ----------
        org_id : Optional[str], optional
            Override organization ID, by default None.

        Returns
        -------
        NvidiaBrevConfig
            Configuration instance created from environment.

        Raises
        ------
        ValueError
            If required environment variables are missing.
        """
        import os

        api_token = os.getenv("NVIDIA_BREV_API_TOKEN")
        if not api_token:
            raise ValueError("NVIDIA_BREV_API_TOKEN environment variable is required")

        return cls(
            api_token=api_token,
            organization_id=org_id or os.getenv("NVIDIA_BREV_ORG_ID"),
            base_url=os.getenv("NVIDIA_BREV_BASE_URL", "https://api.brev.dev"),
            api_version=os.getenv("NVIDIA_BREV_API_VERSION", "v1"),
            timeout=int(os.getenv("NVIDIA_BREV_TIMEOUT", "30")),
            verify_ssl=os.getenv("NVIDIA_BREV_VERIFY_SSL", "true").lower() == "true",
        )
