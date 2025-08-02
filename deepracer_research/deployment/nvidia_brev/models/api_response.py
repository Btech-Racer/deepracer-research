from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from deepracer_research.deployment.nvidia_brev.models.nvidia_brev_error import NvidiaBrevError


@dataclass
class ApiResponse:
    """Generic API response wrapper

    Parameters
    ----------
    success : bool
        Whether the request was successful
    data : Any, optional
        Response data, by default None
    message : str, optional
        Response message, by default ""
    error_code : str, optional
        Error code if request failed, by default None
    status_code : int, optional
        HTTP status code, by default None
    request_id : str, optional
        Unique request identifier, by default None
    metadata : Dict[str, Any], optional
        Additional metadata, by default empty dict
    """

    success: bool
    data: Any = None
    message: str = ""
    error_code: Optional[str] = None
    status_code: Optional[int] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_error(self) -> bool:
        """Check if response contains an error.

        Returns
        -------
        bool
            True if response has an error
        """
        return not self.success or bool(self.error_code)

    def raise_for_status(self) -> None:
        """Raise exception if response contains an error.

        Raises
        ------
        NvidiaBrevError
            If response contains an error
        """
        if self.has_error:
            raise NvidiaBrevError(
                message=self.message or "API request failed",
                error_code=self.error_code,
                status_code=self.status_code,
                request_id=self.request_id,
                details=self.metadata,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of response
        """
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }
