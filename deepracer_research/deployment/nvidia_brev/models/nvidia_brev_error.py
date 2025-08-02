from typing import Any, Dict, Optional


class NvidiaBrevError(Exception):
    """Custom exception for NVIDIA Brev API errors

    Parameters
    ----------
    message : str
        Error message
    error_code : str, optional
        Error code from API, by default None
    status_code : int, optional
        HTTP status code, by default None
    request_id : str, optional
        Request ID for debugging, by default None
    details : Dict[str, Any], optional
        Additional error details, by default None
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.request_id = request_id
        self.details = details or {}

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")

        if self.status_code:
            parts.append(f"Status: {self.status_code}")

        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of error
        """
        return {
            "message": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "details": self.details,
        }
