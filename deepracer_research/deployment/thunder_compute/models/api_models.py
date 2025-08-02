from typing import Optional


class ThunderComputeError(Exception):
    """Thunder Compute API error

    Parameters
    ----------
    message : str
        Error message describing the failure.
    status_code : int, optional
        HTTP status code from the API response, by default None.
    response_data : dict, optional
        Raw response data from the API, by default None.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}

    @property
    def is_client_error(self) -> bool:
        """Check if error is a client error (4xx status code).

        Returns
        -------
        bool
            True if status code indicates a client error.
        """
        return self.status_code is not None and 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if error is a server error (5xx status code).

        Returns
        -------
        bool
            True if status code indicates a server error.
        """
        return self.status_code is not None and 500 <= self.status_code < 600

    @property
    def is_authentication_error(self) -> bool:
        """Check if error is an authentication error (401 status code).

        Returns
        -------
        bool
            True if status code indicates authentication failure.
        """
        return self.status_code == 401

    @property
    def is_authorization_error(self) -> bool:
        """Check if error is an authorization error (403 status code).

        Returns
        -------
        bool
            True if status code indicates authorization failure.
        """
        return self.status_code == 403

    @property
    def is_not_found_error(self) -> bool:
        """Check if error is a not found error (404 status code).

        Returns
        -------
        bool
            True if status code indicates resource not found.
        """
        return self.status_code == 404

    @property
    def is_validation_error(self) -> bool:
        """Check if error is a validation error (422 status code).

        Returns
        -------
        bool
            True if status code indicates validation failure.
        """
        return self.status_code == 422

    @property
    def is_rate_limit_error(self) -> bool:
        """Check if error is a rate limit error (429 status code).

        Returns
        -------
        bool
            True if status code indicates rate limiting.
        """
        return self.status_code == 429

    def get_retry_after(self) -> Optional[int]:
        """Get retry-after value from response headers.

        Returns
        -------
        int, optional
            Number of seconds to wait before retrying, if available.
        """
        if "retry_after" in self.response_data:
            try:
                return int(self.response_data["retry_after"])
            except (ValueError, TypeError):
                pass
        return None

    def __str__(self) -> str:
        """Get string representation of the error.

        Returns
        -------
        str
            Formatted error message including status code if available.
        """
        if self.status_code:
            return f"Thunder Compute API Error (HTTP {self.status_code}): {super().__str__()}"
        return f"Thunder Compute API Error: {super().__str__()}"
