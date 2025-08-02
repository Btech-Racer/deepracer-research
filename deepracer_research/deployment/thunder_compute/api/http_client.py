from typing import Any, Dict, Optional

import requests

from deepracer_research.deployment.thunder_compute.models.api_models import ThunderComputeError
from deepracer_research.utils import debug, error


class HTTPClient:
    """HTTP client for Thunder Compute API requests

    Parameters
    ----------
    api_token : str
        Bearer token for Thunder Compute API authentication.
    base_url : str
        Base URL for Thunder Compute API endpoints.
    timeout : int, optional
        Request timeout in seconds, by default 30.
    """

    def __init__(self, api_token: str, base_url: str, timeout: int = 30):
        """Initialize HTTP client.

        Parameters
        ----------
        api_token : str
            Bearer token for authentication.
        base_url : str
            Base URL for API endpoints.
        timeout : int, optional
            Request timeout in seconds, by default 30.
        """
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "User-Agent": "deepracer-research-thunder-client/1.0",
            }
        )

        debug("HTTP client initialized", extra={"base_url": self.base_url, "timeout": self.timeout})

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request to API endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint path (without base URL).
        params : Dict[str, Any], optional
            Query parameters to include in request, by default None.

        Returns
        -------
        Dict[str, Any]
            JSON response data from the API.

        Raises
        ------
        ThunderComputeError
            If request fails or returns error status.
        """
        return self._make_request("GET", endpoint, params=params)

    def post(
        self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make POST request to API endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint path (without base URL).
        json_data : Dict[str, Any], optional
            JSON data to send in request body, by default None.
        params : Dict[str, Any], optional
            Query parameters to include in request, by default None.

        Returns
        -------
        Dict[str, Any]
            JSON response data from the API.

        Raises
        ------
        ThunderComputeError
            If request fails or returns error status.
        """
        return self._make_request("POST", endpoint, json=json_data, params=params)

    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make DELETE request to API endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint path (without base URL).
        params : Dict[str, Any], optional
            Query parameters to include in request, by default None.

        Returns
        -------
        Dict[str, Any]
            JSON response data from the API.

        Raises
        ------
        ThunderComputeError
            If request fails or returns error status.
        """
        return self._make_request("DELETE", endpoint, params=params)

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, DELETE, etc.).
        endpoint : str
            API endpoint path.
        **kwargs
            Additional arguments to pass to requests.

        Returns
        -------
        Dict[str, Any]
            JSON response data from the API.

        Raises
        ------
        ThunderComputeError
            If request fails or returns error status.
        """
        url = f"{self.base_url}{endpoint}"

        debug(f"Making {method} request", extra={"url": url, "method": method, "timeout": self.timeout})

        try:
            response = self.session.request(method=method, url=url, timeout=self.timeout, **kwargs)

            debug(
                f"Received response",
                extra={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "response_size": len(response.content),
                },
            )

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as e:
                    error(
                        "Failed to parse JSON response",
                        extra={"status_code": response.status_code, "response_text": response.text[:500], "error": str(e)},
                    )
                    raise ThunderComputeError(f"Invalid JSON response: {str(e)}", status_code=response.status_code)
            elif response.status_code == 204:
                debug("Received 204 No Content - operation successful")
                return {}

            self._handle_error_response(response)

        except requests.RequestException as e:
            error("HTTP request failed", extra={"url": url, "method": method, "error": str(e)})
            raise ThunderComputeError(f"Network error: {str(e)}")

    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from the API.

        Parameters
        ----------
        response : requests.Response
            HTTP response object with error status.

        Raises
        ------
        ThunderComputeError
            Always raises with appropriate error message.
        """
        try:
            error_data = response.json() if response.content else {}
        except ValueError:
            error_data = {"error": response.text}

        error(
            "API request failed", extra={"status_code": response.status_code, "response_data": error_data, "url": response.url}
        )

        if response.status_code == 401:
            raise ThunderComputeError(
                "Authentication failed. Please check your API token.", status_code=401, response_data=error_data
            )
        elif response.status_code == 403:
            raise ThunderComputeError("Access forbidden. Check your permissions.", status_code=403, response_data=error_data)
        elif response.status_code == 404:
            raise ThunderComputeError("Resource not found.", status_code=404, response_data=error_data)
        elif response.status_code == 422:
            message = error_data.get("message", "Validation error")
            raise ThunderComputeError(f"Validation error: {message}", status_code=422, response_data=error_data)
        elif response.status_code == 429:
            retry_after = error_data.get("retry_after", 60)
            raise ThunderComputeError(
                f"Rate limit exceeded. Retry after {retry_after} seconds.", status_code=429, response_data=error_data
            )
        elif 500 <= response.status_code < 600:
            raise ThunderComputeError(
                f"Server error: {error_data.get('message', 'Internal server error')}",
                status_code=response.status_code,
                response_data=error_data,
            )
        else:
            raise ThunderComputeError(
                f"Request failed: {error_data.get('message', response.text)}",
                status_code=response.status_code,
                response_data=error_data,
            )

    def close(self) -> None:
        """Close the HTTP session and clean up resources."""
        self.session.close()
        debug("HTTP client session closed")

    def __enter__(self) -> "HTTPClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
