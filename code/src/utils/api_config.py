# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides API configuration classes for defining and managing API endpoints and clients."""
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from urllib.parse import urljoin
from urllib.parse import urlparse

import requests
import urllib3
from pydantic import BaseModel
from pydantic import Field
from requests.adapters import HTTPAdapter
from requests.adapters import Retry
from src.utils.status import Status

APIAuthenticationMethod = Literal["Bearer Token"]


class EndPoint(BaseModel):
    """
    EndPoint represents the configuration for an API endpoint.
    Attributes:
        path (Optional[str]): Path of the API endpoint.
        method (Literal["GET", "POST", "PUT", "DELETE"]): HTTP method for the API endpoint. Defaults to "GET".
        headers (Optional[Dict[str, str]]): Headers to include in the API request. Defaults to an empty dictionary.
        query_params (Optional[Dict[str, str]]): Query parameters for the API request. Defaults to an empty dictionary.
        body (Optional[Any]): Body of the API request for POST/PUT methods. Defaults to None.
    """

    path: Optional[str] = Field(None, description="Path of the API endpoint")
    method: Literal["GET", "POST", "PUT", "DELETE"] = Field(
        "GET", description="HTTP method for the API endpoint"
    )
    headers: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Headers to include in the API request"
    )
    query_params: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Query parameters for the API request"
    )
    body: Optional[Any] = Field(
        default=None, description="Body of the API request for POST/PUT methods"
    )

    def __str__(self):
        """
        String representation of the EndPoint.
        """
        return f"EndPoint(path={self.path}, method={self.method})"


class APIClient(BaseModel):
    """
    APIClient is a configuration and utility class for making HTTP requests to APIs with support for authentication, retries, and flexible request handling.
    """

    base_url: Optional[str] = Field(None, description="Base URL of the API")
    timeout_seconds: int = Field(30, description="Timeout for API requests in seconds")
    authentication_method: APIAuthenticationMethod = Field(
        "Bearer Token", description="API authentication method"
    )
    max_retries: int = Field(
        3, description="Maximum number of retries for API requests"
    )
    verify_ssl: bool = Field(
        False, description="Whether to verify SSL certificates for HTTPS requests"
    )

    def make_request(
        self, endpoint: EndPoint, token: str, **kwargs
    ) -> Optional[requests.Response]:
        """
        Sends an HTTP request to the specified API endpoint using the provided configuration.
        Args:
            endpoint (EndPoint): The API endpoint configuration, including path, method, query parameters, and body.
            token (str): The authentication token to include in the request.
            **kwargs: Additional keyword arguments to pass to the underlying `requests` method.
        Returns:
            Optional[requests.Response]: The HTTP response object if the request is successful, otherwise None.
        Raises:
            ValueError: If the base URL is not set or if no response is received from the API.
            requests.RequestException: If the HTTP request fails (raised internally by `response.raise_for_status()`).
        Notes:
            - Supports GET, POST, PUT, and DELETE HTTP methods.
            - Handles different body types for POST/PUT requests: dict, list, str, bytes, or None.
            - SSL certificate verification is controlled by the verify_ssl field.
            - When verify_ssl=True: Performs on-demand SSL verification including hostname and expiry checks during HTTPS requests.
            - When verify_ssl=False: Completely disables SSL verification and accepts any certificate.
            - Shows warning for HTTP (non-HTTPS) URLs but proceeds with the request.
            - No prior certificate validation or pre-loading is performed.
            - Logs the status of the request using the `Status` class.
            - Ensures the session is properly closed after the request.
        """
        session = None
        try:
            if not self.base_url:
                raise ValueError("Base URL is not set.")

            session = self._make_session(endpoint)

            # set authentication
            self._auth(session, token)

            # set base url
            if not self.base_url.endswith("/"):
                self.base_url = f"{self.base_url}/"

            endpoint_url = urljoin(self.base_url, endpoint.path.lstrip("/"))

            # Execute the request using the session's configured SSL verification
            response: Optional[requests.Response] = self._execute_request(
                session, endpoint, endpoint_url, session.verify, **kwargs
            )

            if response is None:
                raise ValueError("No response received from the API.")

            response.raise_for_status()
            Status.SUCCESS(
                "API request successful", self, endpoint, response=response.status_code
            )
            return response
        except Exception as e:
            Status.FAILED(
                "API request failed", self, endpoint, error=str(e), traceback=False
            )
            return None
        finally:
            if session:
                session.close()

    def _execute_request(
        self,
        session: requests.Session,
        endpoint: EndPoint,
        endpoint_url: str,
        verify_setting: bool,
        **kwargs,
    ) -> Optional[requests.Response]:
        """
        Execute the HTTP request with the given parameters.

        Args:
            session: The requests session to use
            endpoint: The endpoint configuration
            endpoint_url: The full URL to request
            verify_setting: SSL verification setting
            **kwargs: Additional arguments for the request

        Returns:
            Optional[requests.Response]: The response object or None if failed
        """
        try:
            response: Optional[requests.Response] = None
            if endpoint.method == "GET":
                response = session.get(
                    endpoint_url,
                    params=endpoint.query_params,
                    timeout=self.timeout_seconds,
                    verify=verify_setting,
                    **kwargs,
                )
            elif endpoint.method == "POST":
                # Handle different body types
                if isinstance(endpoint.body, (dict, list)):
                    response = session.post(
                        endpoint_url,
                        json=endpoint.body,
                        timeout=self.timeout_seconds,
                        verify=verify_setting,
                        **kwargs,
                    )
                elif isinstance(endpoint.body, (str, bytes)):
                    response = session.post(
                        endpoint_url,
                        data=endpoint.body,
                        timeout=self.timeout_seconds,
                        verify=verify_setting,
                        **kwargs,
                    )
                else:
                    response = session.post(
                        endpoint_url,
                        timeout=self.timeout_seconds,
                        verify=verify_setting,
                        **kwargs,
                    )
            elif endpoint.method == "PUT":
                # Handle different body types
                if isinstance(endpoint.body, (dict, list)):
                    response = session.put(
                        endpoint_url,
                        json=endpoint.body,
                        timeout=self.timeout_seconds,
                        verify=verify_setting,
                        **kwargs,
                    )
                elif isinstance(endpoint.body, (str, bytes)):
                    response = session.put(
                        endpoint_url,
                        data=endpoint.body,
                        timeout=self.timeout_seconds,
                        verify=verify_setting,
                        **kwargs,
                    )
                else:
                    response = session.put(
                        endpoint_url,
                        timeout=self.timeout_seconds,
                        verify=verify_setting,
                        **kwargs,
                    )
            elif endpoint.method == "DELETE":
                response = session.delete(
                    endpoint_url,
                    timeout=self.timeout_seconds,
                    verify=verify_setting,
                    **kwargs,
                )

            return response
        except Exception as e:
            raise ValueError(f"API request failed for {endpoint_url}: {str(e)}") from e

    def _make_session(self, endpoint: EndPoint) -> requests.Session:
        """
        Create a requests session with retry strategy and on-demand SSL verification.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        # create url
        parsed_url = urlparse(self.base_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid base URL provided")

        # Check if URL is HTTP and show warning
        if parsed_url.scheme == "http":
            session.verify = False  # No SSL verification needed for HTTP
            Status.WARNING(
                "HTTP connection detected - SSL verification not applicable.",
                self,
            )
        elif parsed_url.scheme == "https":
            # Configure SSL verification based on verify_ssl setting
            if not self.verify_ssl:
                # Disable SSL verification and warnings
                session.verify = False
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                Status.INFO(
                    f"SSL verification disabled by configuration for {self.base_url}",
                    self,
                )
            else:
                # Enable on-demand SSL verification with hostname and expiry checks
                session.verify = True
                Status.INFO(
                    "On-demand SSL verification enabled.",
                    self,
                )
        else:
            # Unsupported scheme
            raise ValueError(
                f"Unsupported URL scheme: {parsed_url.scheme}. Only HTTP and HTTPS are supported."
            )

        # set session headers
        session.headers.update(endpoint.headers)

        # Mount adapters
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _auth(self, session: requests.Session, token: str) -> None:
        """
        Authenticate the session if required.
        """
        if self.authentication_method == "Bearer Token":

            if token:
                if token.startswith("Bearer "):
                    session.headers["Authorization"] = token
                else:
                    raise ValueError("Invalid Bearer Token format")
            else:
                raise ValueError("Failed to generate authentication token")

    def __str__(self):
        """
        String representation of the APIClient.
        """
        return f"APIClient(base_url={self.base_url}, authentication={self.authentication_method})"
