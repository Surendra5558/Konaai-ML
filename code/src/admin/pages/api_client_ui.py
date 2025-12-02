# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a form for configuring API endpoint settings."""
from typing import get_args
from urllib.parse import urlparse

from nicegui import ui
from src.utils.api_config import APIAuthenticationMethod
from src.utils.api_config import APIClient


class APIClientForm:
    """
    A UI form component for configuring API client settings using NiceGUI.

    Attributes:
        client (APIClient): The data model that stores API configuration fields such as
                            base URL, timeout, authentication method, etc.
    """

    def __init__(self, client: APIClient = None):
        """
        Initialize the form with an optional APIClient instance.

        Args:
            client (APIClient, optional): An existing APIClient object to bind the form to.
                                          If not provided, a new instance is created.
        """
        self.client = client or APIClient()

    def render(self):
        """
        Render the API client configuration form in the UI.

        The form includes the following input fields:
            - Base URL (validated)
            - Timeout (in seconds)
            - Authentication method (dropdown from enum)
            - Max retries (integer)
            - Verify SSL (toggle switch)

        All inputs are two-way bound to the `APIClient` instance.
        """
        with ui.card().classes("w-full"):
            ui.input(
                label="Base URL",
                value=self.client.base_url,
                validation={
                    "Not a valid URL": lambda v: self._validate_url(  # pylint: disable=unnecessary-lambda
                        v
                    )
                },
            ).bind_value_to(self.client, "base_url").classes("w-full")

            ui.number(
                label="Timeout (seconds)",
                value=self.client.timeout_seconds,
                precision=0,
                min=1,
                max=300,
                step=1,
                format="%d",
            ).bind_value_to(self.client, "timeout_seconds").classes("w-full")

            ui.select(
                label="Authentication Method",
                options=list(get_args(APIAuthenticationMethod)),
                value=self.client.authentication_method,
            ).bind_value_to(self.client, "authentication_method").classes("w-full")

            ui.number(
                label="Max Retries",
                value=self.client.max_retries,
                precision=0,
                min=0,
                max=10,
                step=1,
                format="%d",
            ).bind_value_to(self.client, "max_retries").classes("w-full")

            ui.switch(
                text="Verify SSL",
                value=self.client.verify_ssl,
            ).bind_value_to(
                self.client, "verify_ssl"
            ).classes("w-full")

    def _validate_url(self, url: str) -> bool:
        """
        Validate that the provided URL is well-formed.

        Args:
            url (str): The URL string to validate.

        Returns:
            bool: True if the URL has a valid scheme and network location, False otherwise.
        """
        if not url:
            return False
        parsed_url = urlparse(url)
        return bool(parsed_url.scheme and parsed_url.netloc)
