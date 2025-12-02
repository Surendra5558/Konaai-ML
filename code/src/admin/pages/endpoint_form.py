# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a form for configuring API endpoint settings."""
import json
from typing import get_args
from typing import get_type_hints
from typing import List

from nicegui import ui
from src.tools.json_handler import is_valid_json
from src.utils.api_config import EndPoint

required = {"This is a required field.": lambda x: x is not None and x != ""}


class EndPointForm:
    """
    EndPointForm is a UI form class for configuring API endpoint settings.
    Attributes:
    ------------
        endpoint (EndPoint): The endpoint instance being configured.
    """

    def __init__(self, endpoint: EndPoint = None):
        """
        Initialize the EndpointForm with an optional EndPoint instance.

        Args:
            endpoint (EndPoint, optional): An existing EndPoint object to initialize the form with.
                If not provided, a new EndPoint instance is created.
        """
        self.endpoint = endpoint or EndPoint()

    def render(self):
        """
        Renders the endpoint configuration form UI.

        This form allows users to configure the following endpoint properties:

            - Path: The URL path for the endpoint, validated to start with '/' and contain only alphanumeric characters and slashes.
            - Method: The HTTP method for the endpoint, selectable from available options.
            - Headers: Optional HTTP headers, entered as a JSON object.
            - Query Parameters: Optional query parameters, entered as a JSON object.
        All fields include appropriate validation and bind their values to the corresponding attributes of the endpoint instance.
        """
        with ui.card().classes("w-full"):
            ui.input(
                label="Path",
                value=self.endpoint.path,
                validation={
                    "Path must start with '/' and contain only alphanumeric characters and slashes.": lambda x: isinstance(
                        x, str
                    )
                    and x.startswith("/")
                    and all(c.isalnum() or c == "/" for c in x),
                    **required,
                },
            ).bind_value_to(self.endpoint, "path").classes("w-full")

            # Method selection with validation
            methods = list(get_args(get_type_hints(EndPoint).get("method", List[str])))
            ui.select(
                label="Method", options=methods, value=self.endpoint.method
            ).bind_value_to(self.endpoint, "method").classes("w-full")

            # Optional fields for headers, query parameters, and body
            # headers, its a dictionary, so we can use a text area for JSON input
            ui.textarea(
                label="Headers (JSON format)",
                value=str(self.endpoint.headers or {}),
                placeholder='{"Content-Type": "application/json"}',
                validation={
                    "Headers must be a valid JSON object.": lambda x: is_valid_json(  # pylint: disable=unnecessary-lambda
                        x
                    )
                },
            ).bind_value_to(
                self.endpoint,
                "headers",
                forward=lambda x: json.loads(x) if is_valid_json(x) else {},
            ).classes(
                "w-full"
            )

            ui.textarea(
                label="Query Parameters (JSON format)",
                value=str(self.endpoint.query_params or {}),
                validation={
                    "Query parameters must be a valid JSON object.": lambda x: is_valid_json(  # pylint: disable=unnecessary-lambda
                        x
                    )
                },
            ).bind_value_to(
                self.endpoint,
                "query_params",
                forward=lambda x: json.loads(x) if is_valid_json(x) else {},
            ).classes(
                "w-full"
            )
