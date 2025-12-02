# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a utility to call ETL update endpoints for application callbacks."""
from enum import Enum
from typing import Optional

import requests
from src.utils.api_config import EndPoint
from src.utils.auth import generate_token
from src.utils.instance import Instance
from src.utils.status import Status
from src.utils.submodule import Submodule


class APPLICATION_STATUS(Enum):
    """
    Enum representing the various statuses an application can have during its lifecycle.
    Attributes:
    ----------
        IN_QUEUE (int): The application is in the queue and waiting to be processed.
        IN_PROGRESS (int): The application is currently being processed.
        SUCCESS (int): The application has been processed successfully.
        FAILED (int): The application processing has failed.
    """

    IN_QUEUE = 2
    IN_PROGRESS = 3
    SUCCESS = 4
    FAILED = 5


def call_etl_update_endpoint(  # pylint: disable=too-many-positional-arguments
    endpoint: EndPoint,
    instance: Instance,
    submodule: Submodule,
    task_id: str,
    status: APPLICATION_STATUS,
    description: str = None,
) -> Optional[requests.Response]:
    """
    Calls the ETL update endpoint with the provided parameters and handles the response.
    Args:
        endpoint (EndPoint): The API endpoint to call.
        instance (Instance): The instance containing project and client information.
        submodule (Submodule): The submodule information for the ETL update.
        task_id (str): The unique identifier for the task.
        status (APPLICATION_STATUS): The status to update for the ETL process.
        description (str, optional): Additional description or current stage of the ETL process.
    Returns:
        Optional[requests.Response]: The response object from the API call if successful, otherwise None.
    Raises:
        ValueError: If required parameters (instance, submodule, or endpoint) are missing.
    """
    if not instance or not submodule:
        raise ValueError("Instance and Submodule must be provided.")
    if not endpoint or endpoint.path is None:
        raise ValueError(f"No endpoint found for instance {instance}")

    payload = {
        "ProjectId": instance.ProjectUID,
        "ClientId": instance.ClientUID,
        "ModuleName": submodule.module,
        "SubmoduleName": submodule.submodule,
        "Status": status.value,
        "TaskId": task_id,
        "CurrentStage": description,
    }
    endpoint.body = payload
    endpoint.headers.update(
        {"T-ID": instance.ClientUID},
    )
    Status.INFO("Calling ETL update endpoint", endpoint, payload=payload)
    # Generate token for authentication
    token = generate_token(instance.ClientUID, instance.ProjectUID, expiry_minutes=5)
    response = instance.settings.application_api_client.make_request(
        endpoint, token=token
    )
    if response is None or response.status_code != 200:
        Status.FAILED(
            "ETL update endpoint call failed",
            endpoint,
            payload=payload,
            error=response.text if response else "No response",
            traceback=True,
        )
        return None

    Status.SUCCESS(
        "ETL update endpoint call succeeded",
        endpoint,
        payload=payload,
        response=response,
    )
    return response
