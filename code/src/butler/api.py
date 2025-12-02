# # Copyright (C) KonaAI - All Rights Reserved
# Copyright (C) KonaAI - All Rights Reserved
"""Butler API endpoints"""
from typing import Optional

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Header
from src.butler.api_setup import validate_token
from src.utils.api_response import APIResponse
from src.utils.status import Status
from src.utils.task_queue import TaskOutput
from src.utils.task_queue import TaskQueue

butler_router = APIRouter(tags=["Butler"])


@butler_router.get("/health")
async def health_check():
    """
    Performs a health check for the API.

    This asynchronous function returns a dictionary indicating the status
    of the API endpoints.

    Returns:
        dict: A dictionary containing the status message of the API.
    """
    return {"status": "API Endpoints Up & Running"}


class TaskStatusResponse(APIResponse):
    """
    Represents the response returned by the task status endpoint.
    Attributes:
        data (Optional[TaskOutput]): The output of the task if available; otherwise, None.
    """

    data: Optional[TaskOutput] = None


@butler_router.post(
    "/task", response_model=TaskStatusResponse, dependencies=[Depends(validate_token)]
)
async def get_task_status(
    task_id: str = Header(..., alias="taskId"),
):
    """
    Retrieve the status of a task by its ID.

    Args:
        task_id (str): The unique identifier of the task, provided via the "taskId" header.

    Returns:
        TaskStatusResponse: An object containing the status of the task and any associated output data.

    Raises:
        Exception: If an error occurs while retrieving the task status, returns a failed status response with error details.
    """
    try:
        task_id = task_id.lower()
        output, s = TaskQueue.get_task_result(task_id)
        return TaskStatusResponse().assign_status(s, data=output)
    except Exception as e:
        s = Status.FAILED("Task status check failed: ", error=str(e), traceback=False)
        return TaskStatusResponse().assign_status(s, data=None, error=str(e))
