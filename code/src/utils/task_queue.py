# # Copyright (C) KonaAI - All Rights Reserved
"""Task Queue class to manage the tasks in the queue"""
import json
import os
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple

from pydantic import BaseModel
from pydantic import Field
from src.butler.celery_result_backend import TaskResultTable
from src.utils.conf import Setup
from src.utils.status import Status
from src.worker import celery


class TaskOutput(BaseModel):
    """
    TaskOutput is a Pydantic model that represents the output of a task.

    Attributes:
    ----------
        task_id (str): The task ID of the task. Defaults to None.
        output_type (Optional[Literal["File", "JSON"]]): The output type of the task,
            which can either be "File" or "JSON". Defaults to None.
        output (Optional[dict]): The output of the task, if applicable. Defaults to None.
        file_name (Optional[str]): The file name of the task's output, if applicable. Defaults to None.
        status (Literal['In Progress', 'Completed', 'Failed', 'Not Found']): The status of the task,
            which can be one of the following: 'In Progress', 'Completed', 'Failed', or 'Not Found'.
    """

    task_id: str = Field(None, description="The task id of the task")
    output_type: Optional[Literal["File", "JSON"]] = Field(
        None, description="The output type of the task"
    )
    output: Optional[dict] = Field(None, description="The output of the task")
    file_name: Optional[str] = Field(None, description="The file output of the task")
    status: Literal["In Progress", "Completed", "Failed", "Not Found"] = Field(
        ..., description="The status of the task"
    )


class TaskQueue(BaseModel):
    """
    TaskQueue class for managing a persistent queue of tasks associated with a specific instance.
    This class provides methods to add, remove, and retrieve tasks from a queue, with automatic persistence to disk.
    It also offers static methods to interact with Celery tasks, including checking task status, retrieving completion timestamps,
    and obtaining task results.
    Attributes:
    ----------
        instance_id (str): The unique identifier for the task queue instance.
        tasks (Dict): A dictionary mapping task names to task IDs.
    """

    instance_id: str = Field(None, description="The instance id of the task queue")
    tasks: Dict = Field({}, description="The tasks in the queue")

    def __init__(self, instance_id: str):
        super().__init__(instance_id=instance_id)
        self.instance_id = instance_id
        # load the tasks from the file
        self.__load_from_file_()

    def __get_file_path_(self):
        """Get the file path for the task queue"""
        file_name = Setup().global_constants.get("TASK_QUEUE").get("FILE_NAME")
        file_path = os.path.join(Setup().db_path, self.instance_id, file_name)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.__save_to_file_(file_path)
        return file_path

    def __load_from_file_(self):
        """Load the task queue from the file"""
        file_path = self.__get_file_path_()
        try:
            # construct the object from the file

            with open(file_path, encoding="utf-8") as file:
                data = json.load(file)
                self.instance_id = data.get("instance_id")
                self.tasks = data.get("tasks")
        except BaseException as _e:
            Status.FAILED("Could not load task queue", error=str(_e))

    def __save_to_file_(self, file_path: str):
        """Save the task queue to the file"""

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.model_dump(), file)

    def add_task(self, task_name: str, task_id: str):
        """Add a task to the queue and stores in task_queue.json file"""
        self.tasks[task_name] = task_id
        file_path = self.__get_file_path_()
        self.__save_to_file_(file_path)

    def remove_task(self, task_name: str):
        """Remove a task from the task_queue.json file"""
        self.tasks.pop(task_name, None)
        file_path = self.__get_file_path_()
        self.__save_to_file_(file_path)

    def get_task_id_by_name(self, task_name: str) -> Optional[str]:
        """Get the task id by name"""
        return self.tasks.get(task_name)

    @staticmethod
    def task_status_by_id(task_id: str) -> Tuple[bool, bool]:
        """
        Check if a Celery task is completed, and whether it succeeded or failed.
        :param task_id: The ID of the Celery task.
        :return: A tuple of two boolean values: (is_completed, is_successful).
        """
        try:
            is_completed = False
            is_successful = False

            if not task_id:
                return is_completed, is_successful

            # check status
            status = TaskResultTable.get_task_status(task_id)
            Status.INFO(f"Task with ID {task_id} is in state: {status.upper()}")

            if status.upper() in ("SUCCESS", "FAILURE", "REVOKED"):
                is_completed = True

            if is_completed and status == "SUCCESS":
                is_successful = True

        except BaseException as _e:
            Status.FAILED("Error while checking task status", error=str(_e))
        return is_completed, is_successful

    @staticmethod
    def get_task_completed_timestamp(task_id: str) -> Optional[str]:
        """
        Get the timestamp of when a Celery task was completed.
        :param task_id: The ID of the Celery task.
        :return: The timestamp of when the task was completed, or None if the task is not completed or an error occurred.
        """
        try:
            # check if task is completed
            is_completed, _ = TaskQueue.task_status_by_id(task_id)
            if is_completed:
                task = celery.AsyncResult(task_id)
                return task.date_done
            return None
        except BaseException as _e:
            Status.FAILED("Error while getting task completed timestamp", error=str(_e))
            return None

    @staticmethod
    def get_task_result(task_id: str) -> Tuple[TaskOutput, Status]:
        """
        Retrieve the result and status of a task based on its task ID.
        Args:
        -----
            task_id (str): The unique identifier of the task.

        Returns:
            Tuple[TaskOutput, Status]: A tuple containing:
                - TaskOutput: An object representing the task's output, including its status and optional result data.
                - Status: An object representing the status of the operation, including messages and metadata.

        Behavior:
            - If the task ID does not exist, returns a "Not Found" status.
            - If the task is still in progress, returns an "In Progress" status.
            - If the task has completed unsuccessfully, returns a "Failed" status.
            - If the task has completed successfully, returns a "Completed" status along with the task's result.
            - If the task result is not in a supported format (e.g., non-JSON), raises a NotImplementedError.
            - If an error occurs during the process, returns a "Failed" status with error details.

        Exceptions:
            - NotImplementedError: Raised if the task result is in an unsupported format.

        Notes:
            - This function interacts with a task queue and a result table to determine the task's status and retrieve its result.
            - The function assumes that task results are retrieved using a Celery backend.
        """
        try:
            # check if task id is valid
            if not TaskResultTable.task_exists(task_id):
                s = Status.NOT_FOUND("Task not found", task_id=task_id)
                return (
                    TaskOutput(
                        status="Not Found",
                        task_id=task_id,
                    ),
                    s,
                )

            complete, successful = TaskQueue.task_status_by_id(task_id)
            # check if task is completed
            if not complete:
                s = Status.INFO("Task is still in progress", task_id=task_id)
                return (
                    TaskOutput(
                        status="In Progress",
                        task_id=task_id,
                    ),
                    s,
                )

            # check if task is successful
            if not successful:
                s = Status.FAILED("Task failed", task_id=task_id)
                return (
                    TaskOutput(
                        status="Failed",
                        task_id=task_id,
                    ),
                    s,
                )

            # check if task is successful and completed
            s = Status.COMPLETED("Task is complete", task_id=task_id)
            task = celery.AsyncResult(task_id)
            output = TaskOutput(status="Completed", task_id=task_id)
            if task.ready():
                result = task.get()
                if isinstance(result, dict):
                    output.output_type = "JSON"
                    output.output = result
                else:
                    raise NotImplementedError(
                        "Output type not supported. Only JSON output is supported."
                    )
            return output, s
        except BaseException as _e:
            s = Status.FAILED(
                "Error while getting task result", error=str(_e), task_id=task_id
            )
            return TaskOutput(status="Failed", task_id=task_id), s
