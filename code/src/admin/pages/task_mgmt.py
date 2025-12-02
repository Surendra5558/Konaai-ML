# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides visibility on celery tasks"""
from typing import Dict
from typing import List
from typing import Union

from celery.result import AsyncResult
from nicegui import ui
from src.admin import theme
from src.butler.celery_broker import ack_message
from src.butler.celery_broker import get_pending_messages
from src.butler.celery_broker import Message
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.worker import celery


def _dict_to_table(data: Dict[str, Union[str, int]]) -> None:
    """
    Recursively displays a dictionary as a labeled table within a UI column.

    Args:
        data (Dict[str, Union[str, int]]): The dictionary containing keys and values
        to be displayed. Nested dictionaries will be recursively processed.
    """

    with ui.column():
        for k, v in data.items():
            if isinstance(v, dict):
                _dict_to_table(v)
            else:
                key = k.replace("_", " ").title()
                ui.label(f"{key}: {v}").style("font-size: 14px;")


def fetch_tasks() -> None:
    """
    Fetches and displays all pending tasks from the task queue in the UI.
    This function creates an expandable section in the UI to show pending tasks.
    It displays a label explaining the list, shows a spinner while loading tasks,
    and lists each pending task in a table format. For each task, a "Cancel Task"
    button is provided to allow cancellation. If no tasks are found, a notification
    is shown. A horizontal separator is added at the end.

    Returns:
        None
    """
    with (
        ui.expansion("View Pending Tasks")
        .classes("w-full border-2 rounded-md")
        .props("outlined")
    ):
        ui.label(
            "Below list only shows tasks pending for execution. It does not show tasks that are currently running or completed."
        ).classes("w-full").style(
            "color: black; background-color: LightYellow; padding: 10px; length: 100%; border-radius: 10px; font-size: 14px;"
        )
        with ui.spinner(size="lg") as spinner:
            tasks: List[Message] = get_pending_messages()
            if not tasks:
                ui.notify("No tasks found", type="warning")

            for task in tasks:
                t: Message = task
                _dict_to_table(t.to_dict())
                # cancel task button
                ui.button(
                    "Cancel Task",
                    on_click=lambda task_id=t.id: cancel_tasks(task_id),
                ).style("font-size: 14px;")
        spinner.delete()

        # print horizontal line
        ui.separator()


def cancel_tasks(task_id: str = None) -> None:
    """
    Cancel a Celery task by its task ID.
    If a task ID is provided, attempts to revoke and terminate the Celery task with the given ID,
    cancel its consumer, and notify the user of the result. If no task ID is provided, notifies
    the user to provide one. Handles and notifies about any errors encountered during the process.

    Args:
        task_id (str, optional): The ID of the Celery task to cancel. Defaults to None.
    Returns:
        None
    """
    try:
        if task_id:
            task = AsyncResult(task_id)
            task.revoke(terminate=True, signal="SIGKILL")
            celery.control.cancel_consumer(task_id, reply=True)

            if ack_message(task_id):
                ui.notify(f"Task {task_id} has been cancelled", type="positive")
            else:
                ui.notify(f"Failed to cancel task {task_id}", type="negative")
        else:
            ui.notify("Please provide a task ID to cancel", type="warning")

    except BaseException as _e:
        Status.FAILED(f"Error while cancelling task, {_e}")
        ui.notify("Error while cancelling task", type="negative")


def task_mgmt():
    """
    Displays the Task Management interface for troubleshooting the app.
    This function renders a UI frame that shows details of the currently active instance,
    including its ID, client name, and project name. If no active instance is found,
    a notification prompts the user to activate one. The interface also provides a card
    to cancel tasks by entering a Task ID and clicking the "Terminate" button. Finally,
    it fetches and displays the current tasks.

    Returns:
        None
    """
    with theme.frame("Task Management"):
        ui.markdown("# Task Management").style("font-size: 18px;")
        # Show active instance
        if GlobalSettings().active_instance_id:
            instance_obj = GlobalSettings.instance_by_id(
                GlobalSettings().active_instance_id
            )
            if instance_obj:
                with ui.grid(columns=3).classes("w-full gap-1 items-start"):
                    for label, value in [
                        ("Active Instance ID", instance_obj.instance_id),
                        ("Client Name", instance_obj.client_name),
                        ("Project Name", instance_obj.project_name),
                    ]:
                        with ui.column().classes("gap-1"):
                            ui.label(label).classes("text-sm text-gray-500")
                            ui.label(value).classes("text-base font-semibold")
            else:
                ui.label("No active instance found").classes("text-red-500")
        else:
            ui.notify("Please activate an instance to continue", type="negative")
            return
        # cancel tasks
        with ui.card().classes("w-full border-2 rounded-md").props("outlined"):
            ui.label("Cancel Tasks").classes(
                "text-base text-black-400 font-semibold"
            ).style("font-size: 16px;")
            task_id = (
                ui.input("Task ID")
                .classes("")
                .props("outlined")
                .style("width: 30%; font-size: 14px;")
            )
            ui.button("Terminate", on_click=lambda: cancel_tasks(task_id.value)).style(
                "font-size: 14px;"
            )

        fetch_tasks()
