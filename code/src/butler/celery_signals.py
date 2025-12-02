# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the Celery signal handlers."""
from celery.signals import after_setup_logger
from celery.signals import after_setup_task_logger
from celery.signals import after_task_publish
from celery.signals import before_task_publish
from celery.signals import celeryd_after_setup
from celery.signals import celeryd_init
from celery.signals import task_failure
from celery.signals import task_internal_error
from celery.signals import task_postrun
from celery.signals import task_prerun
from celery.signals import task_received
from celery.signals import task_rejected
from celery.signals import task_retry
from celery.signals import task_revoked
from celery.signals import worker_before_create_process
from celery.signals import worker_init
from celery.signals import worker_process_init
from celery.signals import worker_process_shutdown
from celery.signals import worker_ready
from celery.signals import worker_shutdown
from celery.signals import worker_shutting_down
from src.butler.celery_result_backend import TaskResultTable
from src.butler.cleanup import CleanFiles
from src.utils.status import Status


# setup global logger
@after_setup_logger.connect
def setup_global_logger(logger, *args, **kwargs):  # pylint: disable=unused-argument
    """
    Set up the global logger by removing existing handlers and adding the file and stream handlers from logger.

    Parameters:
    - logger: The logger object to set up.

    Returns:
    None
    """
    for handler in logger.handlers:
        logger.removeHandler(handler)

    for handler in Status.log_handlers():
        logger.addHandler(handler)


# setup task logger
@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):  # pylint: disable=unused-argument
    """
    Set up the task logger with the specified logger object.

    Parameters:
    - logger: The logger object to set up.

    Returns:
    - None
    """
    for handler in logger.handlers:
        logger.removeHandler(handler)

    for handler in Status.log_handlers():
        logger.addHandler(handler)


@task_prerun.connect
def task_prerun_handler(
    task_id=None, task=None, **kwargs
):  # pylint: disable=unused-argument
    """
    Handler function that is called before a Celery task runs.
    This function is typically used to perform any setup or logging before the task execution.
    Args:
    -----
        task_id (str, optional): The unique identifier for the task. Defaults to None.
        task (Union[str, object], optional): The task object or task name. Defaults to None.
        **kwargs: Additional keyword arguments.

    Notes:
        - If `task` is a string, it is assumed to be the task name.
        - If `task` is an object and has a `name` attribute, the task name is extracted from it.
    """

    name = None
    if isinstance(task, str):
        name = task
    elif hasattr(task, "name"):
        name = task.name

    Status.INFO(f"Task {name} is about to run with task_id {task_id}")
    CleanFiles.old_files_cleanup()


@task_postrun.connect
def task_postrun_handler(
    task_id=None, task=None, retval=None, state=None, **kwargs
):  # pylint: disable=unused-argument
    """Handler function that is called after a Celery task runs."""
    name = None
    if isinstance(task, str):
        name = task
    elif hasattr(task, "name"):
        name = task.name

    CleanFiles.old_files_cleanup()

    Status.INFO(f"Task {name} has completed with task_id {task_id} and state {state}")


@before_task_publish.connect
def task_before_publish_handler(
    sender=None, headers=None, body=None, **kwargs
):  # pylint: disable=unused-argument
    """
    This function is called before a task is published.

    Parameters:
    - sender: The sender of the task.
    - headers: The headers of the task.
    - body: The body of the task.
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    name = None
    if isinstance(sender, str):
        name = sender
    elif hasattr(sender, "name"):
        name = sender.name

    # update backend with task_id
    status = "PENDING"
    task_id = headers.get("id")
    if TaskResultTable.upsert_task(task_id, status):
        Status.INFO(f"Task {name} updated in backend with task_id {task_id}")
    else:
        Status.FAILED(f"Task {name} not updated in backend with task_id {task_id}")


@after_task_publish.connect
def task_after_publish_handler(
    sender=None, headers=None, body=None, **kwargs
):  # pylint: disable=unused-argument
    """
    This function is called after a task is published.

    Parameters:
    - sender: The sender of the task.
    - headers: The headers of the task.
    - body: The body of the task.
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    name = None  # pylint: disable=unused-variable
    if isinstance(sender, str):
        name = sender
    elif hasattr(sender, "name"):
        name = sender.name

    Status.INFO(f"Task {name} has been published")


@task_retry.connect
def task_retry_handler(
    request=None, reason=None, einfo=None, **kwargs
):  # pylint: disable=unused-argument
    """
    This function is called when a task is retried.

    Parameters:
    - request: The request object.
    - reason: The reason for the retry.
    - einfo: The exception information.
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    name = None
    if isinstance(request, str):
        name = request
    elif hasattr(request, "name"):
        name = request.name

    Status.INFO(f"Task {name} is being retried because of {reason}.", exception=einfo)


@task_failure.connect
def task_failure_handler(
    task_id=None,
    exception=None,
    args=None,
    traceback=None,
    einfo=None,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    This function is called when a task fails.

    Parameters:
    - task_id: The ID of the task.
    - exception: The exception that caused the failure.
    - args: The arguments of the task.
    - kwargs: The keyword arguments of the task.
    - traceback: The traceback of the failure.
    - einfo: The exception information.
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.FAILED(f"Task {task_id} failed", args=args, kwargs=kwargs, exception=einfo)


@task_internal_error.connect
def task_internal_error_handler(
    task_id=None,
    exception=None,
    args=None,
    traceback=None,
    einfo=None,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    This function is called when an internal error occurs in a task.

    Parameters:
    - task_id: The ID of the task.
    - exception: The exception that caused the internal error.
    - args: The arguments of the task.
    - kwargs: The keyword arguments of the task.
    - traceback: The traceback of the internal error.
    - einfo: The exception information.
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.FAILED(
        f"Task {task_id} encountered an internal error",
        args=args,
        kwargs=kwargs,
        exception=einfo,
    )


@task_received.connect
def task_received_handler(request=None, **kwargs):  # pylint: disable=unused-argument
    """
    This function is called when a task is received by a worker.

    Parameters:
    - request: The request object.

    Returns:
    - None
    """
    name = None
    if isinstance(request, str):
        name = request
    elif hasattr(request, "name"):
        name = request.name
    Status.INFO(f"Task {name} has been received by a worker")


@task_revoked.connect
def task_revoked_handler(
    request=None,
    terminated=None,
    signum=None,
    expired=None,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    This function is called when a task is revoked.

    Parameters:
    - request: The request object.
    - terminated: Whether the task was terminated.
    - signum: The signal number.
    - expired: Whether the task expired.

    Returns:
    - None
    """
    name = None
    if isinstance(request, str):
        name = request
    elif hasattr(request, "name"):
        name = request.name
    Status.WARNING(
        f"Task {name} has been revoked with signal number {signum} and expired status {expired}"
    )


@task_rejected.connect
def task_rejected_handler(
    message: None, exc=None, **kwargs
):  # pylint: disable=unused-argument
    """
    This function is called when a task is rejected.

    Parameters:
    - message: The message.
    - exc: The exception.

    Returns:
    - None
    """
    Status.FAILED(f"Task has been rejected because of {exc}.")


@celeryd_after_setup.connect
def celeryd_after_setup_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called after the Celery daemon is set up.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.INFO("Celery daemon has been set up.")


@celeryd_init.connect
def celeryd_init_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called when the Celery daemon is initialized.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.INFO("Celery daemon is initializing.")


@worker_init.connect
def worker_init_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called when a worker is initialized.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.INFO("Worker is initializing.")


@worker_before_create_process.connect
def worker_before_create_process_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called before a worker process is created.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.INFO("Worker process is being created.")


@worker_ready.connect
def worker_ready_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called when a worker is ready.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.INFO("Worker is ready.")


@worker_shutting_down.connect
def worker_shutting_down_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called when a worker is shutting down.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.WARNING("Worker is shutting down.")


@worker_process_init.connect
def worker_process_init_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called when a worker process is initialized.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.INFO("Worker process is initializing.")


@worker_process_shutdown.connect
def worker_process_shutdown_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called when a worker process is shut down.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.WARNING("Worker process is shutting down.")


@worker_shutdown.connect
def worker_shutdown_handler(**kwargs):  # pylint: disable=unused-argument
    """
    This function is called when a worker is shut down.

    Parameters:
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    Status.WARNING(
        "Worker is shutting down. This could be because of broker connection issues or other reasons."
    )
