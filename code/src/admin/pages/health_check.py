# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the health check app"""
import asyncio
from contextlib import contextmanager
from typing import Optional

from kombu import Connection
from nicegui import run
from nicegui import ui
from src.admin import theme
from src.admin.components.clipboard import copy_to_clipboard
from src.admin.components.spinners import create_loader
from src.butler.celery_broker import get_broker_url
from src.utils.auth import generate_token
from src.utils.custom_logger import app_logger
from src.utils.database_config import SQLDatabaseManager
from src.utils.global_config import GlobalSettings
from src.utils.notification import EmailNotification
from src.utils.status import Status


@contextmanager
def disable(button: ui.button):
    """
    A context manager that temporarily disables a UI button and displays a loading spinner while executing a block of code.

    Args:
        button (ui.button): The button to disable during the operation.
    The button is re-enabled and the loader is removed after the block, even if an exception occurs.
    """
    loader = create_loader()
    try:
        yield
    finally:
        loader.delete()
        button.enable()


def health_check():
    """
    Displays a health check dashboard section in the admin panel.
    This function creates a UI section that allows administrators to:
    - Check broker connection status.
    - Validate worker, master, and instance database connections.
    - View details of the currently active instance (ID, client name, project name).
    - Generate and display a JWT token.
    - Send a test email notification.

    If no active instance is set, notifies the user to activate one before proceeding.
    """
    with theme.frame("Health Check"):
        ui.markdown("# Health Check").classes("text-lg font-bold")

        # Check broker connection
        with (
            ui.expansion("Broker Connection", value=False)
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            _broker_check()

        # Check worker database connection
        with (
            ui.expansion("Worker Database Connection", value=False)
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            validate_db_connection(_db="worker")

        # Show active instance
        if GlobalSettings().active_instance_id:
            if instance_obj := GlobalSettings.instance_by_id(
                GlobalSettings().active_instance_id
            ):
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

        # Check instance database connection
        with (
            ui.expansion("Master Database Connection", value=False)
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            validate_db_connection(
                _db="master", instance_id=GlobalSettings().active_instance_id
            )

        # Check instance database connection
        with (
            ui.expansion("Instance Database Connection", value=False)
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            validate_db_connection(
                _db="instance", instance_id=GlobalSettings().active_instance_id
            )

        # Generate JWT token
        with (
            ui.expansion("JWT Token", value=False)
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            _token_check()

        # Send test email notification
        with (
            ui.expansion("Notification", value=False)
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            _notification_check()


async def _check_db_connection(
    database: SQLDatabaseManager, result_label, button: ui.button
):
    """Asynchronously checks the connection status of a database and updates the UI with the result."""
    with disable(button):
        result_label.text = ""
        try:
            is_connected = await run.io_bound(lambda: database.is_db_connected)
            await asyncio.sleep(0.5)
            if is_connected:
                result_label.text = f"{database.Database} Database connected"
                result_label.style("color:green")
            else:
                result_label.text = "Database not connected"
                result_label.style("color:red")
        except Exception as e:
            result_label.text = f"Error: {e}"
            result_label.style("color: red")


def validate_db_connection(_db: str, instance_id=None):
    """
    Validates and displays information about a database connection for a specified database type.

    Displays:
    --------
        - Database Name
        - Server Name
        - User Name (if not using a trusted connection)
        - Trusted Connection status
    Provides a UI button to check the database connection and displays the result.
    """
    database: SQLDatabaseManager = None
    if _db == "instance":
        database = GlobalSettings().instance_by_id(instance_id).settings.projectdb
    elif _db == "master":
        database = GlobalSettings().instance_by_id(instance_id).settings.masterdb
    elif _db == "worker":
        database = GlobalSettings().workerdb

    # Display connection info
    fields = [
        ("Database Name", database.Database),
        ("Server Name", database.Server),
        ("User Name", None if database.Trusted else database.Username),
        ("Trusted Connection", str(database.Trusted) if database.Trusted else None),
    ]
    fields = [(label, val) for label, val in fields if val is not None]

    with ui.grid(columns=len(fields)).classes("w-full items-start"):
        for label, value in fields:
            with ui.column().classes("gap-1"):
                ui.label(label).classes("text-sm text-gray-500")
                ui.label(value).classes("text-base font-semibold")
    check_btn = ui.button(text="Check Connection")
    result_label = ui.label("").style("min-width: 250px")
    check_btn.on(
        "click",
        lambda: _check_db_connection(
            database, result_label, check_btn
        ),  # pylint: disable=unnecessary-lambda
    )


async def _gen_token(button: ui.button, result_container: ui.column):
    """
    Asynchronously generates an authentication token and updates the UI with the result.
    Args:
        button (ui.button): The button that triggers token generation, which will be disabled during the process.
        result_container (ui.column): The UI container where the token or error message will be displayed.
    Raises:
        RuntimeError: If token generation fails or returns an empty value.
    Side Effects:
        - Disables the button while generating the token.
        - Clears and updates the result container with the generated token or an error message.
        - Enables copying the generated token to the clipboard.
    """
    with disable(button):
        result_container.clear()
        try:
            # Generate the token
            instance = GlobalSettings.instance_by_id(
                GlobalSettings().active_instance_id
            )
            token: Optional[str] = await run.io_bound(
                lambda: generate_token(instance.ClientUID, instance.ProjectUID)
            )
            await asyncio.sleep(0.5)

            if token is None or not token:
                raise RuntimeError("Token generation failed")

            # Convert to string only if token is not None
            token = str(token).replace("Bearer ", "")

            with result_container:
                # enable copying the token
                copy_to_clipboard(token)

        except Exception:
            ui.label("Token generation failed. Check configuration settings.").classes(
                "text-red-500"
            )


def _token_check():
    """
    Displays JWT token information and provides a button to generate a new token.

    This function creates a UI grid with three columns showing the Audience, Issuer,
    and Certificate Type of the JWT from global settings. It also adds a button to
    trigger token generation and a label to display the result.
    """

    jwt = GlobalSettings.instance_by_id(
        GlobalSettings().active_instance_id
    ).settings.jwt
    if not jwt:
        ui.label("JWT settings not found in instance configuration.").classes(
            "text-red-500"
        )
        return

    with ui.grid(columns=3).classes("w-full gap-1 items-start"):
        for label, value in [
            ("Audience", jwt.Audience),
            ("Issuer", jwt.Issuer),
            ("Certificate Type", jwt.CertificateType),
        ]:
            with ui.column().classes("gap-1"):
                ui.label(label).classes("text-sm text-gray-500")
                ui.label(value).classes("text-base font-semibold")
    token_btn = ui.button("Generate Token")
    token_container = ui.column().classes("w-full")
    token_btn.on(
        "click",
        lambda: _gen_token(
            token_btn, token_container
        ),  # pylint:disable=unnecessary-lambda
    )


async def send_test_email(result_label, button: ui.button):
    """Asynchronously sends a test email and updates the UI with the result.
    Args:
        result_label: The UI label component to display the result message.
    Behavior:
    ---------
        - Displays a loader while the email is being sent.
        - Attempts to load SMTP configuration, connect to the email server, attach a log file, and send a test email.
        - Updates the result_label with a success message in green if the email is sent successfully.
        - Updates the result_label with an error message in red if any step fails.
        - Removes the loader after the operation completes.
    """
    with disable(button):
        result_label.text = ""
        try:

            def send_email():
                notifier = EmailNotification(
                    instance_id=GlobalSettings().active_instance_id
                )
                if not notifier.load_config():
                    raise RuntimeError("Failed to load SMTP configuration")

                if not notifier.is_connected():
                    raise RuntimeError("Failed to connect to email server")

                notifier.attach(app_logger.log_file_path)
                notifier.add_content("This is a test email", "Sample content")
                if not notifier.send(subject="Test Email"):
                    raise RuntimeError("Failed to send email. Contact support.")

            await run.io_bound(send_email)
            result_label.text = "Email sent successfully!"
            result_label.style("color: green")
        except BaseException as _e:
            result_label.text = str(_e)
            result_label.style("color: red")


def _notification_check():
    """
    Displays the current notification (SMTP) settings in a grid layout and provides a button to send a test email.

    The function retrieves notification settings from the global configuration, including SMTP server, username, port, sender email, and copy emails.
    Each setting is displayed in a labeled column. If "Copy Emails" is a list, each email is shown individually.
    A "Send Test Email" button is provided to trigger a test email, and the result is displayed below.
    """
    notification = GlobalSettings.instance_by_id(
        GlobalSettings().active_instance_id
    ).settings.notification
    if not notification:
        ui.label("Notification settings not found in instance configuration.").classes(
            "text-red-500"
        )
        return

    with ui.grid(columns=5).classes("w-full gap-1 items-start"):
        for label, value in [
            ("SMTP Server", notification.SMTPServer),
            ("SMTP Username", notification.SMTPUsername),
            ("SMTP Port", notification.SMTPPort),
            ("From Email", notification.FromEmail),
        ]:
            with ui.column().classes("gap-1"):
                ui.label(label).classes("text-sm text-gray-500")
                if label == "Copy Emails" and isinstance(value, list):
                    for email in value:
                        ui.label(email).classes("text-base font-semibold")
                else:
                    ui.label(value).classes("text-base font-semibold")
    test_btn = ui.button("Send Test Email")
    result_label = ui.label("").classes("w-full break-words overflow-auto")
    test_btn.on("click", lambda: send_test_email(result_label, test_btn))


def _broker_check():
    broker = GlobalSettings().broker
    if not broker:
        raise RuntimeError("Broker settings not found in instance configuration.")

    with ui.grid(columns=5).classes("w-full gap-1 items-start"):
        for label, value in [
            ("Broker Host", broker.HostName),
            ("Broker User", broker.UserName),
            ("Broker Port", broker.Port),
            ("Broker VHost", broker.VirtualHost),
            ("Broker SSL", "Enabled" if broker.SSL else "Disabled"),
        ]:
            with ui.column().classes("gap-1"):
                ui.label(label).classes("text-sm text-gray-500")
                ui.label(value).classes("text-base font-semibold")

    broker_btn = ui.button("Check Connection")
    label_placeholder = ui.label("").classes("text-base")
    broker_btn.on(
        "click",
        lambda: _broker_connection_check(label_placeholder, broker_btn),
    )


async def _broker_connection_check(label_placeholder: ui.label, button: ui.button):
    """
    Asynchronously checks the connection to the message broker and updates the UI label and button accordingly.
    Args:
        label_placeholder (ui.label): The UI label to display the connection status.
        button (ui.button): The UI button to be disabled during the check.
    Raises:
        RuntimeError: If the broker is not configured properly.
        Exception: If the connection to the broker fails.
    Side Effects:
        - Updates the text and style of the label_placeholder to indicate success or failure.
        - Disables the button during the connection check.
        - Calls Status.FAILED on connection failure.
    """
    with disable(button):
        label_placeholder.text = ""
        try:
            broker_url = get_broker_url()
            if not broker_url:
                raise RuntimeError("Broker is not configured properly.")

            with Connection(broker_url) as conn:
                await run.io_bound(conn.ensure_connection, max_retries=1)
            await asyncio.sleep(0.5)
            label_placeholder.text = "Broker connection successful!"
            label_placeholder.style("color: green")
        except Exception as e:
            await asyncio.sleep(0.5)
            Status.FAILED("Broker connection failed. Check configuration", error=str(e))
            label_placeholder.text = "Broker connection failed. Check configuration"
            label_placeholder.style("color: red")
