# # Copyright (C) KonaAI - All Rights Reserved
"""This is UI form for database configuration"""
import asyncio
from contextlib import contextmanager

from nicegui import run
from nicegui import ui
from pydantic import SecretStr
from src.admin.components.spinners import create_loader
from src.utils.database_config import SQLDatabaseManager


@contextmanager
def disable(button: ui.button):
    """
    Temporarily disables a UI button during an operation.

    This context manager disables the given button when entering the context,
    and re-enables it upon exit, ensuring the button is not interactable
    while the enclosed operation is running.

    Args:
        button (ui.button): The button to disable during the operation.
    """
    try:
        yield
    finally:
        button.enable()


class DatabaseForm:
    """
    A form component for configuring and testing SQL database connections.
    Attributes:
        model (SQLDatabaseManager): An instance managing the SQL database connection details.
    """

    model: SQLDatabaseManager = None

    def __init__(self, model: SQLDatabaseManager = None):
        """Initialize the DatabaseForm with a SQLDatabaseManager instance."""
        self.model = model

    def render(self):
        """
        Renders the database configuration form UI.
        The form includes input fields for server name, database name, and a switch for trusted connection.
        If the trusted connection is disabled, credential fields are rendered.
        The credential fields are dynamically refreshed when the trusted connection value changes.
        """
        with ui.card().classes("w-full p-6 space-y-4 bg-white shadow-md rounded-lg"):
            (
                ui.input(label="Server name", value=self.model.Server)
                .bind_value_to(self.model, "Server")
                .classes("w-full")
            )

            (
                ui.input(label="Database name", value=self.model.Database)
                .bind_value_to(self.model, "Database")
                .classes("w-full")
            )

            trusted = ui.switch(
                text="Trusted connection", value=self.model.Trusted
            ).bind_value_to(self.model, "Trusted")
            self._render_credentials_form(trusted.value)
            trusted.on_value_change(
                lambda: self._render_credentials_form.refresh(trusted.value)
            )

    @ui.refreshable_method
    def _render_credentials_form(self, trusted: bool):
        if not trusted:
            (
                ui.input(label="User name", value=self.model.Username)
                .bind_value_to(self.model, "Username")
                .classes("w-full")
            )
            password = ui.input(
                label="Password",
                value=(
                    self.model.Password.get_secret_value()
                    if self.model and self.model.Password
                    else ""
                ),
                password=True,
                password_toggle_button=True,
            ).classes("w-full")
            # Bind the password to the model
            password.bind_value_to(
                self.model,
                "Password",
                forward=lambda v: (
                    SecretStr(v) if isinstance(v, str) else v
                ),  # pylint: disable=unnecessary-lambda
            )

        check_btn = ui.button(text="Check Connection")
        check_btn.on_click(
            lambda: self._check_connectivity(
                check_btn
            )  # pylint: disable=unnecessary-lambda
        )

    async def _check_connectivity(self, button: ui.button):
        """
        Asynchronously checks the connectivity status of the database when triggered by a button click.
        Disables the button during the check, displays a loading indicator, and notifies the user of the result.
        If the database is connected, shows a positive notification; otherwise, shows a negative notification.
        Handles exceptions by notifying the user of any errors encountered during the connectivity check.
        Args:
            button (ui.button): The button that triggers the connectivity check.
        """

        with disable(button):
            loader = create_loader()
            try:
                is_connected = await run.io_bound(lambda: self.model.is_db_connected)
                await asyncio.sleep(0.5)
                if is_connected:
                    ui.notify(
                        "Database connected successfully",
                        type="positive",
                    )
                else:
                    ui.notify(
                        "Database not connected",
                        type="negative",
                    )
            except Exception as e:
                ui.notify(
                    f"Error connecting to database: {e}",
                    type="negative",
                )
            finally:
                loader.delete()
