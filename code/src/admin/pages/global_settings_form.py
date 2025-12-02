# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a user interface for configuring and managing global application settings."""
from enum import Enum

from nicegui import ui
from src.admin.pages.broker_form import BrokerSettingsForm
from src.admin.pages.database_form import DatabaseForm
from src.utils.database_config import SQLDatabaseManager
from src.utils.global_config import GlobalSettings


class GlobalSettingsType(Enum):
    """Enum for different settings types."""

    Broker = "Broker"
    WorkerDB = "Worker Database"


class GlobalSettingsForm:
    """
    Form for configuring global application settings.
    This class provides a user interface for selecting and configuring different types of global settings,
    such as Broker and Worker Database settings. It dynamically renders the appropriate configuration form
    based on the selected settings type and handles saving the updated settings.
    Methods
    """

    def __init__(self):
        """Initialize the form."""
        self.settings = GlobalSettings()

    def render(self):
        """
        Renders the global settings form UI.
        Displays a dropdown to select the settings type and renders the corresponding widget
        based on the selected value. The widget updates dynamically when the selection changes.
        """
        with ui.row().classes("items-center w-full"):
            setting_type = ui.select(
                label="Select Settings Type",
                options=[st.value for st in GlobalSettingsType],
                value=None,
            ).classes("w-full")

        self._render_widget(setting_type.value)
        setting_type.on_value_change(
            lambda: self._render_widget.refresh(setting_type.value)
        )

    @ui.refreshable_method
    def _render_widget(self, setting_type: str):
        """
        Renders the appropriate settings widget based on the provided setting type.
        Parameters:
            setting_type (str): The type of global setting to render. Determines which settings form is displayed.
        Behavior:
            - If `setting_type` is empty or None, no widget is rendered.
            - For 'Broker' type, displays the Broker Settings Configuration form.
            - For 'WorkerDB' type, displays the Worker Database Settings Configuration form.
            - The rendered widget is styled with a card layout and appropriate labels.
        Returns:
            None
        """
        if not setting_type:
            return

        with ui.card().classes("w-full p-6 space-y-4 bg-white shadow-md rounded-lg"):

            if setting_type == GlobalSettingsType.Broker.value:
                ui.label("Broker Settings Configuration").classes(
                    "text-lg font-semibold"
                )
                BrokerSettingsForm().render()
            elif setting_type == GlobalSettingsType.WorkerDB.value:
                ui.label("Worker Database Settings Configuration").classes(
                    "text-lg font-semibold"
                )
                self._render_workerdb()

    def _render_workerdb(self):
        """
        Renders the Worker Database settings form.
        This method initializes the worker database settings if not already set,
        displays the database configuration form, and adds a "Save Settings" button
        to persist any changes made by the user.
        """

        if self.settings.workerdb is None:
            self.settings.workerdb = SQLDatabaseManager()
        DatabaseForm(self.settings.workerdb).render()

        # Save button
        with ui.row().classes("w-full justify-end mt-4"):
            ui.button(text="Save Settings", icon="save", color="primary").on_click(
                lambda: self._save_settings()  # pylint: disable=unnecessary-lambda
            )

    def _save_settings(self):
        """
        Saves the current global settings for the worker database.

        This method attempts to persist the updated worker database settings using the GlobalSettings class.
        On successful save, a positive notification is displayed to the user.
        If an error occurs during the save process, a negative notification with the error message is shown.

        Raises:
            Exception: If saving the settings fails.
        """
        try:
            # Save the updated settings
            settings = GlobalSettings()
            settings.workerdb = self.settings.workerdb
            settings.save_settings()
            ui.notify("Worker Database settings saved successfully", type="positive")
        except Exception as e:
            ui.notify(f"Failed to save settings: {e}", type="negative")
