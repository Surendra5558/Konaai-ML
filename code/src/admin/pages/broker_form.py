# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a UI form for configuring broker connection settings."""
from nicegui import ui
from pydantic import ValidationError
from src.utils.global_config import BrokerModel
from src.utils.global_config import GlobalSettings
from src.utils.status import Status


class BrokerSettingsForm:
    """
    A NiceGUI-based form for configuring message broker settings.

    This class provides a form UI for configuring broker parameters and saving
    them to global application settings. It supports input validation using
    a Pydantic model and persists changes using the `GlobalSettings` interface.
    """

    def __init__(self):
        """
        Initializes the class instance and sets up the widgets dictionary.
        Attributes:
            widgets (dict): A dictionary to store widget instances.
        """

        self.widgets = {}

    def render(self):
        """
        Renders the broker settings form UI.
        This method displays a card containing input fields for broker configuration,
        including User Name, Password, Host Name, Virtual Host, Port, Consumer Timeout (hrs),
        Heartbeat (seconds), and an SSL switch. For sensitive fields such as Consumer Timeout
        and Heartbeat, warning messages are shown to advise users not to change values unless instructed.
        A "Save Settings" button is provided to submit the form.
        The form fields are initialized with values from the current global broker settings,
        or default values if not set.
        Widgets are stored in `self.widgets` for later access.
        """

        broker_data = GlobalSettings().broker or BrokerModel()
        with ui.card().classes("w-full p-6 space-y-4 bg-white shadow-md rounded-lg"):
            self.widgets = {
                "UserName": ui.input("User Name", value=broker_data.UserName).classes(
                    "w-full"
                ),
                "Password": ui.input(
                    label="Password",
                    value=(
                        broker_data.Password.get_secret_value()
                        if broker_data.Password
                        else ""
                    ),
                    password=True,
                    password_toggle_button=True,
                ).classes("w-full"),
                "HostName": ui.input("Host Name", value=broker_data.HostName).classes(
                    "w-full"
                ),
                "VirtualHost": ui.input(
                    "Virtual Host", value=broker_data.VirtualHost
                ).classes("w-full"),
                "Port": ui.number(
                    label="Port", value=broker_data.Port, min=1, max=65535
                ).classes("w-full"),
            }
            # ConsumerTimeoutHrs with warning
            with ui.column().classes("w-full gap-0 mb-0"):
                self.widgets["ConsumerTimeoutHrs"] = ui.number(
                    label="Consumer Timeout (hrs)",
                    value=broker_data.ConsumerTimeoutHrs,
                    min=0,
                    max=168,
                ).classes("w-full")
                with ui.row().classes("items-center ml-1 mt-1 mb-2 gap-1"):
                    ui.icon("warning", color="orange", size="xxs")
                    ui.label("  Don't change value, unless instructed.").classes(
                        "text-xs text-gray-500 m-0"
                    )

            # HeartbeatSeconds with warning
            with ui.column().classes("w-full gap-0 mb-0"):
                self.widgets["HeartbeatSeconds"] = ui.number(
                    label="Heartbeat (seconds)",
                    value=broker_data.HeartbeatSeconds,
                    min=0,
                    max=86400,
                ).classes("w-full")
                with ui.row().classes("items-center ml-1 mt-1 mb-2 gap-1"):
                    ui.icon("warning", color="orange", size="xxs")
                    ui.label("Don't change value, unless instructed. ").classes(
                        "text-xs text-gray-500 m-0"
                    )
            # SSL switch
            self.widgets["SSL"] = ui.switch("SSL", value=broker_data.SSL).classes(
                "w-full"
            )
            with ui.row().classes("w-full justify-end mt-4"):
                ui.button(
                    text="Save Settings",
                    icon="save",
                    on_click=self.on_submit,
                    color="primary",
                ).classes("mt-2")

    def on_submit(self):
        """
        Handles the submission of the broker configuration form.
        Collects input values from form widgets, validates them using the BrokerModel,
        updates the global broker settings, and attempts to save the settings.
        Provides user notifications for success, validation errors, or other exceptions.
        Exceptions:
            ValidationError: Raised if input validation fails.
            Exception: Catches any other errors during the save process.
        """
        try:
            # Gather input values from widgets
            broker_input = {k: v.value for k, v in self.widgets.items()}

            broker_model = BrokerModel(**broker_input)
            settings = GlobalSettings()
            settings.broker = broker_model
            if settings.save_settings():
                ui.notify(
                    "Broker settings saved successfully.",
                    type="positive",
                    position="bottom",
                )
            else:
                Status.FAILED("Failed to save Broker settings.")
                ui.notify(
                    "Failed to save Broker settings. Contact support.",
                    type="negative",
                    position="top",
                )
        except ValidationError as e:
            ui.notify(str(e), type="negative", position="top")
        except Exception as ex:
            Status.FAILED("Error saving Broker settings", error=ex)
            ui.notify("Failed to save Broker settings.", type="negative")
