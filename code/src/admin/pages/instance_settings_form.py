# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a form for configuring instance settings."""
from enum import Enum

from nicegui import ui
from src.admin.pages.api_client_ui import APIClientForm
from src.admin.pages.database_form import DatabaseForm
from src.admin.pages.endpoint_form import EndPointForm
from src.admin.pages.jwt_form import JWTSettingsForm
from src.admin.pages.notification_form import NotificationSettingsForm
from src.insight_agent.UI.llm_config_ui import LLMConfigUI
from src.utils.api_config import APIClient
from src.utils.database_config import SQLDatabaseManager
from src.utils.instance import Instance
from src.utils.instance import InstanceSettings
from src.utils.jwt_model import JWTModel
from src.utils.llm_config import BaseLLMConfig
from src.utils.smtp_model import NotificationModel


class InstanceSettingsType(Enum):
    """Enum for different settings types."""

    JWT = "JWT"
    Master_Database = "Master Database"
    Project_Database = "Project Database"
    Notification = "Notification"
    Application_API = "Application API Client"
    Anomaly_Callback = "Anomaly Callback Endpoint"
    Automl_Callback = "AutoML Callback Endpoint"
    LLM_Configuration = "LLM Configuration"


class InstanceSettingsForm:
    """
    Form for configuring and managing instance settings in the admin interface.
    This class provides a dynamic UI form that allows users to select and configure
    various settings types for an instance, such as JWT, notification, database,
    API client, and callback endpoints. The form adapts its fields based on the
    selected settings type and supports saving changes to the instance.
    Attributes:
    -----------
        instance (Instance): The instance whose settings are being configured.
        form_container: The UI container for rendering dynamic forms.
        email_form: Reference to the email recipient form, if applicable.
        setting_type_dropdown: Dropdown UI element for selecting the settings type.
    """

    instance: Instance

    def __init__(self, instance: Instance):
        """Initialize the form with an instance."""
        self.instance = instance
        self.instance._load_settings()

        self.form_container = None  # container for forms
        self.email_form = None  # reference to EmailRecipientForm
        self.setting_type_dropdown = None  # dropdown for switching forms

    def render(self):
        """
        Render the instance settings form UI.
        This method creates a dropdown menu for selecting the settings type and sets up a dynamic form container.
        When the dropdown value changes, the corresponding settings widget is rendered in the form container.
        """

        # create a drop down to select settings type
        self.setting_type_dropdown = ui.select(
            label="Select Settings Type",
            options=[st.value for st in InstanceSettingsType],
            value=None,
        ).classes("w-full max-w-xs mb-2")

        self.setting_type_dropdown.on_value_change(
            lambda: self._render_widget(self.setting_type_dropdown.value)
        )

        # Persistent form container for dynamic forms
        self.form_container = ui.column().classes("w-full")

    def _render_widget(self, setting_type: str):
        """Render the widget based on the selected settings type."""
        self.form_container.clear()  # Only clear UI, not data

        # Ensure InstanceSettings object exists
        if self.instance.settings is None:
            self.instance.settings = InstanceSettings()

        # Database settings form
        if setting_type == InstanceSettingsType.JWT.value:
            if self.instance.settings.jwt is None:
                self.instance.settings.jwt = JWTModel()
            with self.form_container:
                JWTSettingsForm(self.instance).render()
        # Notification settings form
        elif setting_type == InstanceSettingsType.Notification.value:
            if self.instance.settings.notification is None:
                self.instance.settings.notification = NotificationModel()
            with self.form_container:
                NotificationSettingsForm(self.instance.settings.notification).render()
        # Master Database settings form
        elif setting_type == InstanceSettingsType.Master_Database.value:
            if self.instance.settings.masterdb is None:
                self.instance.settings.masterdb = SQLDatabaseManager()
            with self.form_container:
                DatabaseForm(self.instance.settings.masterdb).render()
        # Project Database settings form
        elif setting_type == InstanceSettingsType.Project_Database.value:
            if self.instance.settings.projectdb is None:
                self.instance.settings.projectdb = SQLDatabaseManager()
            with self.form_container:
                DatabaseForm(self.instance.settings.projectdb).render()
        # Application API settings form
        elif setting_type == InstanceSettingsType.Application_API.value:
            if self.instance.settings.application_api_client is None:
                self.instance.settings.application_api_client = APIClient()
            with self.form_container:
                APIClientForm(self.instance.settings.application_api_client).render()
        # Anomaly Callback Endpoint form
        elif setting_type == InstanceSettingsType.Anomaly_Callback.value:
            if self.instance.settings.anomaly_callback_endpoint is None:
                self.instance.settings.anomaly_callback_endpoint = EndPointForm(
                    self.instance.settings.anomaly_callback_endpoint
                )
            with self.form_container:
                EndPointForm(self.instance.settings.anomaly_callback_endpoint).render()
        # Automl Callback Endpoint form
        elif setting_type == InstanceSettingsType.Automl_Callback.value:
            if self.instance.settings.automl_callback_endpoint is None:
                self.instance.settings.automl_callback_endpoint = EndPointForm(
                    self.instance.settings.automl_callback_endpoint
                )
            with self.form_container:
                EndPointForm(self.instance.settings.automl_callback_endpoint).render()
        # LLM Configuration form
        elif setting_type == InstanceSettingsType.LLM_Configuration.value:
            if self.instance.settings.llm_config is None:
                self.instance.settings.llm_config = BaseLLMConfig()
            with self.form_container:
                LLMConfigUI(self.instance.settings.llm_config).render()

        # Save button
        with self.form_container:
            with ui.row().classes("w-full justify-end"):
                ui.button(text="Save Settings", icon="save", color="primary").on_click(
                    lambda: self._save_settings()  # pylint: disable=unnecessary-lambda
                )

    def _save_settings(self):
        """
        Saves the current instance settings.
        If an email form is present, updates the recipient emails from the form fields.
        Validates the emails before saving; if invalid, aborts the save operation.
        Attempts to save the instance settings and notifies the user of success or failure.
        Returns:
            None
        """
        # If email form is present, update recipient_emails from form fields
        if self.email_form:
            emails = self.email_form.get_emails()
            if emails is None:
                # Invalid email(s), do not proceed or show success notification
                return
            self.instance.settings.recipient_emails = emails

        if self.instance.save_settings():
            ui.notify(
                "Settings saved successfully",
                type="positive",
            )
        else:
            ui.notify(
                "Failed to save settings",
                type="negative",
            )
