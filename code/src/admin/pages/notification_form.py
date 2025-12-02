# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a form for configuring notification settings."""
import validators
from nicegui import ui
from pydantic import SecretStr
from src.admin.components.email_list import EmailList
from src.utils.smtp_model import NotificationModel


class NotificationSettingsForm:
    """
    NotificationSettingsForm is a UI form class for configuring notification settings.

    Attributes:
    ----------
        model (NotificationModel): The data model instance containing notification settings.

    Renders:

        Renders the notification settings form UI, allowing users to configure:
                - SMTP server address
                - SMTP port
                - SMTP username
                - SMTP password (with toggle visibility)
                - From email address (with validation)
                - Recipient email list (with optional maximum count)
    """

    model: NotificationModel = None

    def __init__(self, model: NotificationModel):
        self.model = model or NotificationModel()

    def render(self):
        """
        Renders the notification settings form UI for configuring SMTP email notifications.

        The form includes the following fields:

            - SMTP Server: Text input for the SMTP server address.
            - SMTP Port: Numeric input for the SMTP port (range: 1-65535).
            - SMTP Username: Text input for the SMTP username.
            - SMTP Password: Password input with toggle visibility, securely handled.
            - From Email: Text input for the sender's email address, with email validation.
            - Recipient Emails: Custom email list input for specifying recipient addresses (up to 15).
        All fields are bound to the corresponding attributes of the model for two-way data binding.
        """
        with ui.card().classes("w-full p-6 space-y-4 bg-white shadow-md rounded-lg"):
            ui.input(
                "SMTP Server",
                value=self.model.SMTPServer,
                placeholder="Enter SMTP server address",
            ).bind_value_to(self.model, "SMTPServer").classes("w-full")

            ui.number(
                "SMTP Port",
                value=self.model.SMTPPort,
                min=1,
                max=65535,
                step=1,
                placeholder="Enter SMTP port",
                format="%d",
            ).bind_value_to(self.model, "SMTPPort").classes("w-full")

            ui.input(
                "SMTP Username",
                value=self.model.SMTPUsername,
                placeholder="Enter SMTP username",
            ).bind_value_to(self.model, "SMTPUsername").classes("w-full")

            password = ui.input(
                label="SMTP Password",
                value=(
                    self.model.SMTPPassword.get_secret_value()
                    if self.model.SMTPPassword
                    else ""
                ),
                password=True,
                password_toggle_button=True,
            ).classes("w-full")
            password.bind_value_to(
                self.model,
                "SMTPPassword",
                forward=lambda v: (SecretStr(v) if isinstance(v, str) else v),
            )

            ui.input(
                "From Email",
                value=self.model.FromEmail,
                placeholder="Enter from email address",
                validation={
                    "Invalid Email": lambda v: validators.email(v) if v else True,
                },
            ).bind_value_to(self.model, "FromEmail").classes("w-full")

            EmailList(
                label="Recipient Emails",
                emails=self.model.RecipientEmails or [],
                max_count=15,  # Set a max if you want to limit
            ).bind_value_to(self.model, "RecipientEmails").classes("w-full")
