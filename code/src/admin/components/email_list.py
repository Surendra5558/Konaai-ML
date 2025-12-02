# # Copyright (C) KonaAI - All Rights Reserved
"""This module defines the EmailList component for managing a dynamic list of email addresses."""
from typing import Callable
from typing import List
from typing import Optional

import validators
from nicegui import ui
from nicegui.element import Element
from nicegui.events import ValueChangeEventArguments


class EmailList(Element):
    """
    EmailList is a UI component for managing a dynamic list of email addresses.
    This component allows users to add, edit, and remove email addresses within a specified maximum count. It provides validation for email format and uniqueness, and supports registering callbacks to react to changes in the email list. The UI consists of labeled input fields for each email, delete buttons, and an "Add Email" button when the maximum count is not reached.
    Attributes:
        max_count (Optional[int]): The maximum number of email addresses allowed. If set to 0, unlimited emails can be added.
        emails (List[str]): The current list of email addresses.
        label (str): The label displayed above the email list input fields.
    """

    max_count: Optional[int] = 1
    emails: List[str] = []
    label: str = "Email Recipients"

    def __init__(
        self,
        label: str,
        emails: Optional[List[str]] = None,
        max_count: Optional[int] = 1,
    ):
        super().__init__("div")  # Initialize as a div element
        self._updating_ui = False  # Initialize flag IMMEDIATELY after super() to prevent any update calls
        self._on_change_callbacks = []

        self.classes("flex flex-col gap-2")  # vertical stack with spacing
        self.max_count = max_count
        self.emails = emails or []
        self.label = label
        self.update()

    def update(self):
        """
        Updates the email list UI component.
        This method refreshes the UI to display the current list of emails, allowing users to edit or delete existing emails,
        and add new ones up to a specified maximum count. It prevents recursive updates by using an internal flag.
        The UI consists of:
            - A label and separator at the top.
            - A list of input fields for each email, each with validation for email format and uniqueness.
            - A delete button next to each email input to remove the email.
            - An "Add Email" button if the maximum count is not reached.
        Returns:
            None
        """

        # Prevent recursive updates
        if self._updating_ui:
            return

        self._updating_ui = True
        try:
            # Clear existing content first
            self.clear()

            with self:  # Use self as the context instead of creating new column
                with (
                    ui.column().classes("w-full items-start gap-2"),
                    ui.card().classes("w-full"),
                ):
                    ui.label(self.label).classes("font-normal text-gray-500 text-lg")
                    ui.separator().classes("w-full mb-2")
                    for i, email in enumerate(self.emails):
                        # show email input with delete button
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            # Input for email address
                            email_input = ui.input(
                                label=f"Email {i + 1}",
                                value=email,
                                validation={
                                    "Invalid Email": lambda v: (
                                        validators.email(v) if v else True
                                    ),
                                    "Already exists": lambda v, index=i: (
                                        not self._value_exists(v, index) if v else True
                                    ),
                                },
                            ).classes("flex-grow w-full")
                            email_input.on_value_change(
                                lambda x, index=i: self._add_value(index, x)
                            )

                            # Delete button for each email input
                            (
                                ui.button(
                                    icon="close",
                                    on_click=lambda index=i: self._delete_email_field(
                                        index
                                    ),
                                )
                                .props("flat dense")
                                .classes("bg-red text-white size=xssm")
                            )

                    if len(self.emails) < self.max_count or self.max_count == 0:
                        ui.button(icon="add", text="Add Email", color="primary").props(
                            "flat dense"
                        ).on_click(
                            lambda: self._add_email_field()  # pylint: disable=unnecessary-lambda
                        )
        finally:
            self._updating_ui = False

    def _value_exists(self, value: str, index: int) -> bool:
        return any(item == value and i != index for i, item in enumerate(self.emails))

    def _add_value(self, index: int, e: ValueChangeEventArguments):
        """Update the email input value."""
        value = e.value
        if validators.email(value):
            self.emails[index] = value
            # Notify all registered callbacks about the change
            for callback in self._on_change_callbacks:
                callback(self.emails)

    def _add_email_field(self):
        """Add a new email input row."""
        self.emails.append("")  # Add an empty email entry
        self.update()  # Re-render the form to include the new input

    def _update_email(self, index: int, value: str):
        """Update email at specific index."""
        if 0 <= index < len(self.emails):
            self.emails[index] = value
            # Notify all registered callbacks about the change
            for callback in self._on_change_callbacks:
                callback(self.emails)

    def _delete_email_field(self, index: int):
        """Delete a specific email input row by index."""
        if 0 <= index < len(self.emails):
            self.emails.pop(index)
            self.update()

            # Notify all registered callbacks about the change
            for callback in self._on_change_callbacks:
                callback(self.emails)

    def on_change(self, callback: Callable):
        """
        Registers a callback function to be called when a change event occurs.

        Args:
            callback (Callable): The function to be called when the change event is triggered.
        """
        self._on_change_callbacks.append(callback)

    def bind_value_to(self, model: object, field_name: str) -> "EmailList":
        """
        Binds the current list of emails to a specified field of a given model object.
        This method sets up a callback so that whenever the email list changes, the specified
        field on the model is updated with the new list of emails. It also immediately updates
        the model's field with the current email list.
        Args:
            model (object): The object whose field will be updated with the email list.
            field_name (str): The name of the field on the model to bind the email list to.
        Returns:
            EmailList: The current instance to allow method chaining.
        """

        def update_model(emails: List[str]):
            setattr(model, field_name, emails)

        self.on_change(update_model)
        update_model(self.emails)
        return self
