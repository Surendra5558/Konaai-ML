# # Copyright (C) KonaAI - All Rights Reserved
"""Active Instance UI Component"""
from typing import Optional

from nicegui import ui
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance


class ActiveInstanceUI:
    """Class to handle the active instance UI component."""

    active_instance: Optional[Instance] = None

    def __init__(self):
        # Check if an active instance is set
        if GlobalSettings().active_instance_id:
            if instance_obj := GlobalSettings.instance_by_id(
                GlobalSettings().active_instance_id
            ):
                self.active_instance = instance_obj
            else:
                ui.label("No active instance found.").classes("text-red-500")
                ui.notify("Active instance not found", type="negative")

    def render(self):
        """
        Renders the active instance details in a 3-column grid layout if an active instance exists.
        Displays the instance ID, client name, and project name with appropriate labels and styling.
        If no active instance is present, shows a notification prompting the user to activate an instance.
        """
        # Show active instance
        if self.active_instance:
            with ui.grid(columns=3).classes("w-full gap-1 items-start"):
                for label, value in [
                    ("Active Instance ID", self.active_instance.instance_id),
                    ("Client Name", self.active_instance.client_name),
                    ("Project Name", self.active_instance.project_name),
                ]:
                    with ui.column().classes("gap-1"):
                        ui.label(label).classes("text-sm text-gray-500")
                        ui.label(value).classes("text-base font-semibold")
        else:
            ui.notify("Activate an instance to continue", type="negative")
