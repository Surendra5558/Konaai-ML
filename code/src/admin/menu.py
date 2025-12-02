# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the menu configuration settings."""
from typing import Dict
from typing import List

from nicegui import app
from nicegui import ui


def logout():
    """Clears the user's session data and redirects to the login page."""

    app.storage.user.clear()
    ui.navigate.to("/login")


class Menu:
    """
    Configures the menu items for the application.
    """

    def items(self) -> List[Dict]:
        """
        Returns a list of menu items for the admin interface.

        Each menu item is represented as a dictionary with the following keys:
        - "title": The display name of the menu item.
        - "target": The URL path the menu item links to.

        Returns:
            List[Dict]: A list of dictionaries representing the menu items.
        """
        return [
            {"title": "Home", "target": "/"},
            {"title": "Global Settings", "target": "/global_settings"},
            {"title": "Instances", "target": "/instance"},
            {"title": "Anomaly Pipeline", "target": "/anomaly_pipeline"},
            {"title": "AutoML Management", "target": "/automl_mgmt"},
            {"title": "Health Check", "target": "/health_check"},
            {"title": "Tasks Management", "target": "/task_mgmt"},
            {"title": "Insight Management", "target": "/insight_management"},
            {"title": "SQL Agent", "target": "/sql_agent"},
            {"title": "Backup & Restore", "target": "/backup_restore"},
            {"title": "Logs", "target": "/logs"},
            {"title": "Logout", "target": "/login", "action": logout},
        ]

    def render(self):
        """
        Renders the menu items as UI components.

        This method iterates through the menu items and creates a link for each item
        with the specified title and target. Each link is styled with specific CSS
        classes. A separator is added after each link for visual separation.

        Returns:
            None
        """
        for item in self.items():
            ui.link(text=item.get("title"), target=item.get("target")).classes(
                replace="text-black w-full"
            )
            ui.separator().classes("w-full")
