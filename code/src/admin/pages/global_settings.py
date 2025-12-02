# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the global settings page."""
from nicegui import ui
from src.admin import theme
from src.admin.pages.global_settings_form import GlobalSettingsForm


def global_settings():
    """
    Renders the Global Settings page within a themed frame.

    This function displays a header and renders the GlobalSettingsForm
    for managing application-wide settings.

    Returns:
        None
    """
    with theme.frame("Settings"):
        ui.markdown("# Settings Management")
        # Render Global settings form
        GlobalSettingsForm().render()
