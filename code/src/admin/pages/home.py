# # Copyright (C) KonaAI - All Rights Reserved
"""_summary_ This module contains the home page"""
import pathlib

from nicegui import ui
from src.admin import theme
from src.utils.conf import Setup


def home():
    """
    Renders the home page of the KonaAI Intelligence Server with a welcome label and logo.

    This function uses the NiceGUI library to create a styled home page. It displays a welcome
    message and a logo image that is dynamically loaded based on a configuration setting. The
    elements are styled with a custom theme frame and layout classes to ensure proper positioning
    and styling.
    """

    with theme.frame("Home"):
        # Add spacing above the label to move it down
        ui.label("Welcome to the KonaAI Intelligence Server").classes(
            "text-lg font-bold mt-16"
        )
        with ui.column().classes("w-full h-full justify-center items-center gap-8"):
            # Increase logo size further
            logo_file_name = (
                Setup().global_constants.get("ASSETS", {}).get("LOGO_FILE", "")
            )
            logo_file_path = pathlib.Path(Setup().assets_path, logo_file_name)
            ui.image(logo_file_path).classes("w-[700px] h-auto max-w-full")
