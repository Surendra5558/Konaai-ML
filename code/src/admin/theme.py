# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the theme configuration settings."""
import os
from contextlib import contextmanager

from nicegui import ui
from src.admin.menu import Menu
from src.utils.conf import config
from src.utils.conf import ENVIRONMENT_VAR_NAME


@contextmanager
def frame(navtitle: str):
    """
    Creates a custom page frame to ensure consistent styling and behavior across all pages.

    Args:
        navtitle (str): The title to display in the navigation header.

    Behavior:

        - Sets the primary color scheme for the UI.
        - Renders a centered column layout for page content.
        - Displays a header with the navigation title and a logout button if the user is authenticated.
        - Shows a left navigation drawer (except on the Login page) with menu options.
        - Displays a footer with version information and the current environment variable status.
    Yields:
        The main content area for the page, allowing child components to be rendered within the frame.
    """
    ui.colors(
        primary="#003057", secondary="#26A69A", accent="#59B84B", positive="#53B689"
    )

    with ui.column().classes(
        "absolute-center items-center h-screen no-wrap p-5 w-full"
    ):
        yield

    if navtitle != "Login":
        # Left drawer for navigation
        with ui.left_drawer().style("background-color: #E8EEF2"):
            ui.label("Navigation").classes("font-bold text-xl")
            with ui.column().classes("w-full gap-2"):
                Menu().render()

    # show version info in the footer
    with ui.footer().classes("w-full p-1 bg-primary text-white text-center"):
        with ui.row(align_items="stretch").classes("w-full"):
            # show the version info
            ui.label(f"Version: {config.get('PROJECT_INFO', 'VERSION')}").classes(
                "text-sm"
            )

            ui.space().classes("grow")

            # get the environment variable
            env_var = os.getenv(ENVIRONMENT_VAR_NAME) or os.getenv(
                str(ENVIRONMENT_VAR_NAME).lower()
            )
            if env_var := env_var.replace('"', "").replace("'", "").strip():
                ui.label(f"{ENVIRONMENT_VAR_NAME}: {env_var}").classes("text-sm")
            else:
                ui.label(f"{ENVIRONMENT_VAR_NAME} is not set.").classes("text-sm")
