# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a function to create spinners and loaders in the current UI context."""
from nicegui import ui


def create_overlay_spinner(message: str = None) -> ui.element:
    """
    Create a full-screen overlay spinner with an optional message.

    This spinner appears centered on a semi-transparent dark background,
    blocking user interaction with the rest of the UI. It is useful for
    indicating a global loading or blocking state.

    Args:
        message (str, optional): A text message to display above the spinner. Defaults to None.

    Returns:
        ui.element: A NiceGUI element representing the overlay spinner.
    """
    with ui.element("div").classes(
        "absolute inset-0 flex items-center justify-center bg-black/40 z-50"
    ) as overlay:
        with ui.column().classes("items-center"):
            if message:
                (ui.label(message).classes("text-white text-lg mb-4"))
            ui.spinner(size="64px", color="#ffffff")
    return overlay


def create_loader(message: str = None) -> ui.element:
    """
    Create an inline loader spinner with an optional message.

    This loader appears as a row with a "dots"-style spinner and a label,
    suitable for use inside components where local loading feedback is needed.

    Args:
        message (str, optional): A message to display next to the spinner. Defaults to None.

    Returns:
        ui.element: A NiceGUI element representing the inline loader.
    """

    with ui.row().classes("items-center") as loader:
        ui.spinner(type="dots", size="lg")
        if message:
            ui.label(message).classes("text-lg ml-2")

    return loader
