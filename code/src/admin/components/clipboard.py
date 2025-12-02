# # Copyright (C) KonaAI - All Rights Reserved
"""Clipboard utility for displaying and copying text in NiceGUI UI."""
from nicegui import ui


def copy_to_clipboard(text: str):
    """
    Displays the given text in a styled label and provides a button to copy the text to the clipboard.

    Args:
        text (str): The text to display and copy to the clipboard.

    UI Elements:
        - A label showing the text, styled for readability and easy selection.
        - A button with a copy icon that, when clicked, copies the text to the user's clipboard.
    """
    with ui.row().classes("w-full gap-0"):
        ui.label(text=text).style("user-select: all;").classes(
            "break-words overflow-auto font-mono text-sm bg-gray-100 p-2 rounded"
        )
        # Create a button that copies the text to clipboard
        ui.button(icon="copy_all").on(
            "click",
            js_handler=f"""
        () => navigator.clipboard.writeText("{text}")
        """,
        ).props("flat dense").classes("ml-0")
