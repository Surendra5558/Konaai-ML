# # Copyright (C) KonaAI - All Rights Reserved
# Copyright (C) KonaAI - All Rights Reserved
"""This module contains the backup and restore app"""
import os
import zipfile
from contextlib import contextmanager
from datetime import datetime

from nicegui import run
from nicegui import ui
from nicegui.events import UploadEventArguments
from src.admin import theme
from src.admin.components.spinners import create_loader
from src.utils.conf import Setup
from src.utils.file_mgmt import FileHandler
from src.utils.status import Status


@contextmanager
def disable(button: ui.button):
    """
    Temporarily disable a UI button during an operation.

    Args:
        button (ui.button): The button to be disabled.

    Usage:
        with disable(my_button):
            # Do some operation
    """
    button.disable()
    try:
        yield
    finally:
        button.enable()


def backup_restore():
    """
    Render the Backup and Restore UI page.

    Features:
        - Backup: Compress and download current database contents.
        - Restore: Upload and extract a ZIP backup to restore the database.
    """
    with theme.frame("Backup and Restore"):
        ui.markdown("# Backup and Restore")

        with ui.column().classes("w-full"):
            with ui.column().classes("w-full gap-2"):
                download_btn = ui.button("Backup Download")
                download_btn.on(
                    "click",
                    lambda: backup_and_download(
                        backup_label, download_btn
                    ),  # pylint: disable=unnecessary-lambda
                )
                backup_label = ui.label("").style("color: green")

            with ui.column().classes("w-full gap-2"):
                # Place spinner and label below the upload widget
                ui.upload(
                    label="Restore backup",
                    on_upload=lambda e: restore_from_file(e, restore_label),
                    auto_upload=True,
                    max_files=1,
                    max_total_size=5 * 1024 * 1024 * 1024,  # 5GB
                ).classes("w-full")

                restore_label = ui.label("").style("color: green")


async def backup_and_download(label: ui.element, button: ui.button) -> None:
    """
    Compress the database folder into a ZIP archive and trigger download.

    Args:
        label (ui.element): The label to show success/failure messages.
        button (ui.button): The button triggering the action (to be disabled temporarily).

    Process:
        - Validates if there is data to backup.
        - Zips the content asynchronously using NiceGUI's `run.io_bound`.
        - Initiates download in the browser.
        - Displays relevant success or error messages.
    """
    # Show spinner, clear label
    with disable(button):
        loader = create_loader()
        label.text = ""
        try:
            ui.notify("Backing up data", type="info")

            source_path = Setup().db_path
            if not os.path.exists(source_path) or not os.listdir(source_path):
                ui.notify("No data to backup", type="warning")
                return

            file_handler = FileHandler()
            _, target_path = file_handler.get_new_file_name(file_extension="zip")

            # Move zipping to background thread
            def zip_folder():
                with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(source_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(
                                file_path, os.path.relpath(file_path, source_path)
                            )

            await run.io_bound(zip_folder)

            today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            download_filename = f"backup_{today}.zip"

            # Use file path directly for immediate download
            ui.download(src=target_path, filename=download_filename)
            label.text = "Backup complete. Check downloads."
            label.style("color: green")
        except BaseException as _e:
            Status.FAILED("Error backing up data", error=_e)
            ui.notify("Something went wrong. Contact support.", type="negative")
        finally:
            loader.delete()


# Update restore_from_file to accept spinner and label
async def restore_from_file(
    uploaded_file: UploadEventArguments, label: ui.element
) -> None:
    """
    Restore the database from an uploaded backup ZIP file.

    Args:
        uploaded_file (UploadEventArguments): File uploaded by the user via the UI.
        label (ui.element): Label for displaying success/failure feedback.

    Process:
        - Saves the uploaded file to disk.
        - Extracts its contents to the configured database directory.
        - Shows notifications on success or failure.
    """
    loader = create_loader()
    label.text = ""
    try:
        # Check if file was uploaded
        if not uploaded_file.file:
            ui.notify("No file uploaded", type="warning")
            return

        _, source_path = FileHandler().get_new_file_name(file_extension="zip")

        # Read the file content first (this is async)
        file_content = await uploaded_file.file.read()

        # Save uploaded file - CORRECTED VERSION
        def save_uploaded():
            with open(source_path, "wb") as file:
                file.write(file_content)

        await run.io_bound(save_uploaded)

        target_path = Setup().db_path
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        # Extract zip
        def extract_zip():
            with zipfile.ZipFile(source_path, "r") as zip_ref:
                zip_ref.extractall(target_path)

        await run.io_bound(extract_zip)

        ui.notify("Backup restored successfully", type="positive")
        label.text = "File uploaded successfully."
        label.style("color: green")
    except BaseException as _e:
        Status.FAILED("Error restoring backup", error=_e)
        ui.notify("Something went wrong. Contact support.", type="negative")
    finally:
        loader.delete()
