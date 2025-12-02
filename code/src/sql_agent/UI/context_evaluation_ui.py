# # Copyright (C) KonaAI - All Rights Reserved
"""Class to handle the UI for context evaluation (module descriptions)."""
import contextlib
from pathlib import Path
from typing import Optional

from nicegui import ui
from src.admin.components.file_works import FileDownloadButton
from src.admin.components.file_works import SingleFileUploadButton
from src.sql_agent.context_evaluation import ContextEvaluation
from src.utils.instance import Instance
from src.utils.status import Status


class ContextEvaluationUI:
    """Class to handle the UI for context evaluation (module descriptions)."""

    instance: Instance = None

    def __init__(self, instance: Instance):
        self.instance = instance

    def render(self, element: Optional[ui.element] = None):
        """
        Renders the UI for the context evaluation (module descriptions) management.
        """

        # If element is provided, clear it as well to prevent stacking
        if element:
            with contextlib.suppress(Exception):
                element.clear()
        else:
            element = ui.element()

        with element:
            with ui.column().classes("w-full gap-1"):
                ui.markdown(
                    "**Upload module descriptions to enable automatic module detection.**"
                ).classes("text-sm text-gray-600")

                # Download button for context evaluation file
                FileDownloadButton(
                    file_provider_func=self._prepare_file_download,
                    file_name="module_descriptions.txt",
                )

                # Upload button for context evaluation file
                SingleFileUploadButton(
                    button_text="Upload Module Descriptions",
                    file_extensions=[".txt"],
                    upload_handler=self._handle_file_upload,
                )

                # Display uploaded filename if exists
                self._display_filename()

    @ui.refreshable_method
    def _display_filename(self):
        """Display the uploaded filename if it exists."""
        if filename := ContextEvaluation(self.instance).get_uploaded_filename():
            with ui.row().classes(
                "w-full items-center gap-2 mt-2 p-2 bg-gray-50 rounded"
            ):
                ui.icon("check_circle", color="green", size="sm")
                ui.label(f"Uploaded: {filename}").classes("text-sm text-gray-700")

    def _prepare_file_download(self) -> Optional[bytes]:
        """Prepare the context evaluation file for download."""
        try:
            context_eval = ContextEvaluation(self.instance)
            if context_text := context_eval.load_context_from_file():
                return context_text.encode("utf-8")

            raise ValueError("No module descriptions found to download.")
        except Exception as ex:
            Status.FAILED("Error preparing file download.", error=ex)
            return None

    def _handle_file_upload(self, file_path: str) -> bool:
        """Handle uploading the context evaluation file."""
        context_eval = ContextEvaluation(self.instance)
        content = None
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if not content:
            Status.FAILED("Empty file uploaded for module descriptions.")
            ui.notify("Empty file uploaded for module descriptions.", type="negative")
            return False

        # remove extra whitespace
        content = "\n".join(
            [line.strip() for line in content.splitlines() if line.strip()]
        )

        # Save the context file
        if not context_eval.save_context_to_file(content):
            Status.FAILED("Failed to save module descriptions.")
            ui.notify("Failed to save module descriptions.", type="negative")
            return False

        # Save the uploaded filename (temp file has original filename)
        context_eval.save_uploaded_filename(Path(file_path).name)
        self._display_filename.refresh()

        Status.SUCCESS("Module descriptions uploaded successfully")
        ui.notify("Module descriptions uploaded successfully", type="positive")
        return True
