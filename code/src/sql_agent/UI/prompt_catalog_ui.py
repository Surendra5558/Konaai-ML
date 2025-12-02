# # Copyright (C) KonaAI - All Rights Reserved
"""Class to handle the UI for prompt catalogues."""
import contextlib
from pathlib import Path
from typing import Optional

from nicegui import ui
from src.admin.components.file_works import FileDownloadButton
from src.admin.components.file_works import SingleFileUploadButton
from src.admin.components.submodule_selector import SubModuleSelectorUI
from src.sql_agent.prompt_catalogue import PromptCatalogue
from src.utils.instance import Instance
from src.utils.status import Status
from src.utils.submodule import Submodule


class PromptCatalogueUI:
    """Class to handle the UI for prompt catalogues."""

    instance: Instance = None
    sub: Submodule = None

    def __init__(self, instance: Instance):
        self.instance = instance

    async def render(self, element: Optional[ui.element] = None):
        """
        Renders the UI for the prompt data management.
        """
        subui = SubModuleSelectorUI()

        # If element is provided, clear it as well to prevent stacking
        if element:
            with contextlib.suppress(Exception):
                element.clear()
                subui.reset()
        else:
            element = ui.element()

        with element:
            with ui.column().classes("w-full gap-1"):
                await subui.render(self._initiate_draw, submodule=subui.submodule)

    def _initiate_draw(self, submodule: Submodule):
        if not submodule.module or not submodule.submodule:
            return

        # Download button
        FileDownloadButton(
            file_provider_func=lambda: self._download_prompt_data(submodule),
            file_name=f"{submodule.module}_{submodule.submodule}_prompt_data.txt",
        )

        # Upload button
        self.sub = submodule
        SingleFileUploadButton(
            upload_handler=self._upload_prompt_data,
            file_extensions=[".txt"],
        )

        # Display uploaded filename if exists
        self._display_filename(submodule)

    @ui.refreshable_method
    def _display_filename(self, submodule: Submodule):
        """Display the uploaded filename if it exists."""
        try:
            catalog = PromptCatalogue(
                module=submodule.module,
                submodule=submodule.submodule,
                instance_id=self.instance.instance_id,
            )
            if filename := catalog.get_uploaded_filename():
                if catalog.exists:  # Check if prompt data exists
                    with ui.row().classes(
                        "w-full items-center gap-2 mt-2 p-2 bg-gray-50 rounded"
                    ):
                        ui.icon("check_circle", color="green", size="sm")
                        ui.label(f"Uploaded: {filename}").classes(
                            "text-sm text-gray-700"
                        )
        except Exception:  # nosec B110
            pass

    def _download_prompt_data(self, submodule: Submodule):
        try:
            # Fetch the prompt data from the database
            data: Optional[str] = PromptCatalogue(
                module=submodule.module,
                submodule=submodule.submodule,
                instance_id=self.instance.instance_id,
            ).get_prompt_data()

            if not data:
                raise ValueError("No prompt data found for the selected submodule.")

            # remove extra spaces
            data = "\n".join(
                [line.strip() for line in data.splitlines() if line.strip()]
            )

            return data.encode("utf-8")
        except Exception as e:
            Status.FAILED(
                "Error downloading prompt data",
                submodule,
                error=str(e),
                traceback=False,
            )
            return None

    def _upload_prompt_data(self, file_path: str):
        """Upload a prompt data."""
        try:
            if not self.instance.settings.projectdb:
                raise ValueError("Instance project database is not configured.")

            # read the uploaded text file
            content = ""
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # remove extra spaces
            content = "\n".join(
                [line.strip() for line in content.splitlines() if line.strip()]
            )

            # Save the prompt data
            catalog = PromptCatalogue(
                module=self.sub.module,
                submodule=self.sub.submodule,
                instance_id=self.instance.instance_id,
            )
            catalog.prompt_data = content.encode("utf-8")
            if not catalog.upsert():
                raise ValueError("Failed to upload prompt data to the database.")

            # Save the uploaded filename (temp file has original filename)
            catalog.save_uploaded_filename(Path(file_path).name)
            self._display_filename.refresh(self.sub)

            Status.SUCCESS("Prompt data uploaded successfully.", self.sub)
            ui.notify("Prompt data uploaded successfully.", type="positive")
        except Exception as ex:
            Status.FAILED("Error uploading prompt data.", error=ex)
            ui.notify("Failed to upload prompt data. Contact support.", type="negative")
