# # Copyright (C) KonaAI - All Rights Reserved
"""A UI component for file upload and download."""
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional

from nicegui import events
from nicegui import run
from nicegui import ui
from src.admin.components.spinners import create_overlay_spinner
from src.utils.file_mgmt import FileHandler
from src.utils.status import Status


class FileDownloadButton(ui.element):
    """A UI component for downloading files."""

    spinner: ui.element = None
    file_path: Path = None

    def __init__(
        self,
        file_provider_func: Callable,
        file_name: str,
        button_text: Optional[str] = "Download File",
    ):
        """
        Initialize a file download component.

        Args:
            file_provider_func (Callable): A callable that provides/generates the file content to be downloaded. It should either provide a bytes object or a valid file path as a string or Path object.
            file_name (str): The name of the file to be downloaded including extension.
            button_text (Optional[str], optional): The text to display on the download button.
                Defaults to "Download File".

        Returns:
            None
        """
        super().__init__()
        self.file_provider_func = file_provider_func
        self.button_text = button_text
        self.file_name = file_name
        self._create_button()

    def _safe_delete_spinner(self):
        """Safely delete the spinner element."""
        if self.spinner:
            try:
                self.spinner.delete()
            except Exception:
                # Handle cases where spinner is already deleted or invalid
                pass  # nosec
            finally:
                self.spinner = None

    def _create_button(self):
        """Creates the download button in the UI."""
        btn = ui.button(self.button_text).classes("mt-2")
        btn.on_click(self._download_file)

    async def _handle_file_output(self):
        """Retrieves the file path from the provider function."""
        try:
            # validate that the file_provider_func is not async
            if callable(getattr(self.file_provider_func, "__await__", None)):
                raise ValueError("file_provider_func must be a synchronous function")

            output = await run.io_bound(self.file_provider_func)
            if not output:
                raise ValueError("file_provider_func returned None or empty value")

            if not isinstance(output, (bytes, str, Path)):
                raise TypeError(
                    "file_provider_func must return a bytes, str, or Path object"
                )

            fh = FileHandler()
            file_extension = Path(self.file_name).suffix
            if not file_extension:
                raise ValueError("file_name must have a valid file extension")

            _, temp_file_path = fh.get_new_file_name(file_extension=file_extension)

            # handle bytes output by writing to a temp file
            if isinstance(output, bytes):
                with open(temp_file_path, "wb") as f:
                    f.write(output)
                self.file_path = temp_file_path
                return

            # handle str or Path output
            if isinstance(output, str) and Path(output).is_file():
                self.file_path = output
                return

            if isinstance(output, Path) and output.is_file():
                self.file_path = output
            else:
                raise ValueError("file_provider_func returned an invalid file path")
        except Exception as ex:
            Status.FAILED("Failed to prepare file for download", error=ex)
            self.file_path = None

    async def _download_file(self, _: events.ClickEventArguments):
        """Handles the file download when the button is clicked."""
        self.spinner = create_overlay_spinner("Preparing your download...")
        try:
            await self._handle_file_output()
            if not self.file_path or not Path(self.file_path).is_file():
                raise FileNotFoundError("The file to download was not found.")

            ui.download(
                src=self.file_path,
                filename=self.file_name,
                media_type="application/octet-stream",
            )
        except Exception as ex:
            Status.FAILED("Failed to prepare download.", error=ex)
            ui.notify("Error preparing download. Contact support.", type="negative")
        finally:
            self._safe_delete_spinner()


class SingleFileUploadButton(ui.element):
    """A UI component for uploading files."""

    spinner: ui.element = None
    upload_component: ui.upload = None

    def __init__(
        self,
        upload_handler: Callable,
        file_extensions: List[str] = None,
        button_text: Optional[str] = "Upload File",
    ):
        """
        Initialize the file upload component.
        Args:
            upload_handler (Callable): A callback function that handles the uploaded file(s).
                This function will be called when files are uploaded. It support maximum of 500 MB file size. It should accept a single argument which is the path to the uploaded file.
            file_extensions (List[str], optional): A list of allowed file extensions (e.g., ['.txt', '.pdf']).
                If None or empty list, all file types are allowed. Defaults to None.
            button_text (Optional[str], optional): The text to display on the upload button.
                Defaults to "Upload File".
        Returns:
            None
        """

        super().__init__()
        self.upload_handler = upload_handler
        self.file_extensions = file_extensions or []
        self.button_text = button_text
        self._create_button()

    def _safe_delete_spinner(self):
        """Safely delete the spinner element."""
        if self.spinner:
            try:
                self.spinner.delete()
            except Exception:
                # Handle cases where spinner is already deleted or invalid
                pass  # nosec
            finally:
                self.spinner = None

    def _on_begin_upload(self, _):
        """Handle the beginning of file upload by creating spinner."""
        self.spinner = create_overlay_spinner("Uploading file...")

    def _create_button(self):
        """Handles the file upload when the button is clicked."""
        try:
            if self.file_extensions:
                file_extensions = ",".join(
                    [
                        ext if ext.startswith(".") else f".{ext}"
                        for ext in self.file_extensions
                    ]
                )
            else:
                file_extensions = "*/*"

            self.upload_component = (
                ui.upload(
                    max_file_size=500 * 1024 * 1024,  # 500 MB limit
                    multiple=False,
                    label=self.button_text,
                    auto_upload=True,
                    on_begin_upload=self._on_begin_upload,
                    on_upload=self._handle_upload,
                )
                .classes("grow mt-2 w-full")
                .props(f"accept={file_extensions}")
            )
        except Exception as ex:
            Status.FAILED(f"Failed to upload file: {ex}")
            ui.notify("Error uploading file. Contact support.", type="negative")

    async def _handle_upload(self, e: events.UploadEventArguments):
        """Placeholder for the upload handler."""
        try:
            filename = e.file.name
            if (
                Path(filename).suffix not in self.file_extensions
                and self.file_extensions
            ):
                ui.notify(f"Invalid file type: {filename}", type="negative")
                return

            content = await e.file.read()
            if not content:
                ui.notify(f"Empty file uploaded: {filename}", type="negative")
                return

            # save the file to a temp location
            fh = FileHandler()
            _, temp_file_path = fh.get_new_file_name(
                file_extension=Path(filename).suffix, file_name=filename
            )
            with open(temp_file_path, "wb") as f:
                f.write(content)

            # Call the provided upload handler with the temp file path
            self.upload_handler(temp_file_path)

        except Exception as ex:
            Status.FAILED(f"Failed to handle uploaded file: {ex}")
            ui.notify(
                "Error processing uploaded file. Contact support.",
                type="negative",
            )
        finally:
            # Always stop the spinner when upload is complete (success or failure)
            self._safe_delete_spinner()
            # Clear upload component to show only one file
            if self.upload_component:
                try:
                    # Use JavaScript to reset the file input
                    self.upload_component.run_method("reset")
                except Exception:  # nosec B110
                    try:
                        self.upload_component.clear()
                    except Exception:  # nosec B110
                        pass
