# # Copyright (C) KonaAI - All Rights Reserved
"""Class to handle the UI for data dictionaries."""
import contextlib
from typing import Optional

import pandas as pd
from nicegui import ui
from src.admin.components.file_works import FileDownloadButton
from src.admin.components.file_works import SingleFileUploadButton
from src.admin.components.submodule_selector import SubModuleSelectorUI
from src.sql_agent.data_dictionary import SQLDataDictionary
from src.utils.file_mgmt import file_handler
from src.utils.instance import Instance
from src.utils.status import Status
from src.utils.submodule import Submodule


class DataDictionaryUI:
    """Class to handle the UI for data dictionaries."""

    instance: Instance = None

    def __init__(self, instance: Instance):
        self.instance = instance
        self.sub: Submodule = None

    async def render(self, element: Optional[ui.element] = None):
        """Renders the UI for the data dictionary management."""
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
                # show the submodule selector

                await subui.render(
                    self._draw_data_dictionary_widget, submodule=subui.submodule
                )

    def _draw_data_dictionary_widget(self, submodule: Submodule):
        if not submodule.module or not submodule.submodule:
            return

        # Download button
        FileDownloadButton(
            file_provider_func=lambda: self._download_data_dictionary(submodule),
            file_name=f"{submodule.module}_{submodule.submodule}_data_dictionary.xlsx",
        )

        # Upload button
        self.sub = submodule
        SingleFileUploadButton(
            file_extensions=[".xlsx", ".xls"],
            upload_handler=self._upload_data_dictionary,
        )

    def _upload_data_dictionary(self, file_path: str):
        try:
            if not self.instance.settings.projectdb:
                raise ValueError("Instance project database is not configured.")

            # read the uploaded excel file
            df = pd.read_excel(file_path)
            if df is None or df.empty:
                raise ValueError("Uploaded file is empty or invalid.")

            # access the udm table name from the submodule
            udm_table_name = self.sub.get_data_table_name()
            schema_name, table_name = (
                udm_table_name.split(".", 1) if udm_table_name else (None, None)
            )
            if not schema_name or not table_name:
                raise ValueError("UDM table name could not be determined.")

            # access the data dictionary
            dictionary = SQLDataDictionary(
                table_schema=schema_name,
                table_name=table_name,
                db=self.instance.settings.projectdb,
            )
            dd_table_name = dictionary.data_dictionary_table_name
            if not dd_table_name:
                raise ValueError("Data dictionary table name could not be determined.")

            # convert the dataframe to parquet format
            _, parquet_file_path = file_handler.get_new_file_name(
                file_extension=".parquet"
            )
            cols = [col.value for col in dictionary.DDMetadata]

            # check if the required columns are present
            if any(col not in df.columns for col in cols):
                raise ValueError("Uploaded file is missing required columns.")

            df[cols].to_parquet(parquet_file_path, index=False)
            self.instance.settings.projectdb.upload_table(
                data_file_path=parquet_file_path, table_name=dd_table_name
            )

            Status.SUCCESS("Data dictionary uploaded successfully.", self.sub)
            ui.notify("Data dictionary uploaded successfully.", type="positive")
        except Exception as err:
            Status.FAILED(
                "Failed to upload data dictionary",
                self.sub,
                traceback=True,
                error=str(err),
            )
            ui.notify(
                "Failed to upload data dictionary. Contact support.", type="negative"
            )

    def _download_data_dictionary(self, submodule: Submodule) -> Optional[str]:
        try:
            if not self.instance.settings.projectdb:
                Status.WARNING("Instance project database is not configured.")
                return None

            udm_table_name = submodule.get_data_table_name()
            schema_name, table_name = (
                udm_table_name.split(".", 1) if udm_table_name else (None, None)
            )
            if not schema_name or not table_name:
                return None

            # access the data dictionary
            dictionary = SQLDataDictionary(
                table_schema=schema_name,
                table_name=table_name,
                db=self.instance.settings.projectdb,
            )
            dd_table_name = dictionary.data_dictionary_table_name
            if not dd_table_name:
                return None

            # create as an excel file
            _, file_path = file_handler.get_new_file_name(
                file_extension=".xlsx", file_name=f"{dd_table_name}.xlsx"
            )
            df: pd.DataFrame = dictionary.get_schema()
            if df is None or df.empty:
                df = dictionary._load_columns()

            if df is None or df.empty:
                return None

            if dictionary.DDMetadata.EXCLUDE.value not in df.columns:
                df[dictionary.DDMetadata.EXCLUDE.value] = True
            if dictionary.DDMetadata.DESCRIPTION.value not in df.columns:
                df[dictionary.DDMetadata.DESCRIPTION.value] = ""

            # save the dataframe to an excel file
            df.to_excel(file_path, index=False)

            return file_path
        except Exception as e:
            Status.FAILED("Failed to download data dictionary", submodule, error=str(e))
            return None

    def _check_data_dictionary_exists(self, submodule: Submodule) -> bool:
        """
        Check if data dictionary exists for the submodule (lightweight check).
        This method only checks if the data dictionary table exists without
        downloading or merging data, to avoid timeouts during UI rendering.
        """
        try:
            if not self.instance or not submodule:
                return False
            udm_table_name = submodule.get_data_table_name()
            schema_name, table_name = (
                udm_table_name.split(".", 1) if udm_table_name else (None, None)
            )
            if not schema_name or not table_name:
                return False
            dictionary = SQLDataDictionary(
                table_schema=schema_name,
                table_name=table_name,
                db=self.instance.settings.projectdb,
            )
            # Lightweight check: only verify table exists, don't download/merge data
            # This avoids timeout issues during UI rendering
            dd_table_name = dictionary.data_dictionary_table_name
            if not dd_table_name:
                return False
            # Just check if table exists - this is fast and won't timeout
            return dictionary.db.does_table_exist(table_name=dd_table_name)
        except Exception:
            return False
