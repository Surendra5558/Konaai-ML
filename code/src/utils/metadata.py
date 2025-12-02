# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to get metadata from the database"""
from typing import List
from typing import Optional

import pandas as pd
from src.utils.conf import Setup
from src.utils.global_config import GlobalSettings
from src.utils.status import Status


class Metadata:
    """
    Metadata is a singleton class used to retrieve and manage metadata from the database for a specific instance.
    Attributes:
    ----------
        instance_id (str): The unique identifier for the instance.
        instance (Instance): The instance object retrieved using the instance_id.
        client_id (str): The client unique identifier associated with the instance.
        project_id (str): The project unique identifier associated with the instance.
    """

    _instances = {}
    __modules: pd.DataFrame = None
    __submodules: pd.DataFrame = None

    def __new__(cls, instance_id: str):
        if instance_id in cls._instances:
            return cls._instances[instance_id]
        obj = super().__new__(cls)
        cls._instances[instance_id] = obj
        return obj

    def __init__(self, instance_id: str) -> None:
        # __init__ may be called multiple times due to __new__ returning a
        # cached object; guard against re-initialization.
        if self._instances:
            return

        self.instance_id: str = instance_id
        self.instance = GlobalSettings.instance_by_id(instance_id)
        self.client_id: str = self.instance.ClientUID
        self.project_id: str = self.instance.ProjectUID
        self.__modules: pd.DataFrame = None
        self.__submodules: pd.DataFrame = None

    def __get_module_data(self) -> Optional[pd.DataFrame]:
        """This method is used to get module data from the database"""
        try:
            return self.__download_query("MODULES_QUERY")
        except BaseException as _e:
            Status.FAILED(
                "Error in getting module data",
                client_id=self.client_id,
                project_id=self.project_id,
                error=_e,
                traceback=False,
            )
            return pd.DataFrame()

    @property
    def modules(self) -> List[str]:
        """
        Returns a list of unique module names from the module data.
        If the module data is not already loaded, it attempts to load it. If no module data is found,
        logs a failure status and returns an empty list. The method retrieves the module name column
        from global constants and returns the unique values from that column as a list.
        Returns:
            List[str]: A list of unique module names, or an empty list if no module data is found.
        """

        if self.__modules is None:
            self.__modules = self.__get_module_data()

        if self.__modules is None or len(self.__modules.index) == 0:
            Status.FAILED(
                "No module data found",
                client_id=self.client_id,
                project_id=self.project_id,
                traceback=False,
            )
            return []

        module_column = (
            Setup().global_constants.get("METADATA", {}).get("MODULE_NAME_COLUMN")
        )

        if module_column in self.__modules.columns:
            return self.__modules[module_column].unique().tolist()
        return []

    def __get_submodules_data(self) -> pd.DataFrame:
        """This method is used to get submodules data from the database"""
        try:
            return self.__download_query("SUBMODULES_QUERY")
        except BaseException as _e:
            Status.FAILED(
                "Error in getting submodule data",
                client_id=self.client_id,
                project_id=self.project_id,
                error=_e,
            )
            return pd.DataFrame()

    def __download_query(self, option: str) -> Optional[pd.DataFrame]:
        """This method is used to download data from the database"""
        modules_query = Setup().global_constants.get("METADATA", {}).get(option)
        modules_query = modules_query.format(
            project_id=self.project_id, client_id=self.client_id
        )

        dff = self.instance.settings.masterdb.download_table_or_query(
            query=modules_query
        )
        if dff is None or len(dff.index) == 0:
            Status.FAILED(
                "Error in downloading data",
                client_id=self.client_id,
                project_id=self.project_id,
            )
            return None

        return dff.compute()

    def get_submodule_names(self, module_name: str) -> List[str]:
        """
        Retrieve the list of submodule names associated with a given module name from the database.
        Args:
        -----
            module_name (str): The name of the module for which submodule names are to be fetched.

        Returns:
            List[str]: A list of submodule names corresponding to the specified module. Returns an empty list if no submodules are found.

        Logs:
            - Logs an informational message when fetching submodule names.
            - Logs a failure message if no submodule data is found.

        Raises:
            IndexError: If the specified module name does not exist in the modules data.
        """
        Status.INFO("Fetching submodule names", module=module_name)
        # get meta data submodule table name
        submodule_column = (
            Setup().global_constants.get("METADATA", {}).get("SUBMODULE_NAME_COLUMN")
        )
        moduleid_column = (
            Setup().global_constants.get("METADATA", {}).get("MODULE_ID_COLUMN")
        )
        module_column = (
            Setup().global_constants.get("METADATA", {}).get("MODULE_NAME_COLUMN")
        )

        if self.__modules is None or len(self.__modules.index) == 0:
            self.__modules = self.__get_module_data()

        if self.__submodules is None or len(self.__submodules.index) == 0:
            self.__submodules = self.__get_submodules_data()

        if self.__submodules is None or len(self.__submodules.index) == 0:
            Status.FAILED(
                "No submodule data found",
                module=module_name,
                client_id=self.client_id,
                project_id=self.project_id,
            )
            return []

        module_id = (
            self.__modules[self.__modules[module_column] == module_name][
                moduleid_column
            ]
            .unique()
            .tolist()[0]
        )

        return (
            self.__submodules[self.__submodules[moduleid_column] == module_id][
                submodule_column
            ]
            .unique()
            .tolist()
        )
