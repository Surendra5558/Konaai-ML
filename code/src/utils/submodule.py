# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to create the submodule class"""
import json
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from sqlalchemy import quoted_name
from src.automl.ml_params import MLParameters
from src.automl.questionnaire import TemplateQuestion
from src.automl.utils import config
from src.utils.conf import Setup
from src.utils.database_config import SQLDatabaseManager
from src.utils.global_config import GlobalSettings
from src.utils.status import Status


class Submodule(BaseModel):
    """
    Submodule class for managing submodule configurations and metadata.
    This class is used to represent and manage a submodule within a larger module-based system.
    It provides methods for loading, saving, and querying submodule configurations, as well as
    retrieving related metadata from databases and configuration files.
    Attributes:
    -----------
        module (Optional[str]): The module to which the submodule belongs.
        submodule (Optional[str]): The submodule to be used for training the model.
        instance_id (Optional[str]): The instance ID for the model training context.
        active_model (Optional[str]): The currently active model in the submodule.
        alert_status (Optional[str]): The alert status for the submodule.
        concern_questionnaire (Optional[List[TemplateQuestion]]): Questionnaire for concern cases.
        no_concern_questionnaire (Optional[List[TemplateQuestion]]): Questionnaire for no-concern cases.
        template_id (Optional[int]): ID of the questionnaire template.
        ml_params (Optional[MLParameters]): Machine learning parameters for the submodule.
    """

    model_config = ConfigDict(frozen=False)

    # mandatory fields
    module: Optional[str] = Field(
        None,
        title="Module",
        description="The module to which the submodule belongs",
    )
    submodule: Optional[str] = Field(
        None,
        title="Submodule",
        description="The submodule to be used for training the model",
    )
    instance_id: Optional[str] = Field(
        None,
        title="Instance ID",
        description="The instance ID to be used for training the model",
    )

    # optional fields
    active_model: Optional[str] = Field(
        None, title="Active Model", description="The active model in the submodule"
    )
    alert_status: Optional[str] = Field(
        None,
        title="Alert Status",
        description="The alert status to be used for training the model",
    )
    concern_questionnaire: Optional[List[TemplateQuestion]] = Field(
        [],
        title="Concern Questionnaire",
        description="The questionnaire to be used for training the model",
    )
    no_concern_questionnaire: Optional[List[TemplateQuestion]] = Field(
        [],
        title="No concern Questionnaire",
        description="The questionnaire to be used for traing the model",
    )
    template_id: Optional[int] = Field(
        None,
        title="Questionnaire Template ID",
        description="The questionnaire template id to be used for training the model",
    )
    ml_params: Optional[MLParameters] = Field(
        MLParameters(),
        title="ML Parameters",
        description="The ml parameters to be used for training the model",
        alias="ML Parameters",
    )

    def __init__(
        self, module: str = None, submodule: str = None, instance_id: str = None, **data
    ) -> None:
        """This function is used to initialize the class"""

        if data:
            super().__init__(**data)
            return

        if any([module, submodule, instance_id]) is None:
            return

        if data_dict := self.load_configuration(module, submodule, instance_id):
            data = data_dict

        if not data:
            data = {
                "module": module,
                "submodule": submodule,
                "instance_id": instance_id,
            }

        # initialize the base model
        super().__init__(
            **data,
        )

        self.ml_params = MLParameters(**data.get("ml_params", {}))
        self.concern_questionnaire = [
            TemplateQuestion(**q) for q in data.get("concern_questionnaire", [])
        ]
        self.no_concern_questionnaire = [
            TemplateQuestion(**q) for q in data.get("no_concern_questionnaire", [])
        ]

    def get_module_id(self) -> Optional[str]:
        """This function is used to get the module id"""
        lookup_table_name = (
            Setup().global_constants.get("LOOKUP", {}).get("LOOKUP_TABLE")
        )
        name_column = Setup().global_constants.get("LOOKUP", {}).get("NAME_COLUMN")
        id_column = Setup().global_constants.get("LOOKUP", {}).get("RowID_COLUMN")

        # download lookup table
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            return None

        dff = instance.settings.projectdb.download_table_or_query(
            table_name=lookup_table_name
        )
        if dff is None or len(dff) == 0:
            Status.FAILED(
                "Lookup table not found",
                module=self.module,
                submodule=self.submodule,
                instance_id=self.instance_id,
            )
            return None

        # read lookup table
        data = dff.compute()

        # get module id
        return data.loc[(data[name_column] == self.module)][id_column].values[0] or None

    def get_submodule_id(self) -> Optional[str]:
        """This function is used to get the submodule id"""
        lookup_table_name = (
            Setup().global_constants.get("LOOKUP", {}).get("LOOKUP_TABLE")
        )
        name_column = Setup().global_constants.get("LOOKUP", {}).get("NAME_COLUMN")
        id_column = Setup().global_constants.get("LOOKUP", {}).get("RowID_COLUMN")
        parent_id_column = (
            Setup().global_constants.get("LOOKUP", {}).get("PARENT_ID_COLUMN")
        )
        lookup_id_column = (
            Setup().global_constants.get("LOOKUP", {}).get("LOOKUP_ID_COLUMN")
        )

        # download lookup table
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            return None
        dff = instance.settings.projectdb.download_table_or_query(
            table_name=lookup_table_name
        )
        if dff is None or len(dff) == 0:
            Status.FAILED(
                "Lookup table not found",
                module=self.module,
                submodule=self.submodule,
                instance_id=self.instance_id,
            )
            return None

        # read lookup table
        data = dff.compute()

        # get parent id for module
        lookup_id = (
            data.loc[(data[id_column] == self.get_module_id())][
                lookup_id_column
            ].values[0]
            or None
        )

        # get submodule id
        return (
            data.loc[
                (data[name_column] == self.submodule)
                & (data[parent_id_column] == lookup_id)
            ][id_column].values[0]
            or None
        )

    def get_data_table_name(self) -> Optional[str]:
        """This method is used to get submodule table name from the database"""
        udm_table_query = (
            Setup()
            .global_constants.get("METADATA", {})
            .get("UDM_TABLE_QUERY")
            .format(module=self.module, submodule=self.submodule)
        )

        # download data
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            return None

        dff = instance.settings.projectdb.download_table_or_query(query=udm_table_query)
        if dff is None or len(dff) == 0:
            Status.FAILED(
                "Submodule UDM table name not found",
                module=self.module,
                submodule=self.submodule,
            )
            return None

        # read module data
        udm_table = dff.compute()

        if udm_table is None or len(udm_table.index) == 0:
            Status.FAILED(
                "Submodule UDM table name not found",
                module=self.module,
                submodule=self.submodule,
            )
            return None

        udm_table_column = (
            Setup().global_constants.get("METADATA", {}).get("UDM_TABLE_COLUMN")
        )
        full_table_name = udm_table[udm_table_column].values[0]
        schema_name, table_name = (
            full_table_name.split(".", 1) if full_table_name else (None, None)
        )
        return (
            ".".join([quoted_name(schema_name, True), quoted_name(table_name, True)])
            if schema_name and table_name
            else None
        )

    def get_amount_column(self) -> Optional[str]:
        """
        Retrieves the amount column value from a UDM table for the specified module and submodule.

        This method performs the following steps:
        1. Initializes a database handler and sets the connection string.
        2. Constructs a query to fetch the UDM table based on the module and submodule.
        3. Downloads the data using the constructed query.
        4. Reads the downloaded data into a DataFrame.
        5. Checks if the DataFrame is empty or None, and logs a failure status if true.
        6. Retrieves the column name for the amount from the global configuration.
        7. Returns the value of the amount column from the first row of the DataFrame.

        Returns:
            str: The value of the amount column from the UDM table.

        Raises:
            Status.FAILED: If the UDM table is empty or None.
        """
        udm_table_query = (
            Setup()
            .global_constants.get("METADATA", {})
            .get("UDM_TABLE_QUERY")
            .format(module=self.module, submodule=self.submodule)
        )

        # download data
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            return None
        dff = instance.settings.projectdb.download_table_or_query(query=udm_table_query)
        if dff is None or len(dff) == 0:
            Status.FAILED("Submodule UDM table name not found", self)
            return None

        # read module data
        udm_table = dff.compute()

        if udm_table is None or len(udm_table.index) == 0:
            Status.FAILED("Submodule UDM table name not found", self)
            return None

        udm_table_column = (
            Setup()
            .global_constants.get("METADATA", {})
            .get("RISK_TRANSACTION_AMOUNT_COLUMN")
        )
        return udm_table[udm_table_column].values[0]

    def load_configuration(
        self, module: str, submodule: str, instance_id: str
    ) -> Optional[Dict[str, Any]]:
        """This function is used to load the configuration from the file"""
        if not all([module, submodule, instance_id]):
            return None

        # check if configuration is present
        file_path = os.path.join(
            Setup().db_path, instance_id, config.get("QUESTIONNAIRE", "FILE_NAME")
        )

        if not os.path.exists(file_path):
            return None

        data_list: List = None
        with open(file_path, encoding="utf-8") as file:
            try:
                data_list = json.load(file)
            except BaseException as _e:
                Status.FAILED(
                    "Error loading submodule automl configuration",
                    error=_e,
                    module=module,
                    submodule=submodule,
                    instance_id=instance_id,
                )
                return None

        object_dict: Dict = None
        if data_list and len(data_list) > 0:
            # filter for module and submodule
            single_object: Dict = [
                x
                for x in data_list
                if x["module"] == module and x["submodule"] == submodule
            ]
            if len(single_object) > 0:
                object_dict = single_object[0]

        return None if object_dict is None else object_dict

    def set_active_model(self, model_name: str) -> bool:
        """This function is used to set the active model"""
        self.active_model = model_name
        return self.save_configuration()

    def get_test_patterns(self) -> Optional[pd.DataFrame]:
        """this function is to get test patterns

        Returns:
            pd.DataFrame: _description_
        """
        try:
            pattern_table = (
                Setup().global_constants.get("TEST_PATTERNS", {}).get("PATTERN_TABLE")
            )
            module = (
                Setup()
                .global_constants.get("TEST_PATTERNS", {})
                .get("MODULE_COLUME_NAME")
            )
            submodule = (
                Setup()
                .global_constants.get("TEST_PATTERNS", {})
                .get("SUBMODULE_COLUME_NAME")
            )

            # Downloading Table
            instance = GlobalSettings.instance_by_id(self.instance_id)
            if not instance:
                Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
                return None
            dff = instance.settings.projectdb.download_table_or_query(
                table_name=pattern_table
            )
            if dff is None or len(dff) == 0:
                Status.FAILED(
                    "Test patterns data not found",
                    module=self.module,
                    submodule=self.submodule,
                    instance_id=self.instance_id,
                )
                return None

            pattern_data: pd.DataFrame = dff.compute()

            # filter for enabled patterns, filter for True
            enabled_column = (
                Setup().global_constants.get("TEST_PATTERNS", {}).get("ENABLED_COLUMN")
            )
            pattern_data = pattern_data[pattern_data[enabled_column] == 1]

            # sort by weightage column descending
            weightage_column = (
                Setup()
                .global_constants.get("TEST_PATTERNS", {})
                .get("WEIGHTAGE_COLUMN")
            )
            pattern_data = pattern_data.sort_values(
                by=[weightage_column], ascending=False
            )

            # filter for module and submodule
            return pattern_data.loc[
                (pattern_data[module] == self.module)
                & (pattern_data[submodule] == self.submodule)
            ]
        except BaseException as _e:
            Status.FAILED(
                "Error in downloading patterns data",
                error=_e,
                module=self.module,
                submodule=self.submodule,
                instance_id=self.instance_id,
            )
            return None

    def __get_file_path(self) -> str:
        """This function is used to get the file path"""
        # config file attributes
        _file_name = config.get("QUESTIONNAIRE", "FILE_NAME")

        return os.path.join(Setup().db_path, self.instance_id, _file_name)

    def __configuration_present(self) -> Tuple[bool, Dict]:
        """This function is used to check if the configuration is present for given submodule

        Returns:
            Tuple[bool, Dict]: Returns whether the configuration is present and the configuration data
        """
        file_path = self.__get_file_path()
        if not os.path.exists(file_path):
            # save empty configuration
            with open(file_path, "w", encoding="utf-8") as file:
                try:
                    json.dump([], file, indent=4)
                except BaseException as _e:
                    Status.FAILED(
                        "Error saving submodule automl configuration",
                        error=_e,
                        module=self.module,
                        submodule=self.submodule,
                        instance_id=self.instance_id,
                    )

            return False, self

        data = []
        with open(file_path, encoding="utf-8") as file:
            try:
                data = json.load(file)
            except BaseException as _e:
                Status.FAILED(
                    "Error loading submodule automl configuration",
                    error=_e,
                    module=self.module,
                    submodule=self.submodule,
                    instance_id=self.instance_id,
                )

        # check if submodule data is present
        submodule_data = [
            x
            for x in data
            if x["module"] == self.module and x["submodule"] == self.submodule
        ]

        return len(submodule_data) > 0, submodule_data[0] if submodule_data else self

    def save_configuration(self):
        """
        Saves the current configuration to a file. If a configuration for the current module and submodule
        already exists, it updates the existing entry; otherwise, it appends a new configuration entry.
        Handles errors during file reading and writing, and logs failures using the Status.FAILED method.

        Returns:
            bool: True if the configuration was saved successfully, False otherwise.
        """
        try:
            # get file path
            file_path = self.__get_file_path()

            # check if configuration is present
            present, _ = self.__configuration_present()

            # if present, update the configuration
            data = []
            with open(file_path, encoding="utf-8") as file:
                try:
                    data = json.load(file)
                except BaseException as _e:
                    Status.FAILED(
                        "Error loading submodule automl configuration",
                        error=_e,
                        module=self.module,
                        submodule=self.submodule,
                        instance_id=self.instance_id,
                    )

            if present:
                # update the configuration
                for idx, x in enumerate(data):
                    if x["module"] == self.module and x["submodule"] == self.submodule:
                        data[idx] = self.model_dump(by_alias=False)
                        break
            else:
                # add the configuration
                data.append(self.model_dump(by_alias=False))

            # save the configuration
            with open(file_path, "w", encoding="utf-8") as file:
                try:
                    json.dump(data, file, indent=4)
                except BaseException as _e:
                    Status.FAILED(
                        "Error saving submodule automl configuration",
                        error=_e,
                        module=self.module,
                        submodule=self.submodule,
                        instance_id=self.instance_id,
                    )
                    return False

            return True
        except BaseException as _e:
            Status.FAILED(
                "Error saving submodule automl configuration",
                error=_e,
                module=self.module,
                submodule=self.submodule,
                instance_id=self.instance_id,
            )
            return False

    def __str__(self):
        return f"Module: {self.module} - Submodule: {self.submodule} - Instance Id: {self.instance_id}"

    def __repr__(self):
        return self.__str__()

    @property
    def is_valid(self) -> bool:
        """This function is used to check if the submodule is valid"""
        # none of module, submodule and instance should be none
        return all([self.module, self.submodule, self.instance_id])

    def get_database(self) -> Optional[SQLDatabaseManager]:
        """
        Return the SQLDatabaseManager associated with this instance, or None if unavailable.
        This method looks up an instance using GlobalSettings.instance_by_id(self.instance_id).
        - If no instance is found, it records a NOT_FOUND status via Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            and returns None.
        - If the instance exists but has no settings, it records a NOT_FOUND status via
            Status.NOT_FOUND("Instance settings not found", instance_id=self.instance_id) and returns None.
        - Otherwise, it returns instance.settings.projectdb.
        Returns:
                Optional[SQLDatabaseManager]: The project's SQLDatabaseManager if available, otherwise None.
        Side effects:
                - Emits status notifications through Status.NOT_FOUND when the instance or its settings are missing.
        """

        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            return None

        if not instance.settings:
            Status.NOT_FOUND(
                "Instance settings not found", instance_id=self.instance_id
            )
            return None

        return instance.settings.projectdb
