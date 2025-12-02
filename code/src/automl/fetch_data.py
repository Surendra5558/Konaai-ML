# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to get training data from the database"""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import dask.dataframe as dd
import pandas as pd
from pydantic import BaseModel
from pydantic import Field
from src.automl.questionnaire import TemplateQuestion
from src.automl.utils import config
from src.tools.dask_tools import compute
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule


class ModelData:
    """This class is used to get training data from database and assign target values based on index
    for a Dask DataFrame partition."""

    index = config.get("DATA", "INDEX")

    def __update_index(self, df: dd.DataFrame) -> None:
        """This function is used to update index column in dataframe"""
        Status.INFO("finding index column in data")

        # find index column irrespective of case
        if self.index.lower() in df.columns.str.lower():
            self.index = df.columns[df.columns.str.lower() == self.index.lower()][0]

    def __init__(self, submodule_obj: Submodule) -> None:
        self.submodule_obj = submodule_obj

    def _validate_y(
        self,
        concern_keys: dd.Series,
        no_concern_keys: dd.Series,
    ) -> Status:
        try:
            Status.INFO("Validating target data", self.submodule_obj)

            total_min_data = int(config.get("DATA", "MIN_DATA_FOR_TRAINING"))
            min_data_for_class = int(config.get("DATA", "MIN_DATA_FOR_CLASS"))

            # check if concern keys are empty
            total_concern_records = (
                len(concern_keys.index) if concern_keys is not None else 0
            )

            # Check if no concern keys are empty
            total_no_concern_records = (
                len(no_concern_keys.index) if no_concern_keys is not None else 0
            )

            total_records = total_concern_records + total_no_concern_records
            Status.INFO(
                f"Total Concern Records : {total_concern_records}, Total No Concern Records : {total_no_concern_records}. Total Records : {total_records}.",
                self.submodule_obj,
            )

            # --- original training checks (unchanged) ---
            if total_concern_records < min_data_for_class:
                raise ValueError(
                    f"Minimum concern data not found for training. Expected at least {min_data_for_class} records. Found {total_concern_records} records"
                )

            if total_no_concern_records < min_data_for_class:
                raise ValueError(
                    f"Minimum no concern data not found for training. Expected at least {min_data_for_class} records. Found {total_no_concern_records} records"
                )

            if total_records < total_min_data:
                raise ValueError(
                    f"Minimum total data not available for training. Requires {total_min_data} records. Found {total_records} records"
                )

        except BaseException as _e:
            return Status.FAILED(
                f"Validation Error: {str(_e)}",
                alert_status=self.submodule_obj.alert_status,
            )
        return None

    def _get_concern_no_concern_keys(self) -> Tuple[pd.Series, pd.Series]:
        """
        Retrieves concern and no concern keys from the questionnaire data.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple containing two pandas Series:
                - The first Series contains the concern keys.
                - The second Series contains the no concern keys.
        """
        # get all questionnaire data
        concern_keys, no_concern_keys = self.get_filtered_questionnaire_data()
        # identify all no concern keys that also exist in concern keys
        common_keys = concern_keys[concern_keys.isin(no_concern_keys)]
        if len(common_keys) > 0:
            Status.INFO(
                f"Found {len(common_keys)} common records in both concern and no concern data. Considering only concern records and removing them from no concern data.",
                self.submodule_obj,
            )
            no_concern_keys = no_concern_keys[~no_concern_keys.isin(common_keys)]
        return concern_keys, no_concern_keys

    class TrainingDataValidationResult(BaseModel):
        """This class is used to store the result of training data validation"""

        message: Optional[str] = Field(None, description="Validation status message")
        total_concern_records: Optional[int] = Field(
            None, description="Total number of concern records"
        )
        total_no_concern_records: Optional[int] = Field(
            None, description="Total number of no concern records"
        )
        min_data_per_class: Optional[int] = Field(
            None, description="Minimum data required per class"
        )
        total_training_data: Optional[int] = Field(
            None, description="Total training data available"
        )
        min_training_data: Optional[int] = Field(
            None, description="Minimum training data required"
        )
        has_min_training_data: Optional[bool] = Field(
            False, description="Indicates if minimum training data is available"
        )

    def min_data_validator(self) -> TrainingDataValidationResult:
        """
        Validate that there is sufficient training data for both classes and overall.
        This method:
        - Retrieves labeled records split into "concern" and "no concern" sets via
            self._get_concern_no_concern_keys().
        - Performs additional label/format validation via self._validate_y(concern_keys, no_concern_keys).
        - Reads minimum thresholds from the configuration:
            - DATA.MIN_DATA_FOR_TRAINING (total minimum records required)
            - DATA.MIN_DATA_FOR_CLASS (minimum records required per class)
        - Constructs and returns a ModelData.TrainingDataValidationResult populated with:
            - total_concern_records: int number of concern records
            - total_no_concern_records: int number of no-concern records
            - min_data_per_class: int threshold for each class (from config)
            - total_training_data: int sum of concern + no-concern records
            - min_training_data: int overall minimum required (from config)
            - has_min_training_data: bool indicating whether minimums are satisfied
            - message: human-readable status or validation error message
        Return:
                ModelData.TrainingDataValidationResult: result object with the fields above set.
                If self._validate_y returns a truthy error object, the method sets
                result.message = _error.message, result.has_min_training_data = False and
                returns immediately. Otherwise it sets result.message to
                "Sufficient training data available." and result.has_min_training_data = True.
        Notes:
        - The method assumes that concern_keys and no_concern_keys expose a .index whose
            length reflects the number of records (len(concern_keys.index)).
        - Reading configuration values may raise exceptions if keys are missing or not
            convertible to int; such exceptions are propagated to the caller.
        """
        # get all questionnaire data
        concern_keys, no_concern_keys = self._get_concern_no_concern_keys()
        _error = self._validate_y(concern_keys, no_concern_keys)

        total_min_data = int(config.get("DATA", "MIN_DATA_FOR_TRAINING"))
        min_data_for_class = int(config.get("DATA", "MIN_DATA_FOR_CLASS"))

        result = ModelData.TrainingDataValidationResult()
        result.total_concern_records = len(concern_keys.index)
        result.total_no_concern_records = len(no_concern_keys.index)
        result.min_data_per_class = min_data_for_class
        result.total_training_data = len(concern_keys.index) + len(
            no_concern_keys.index
        )
        result.min_training_data = total_min_data

        if _error:
            result.message = _error.message
            result.has_min_training_data = False
            return result

        result.message = "Sufficient training data available."
        result.has_min_training_data = True
        return result

    def get_training_data(self) -> Tuple[dd.DataFrame, dd.Series, Union[Status, None]]:
        """
        Fetches and prepares the training data for the model.

        Returns:
            Tuple[dd.DataFrame, dd.Series, Union[Status, None]]:
                - X (dd.DataFrame): The feature matrix for training.
                - y (dd.Series): The target variable series for training.
                - _error (Union[Status, None]): Status object indicating any errors encountered during data fetching and preparation.

        Steps:
        -----
            1. Retrieves keys from the questionnaire data to identify concern and no concern records.
            2. Identifies and removes common keys present in both concern and no concern records.
            3. Validates the target variable (y) for any inconsistencies.
            4. Creates the target variable (y) with values 1 for concern and 0 for no concern.
            5. Fetches the feature matrix (X) from submodule data.
            6. Drops duplicate records from the feature matrix (X).
            7. Updates and sets the index for the feature matrix (X).
            8. Ensures all target variable (y) indexes are present in the feature matrix (X).
            9. Logs the total number of unique training records.
        Returns:
            Tuple containing the feature matrix (X), target variable (y), and any error status encountered.
        """
        # get keys from questionnaire for y
        # concern_keys and no_concern_keys are the series of the indexes of the data that are marked as concern and no concern
        concern_keys, no_concern_keys = self._get_concern_no_concern_keys()

        # validate y
        if _error := self._validate_y(concern_keys, no_concern_keys):
            return pd.DataFrame(), pd.Series(), _error

        # create a y series with concern and no concern keys
        # with values 1 and 0 respectively for concern and no concern
        # with self.index as index
        y = pd.Series(
            index=concern_keys.values.tolist() + no_concern_keys.values.tolist()
        )
        y[concern_keys.values] = 1
        y[no_concern_keys.values] = 0
        # convert to dask series
        y = dd.from_pandas(y, npartitions=1)
        # rename series
        target = config.get("TARGET", "TARGET_COLUMN")
        y: dd.Series = compute(y.rename(target))

        # get actual X
        X: dd.DataFrame = self.get_submodule_data(archived_alerts=True)
        if X is None or len(X.index) == 0:
            return (
                pd.DataFrame(),
                pd.Series(),
                Status.FAILED("No submodule data found", self.submodule_obj),
            )
        # drop duplicates
        X = compute(X.drop_duplicates(subset=self.index))

        # update index
        self.__update_index(X)
        # set index
        X = compute(X.set_index(self.index))

        # check if all y indexes are present in X
        # Check if y indexes are present in the DataFrame index
        y_index = y.index.compute().values.tolist()
        present_indexes = []
        for i in range(X.npartitions):
            partition = X.index.get_partition(i).compute()
            present_indexes.extend(
                [index for index in y_index if index in partition.values.tolist()]
            )
        if missing_indexes := [
            index for index in y_index if index not in present_indexes
        ]:
            Status.WARNING(
                f"Missing {len(missing_indexes)} training record in submodule data. Removing missing records from training data",
                self.submodule_obj,
            )
            # remove missing indexes from y
            y = y.map_partitions(
                lambda s: s[~s.index.isin(missing_indexes)], meta=y._meta
            )
            # validate y
            if _error := self._validate_y(concern_keys, no_concern_keys):
                return pd.DataFrame(), pd.Series(), _error

        Status.INFO(
            f"Total unique training records : {len(y.index)}", self.submodule_obj
        )
        return X, y, _error

    def get_archived_data(self, table_name: str) -> Optional[dd.DataFrame]:
        """
        Retrieves archived data from a specified table.

        Args:
            table_name (str): The name of the table to fetch archived data from.

        Returns:
            dd.DataFrame: A Dask DataFrame containing the archived data if found,
                          otherwise None.

        Raises:
            Status.NOT_FOUND: If no data is found in the specified archived table.
        """
        if not table_name:
            return None

        # Downloading Archived
        instance = GlobalSettings.instance_by_id(self.submodule_obj.instance_id)
        if not instance:
            Status.NOT_FOUND(
                "Instance not found",
                self.submodule_obj,
                alert_status=self.submodule_obj.alert_status,
            )
            return None
        instance_db = instance.settings.projectdb

        dff = instance_db.download_table_or_query(table_name=table_name)
        if dff is None or len(dff) == 0:
            Status.NOT_FOUND(f"No data found in {table_name} table", self.submodule_obj)
            return None

        return dff.drop_duplicates()

    def get_submodule_data(self, archived_alerts: bool = False) -> dd.DataFrame:
        """
        Downloads submodule data from the UDM table and optionally merges it with archived alerts data.

        Args:
            archived_alerts (bool): If True, merges the submodule data with archived alerts data. Defaults to False.

        Returns:
            dd.DataFrame: A Dask DataFrame containing the submodule data, optionally merged with archived alerts data.
                          Returns None if data download fails.

        Raises:
            ValueError: If the UDM table name is not found or if no data is found in the UDM table.

        Logs:
            Status.INFO: Logs the start of the data download process.
            Status.SUCCESS: Logs the successful download and merge of data.
            Status.WARNING: Logs a warning if the data download fails.
        """

        Status.INFO("Downloading submodule data", self.submodule_obj)
        submodule_data = None
        archived_table_name = None  # Initialize variable
        try:
            # Getting UDM Table name
            udm_table_name = self.submodule_obj.get_data_table_name()
            if not udm_table_name:
                raise ValueError("UDM Table name not found")

            # Downloading UDM Table
            instance = GlobalSettings.instance_by_id(self.submodule_obj.instance_id)
            if not instance:
                Status.NOT_FOUND(
                    "Instance not found",
                    self.submodule_obj,
                    alert_status=self.submodule_obj.alert_status,
                )
                return None
            instance_db = instance.settings.projectdb
            dff = instance_db.download_table_or_query(table_name=udm_table_name)
            if dff is None or len(dff) == 0:
                raise ValueError(f"No data found in {udm_table_name} Table")
            submodule_data = dff.drop_duplicates()

            # Check if data is available
            if submodule_data is None or len(submodule_data) == 0:
                raise ValueError(f"No data found in {udm_table_name} Table")

            # Update index
            self.__update_index(df=submodule_data)

            if not archived_alerts:
                return submodule_data

            # Downloading Archived data
            archived_table_name = f"{udm_table_name}ArchivedAlerts"
            archived_submodule_data = self.get_archived_data(archived_table_name)
            if archived_submodule_data is not None and len(archived_submodule_data) > 0:
                self.__update_index(df=archived_submodule_data)
                submodule_data = dd.concat(
                    [submodule_data, archived_submodule_data], ignore_index=True
                ).drop_duplicates()

                Status.SUCCESS(
                    f"Data downloaded and merged from {udm_table_name} and {archived_table_name} tables",
                    self.submodule_obj,
                )
        except BaseException as _e:
            Status.WARNING(f"Can not download {archived_table_name} table")
            return None
        return submodule_data

    def get_filtered_questionnaire_data(self) -> Tuple[pd.Series, pd.Series]:
        """
        Retrieves and filters questionnaire data based on user input and returns
        the filtered data for concern and non-concern questions.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple containing two pandas Series:
                - The first Series contains the filtered data for concern questions.
                - The second Series contains the filtered data for non-concern questions.

        Raises:
            Status.FAILED: If no questionnaire data is found or if the filtered data is empty.
        """
        # get all questionnaire data
        all_questionnaire_data = self.__get_all_questionnaire_data()
        if all_questionnaire_data is None or len(all_questionnaire_data.index) == 0:
            Status.FAILED("No questionnaire data found", self.submodule_obj)
            return pd.Series(), pd.Series()

        # filter based on user input
        filter_cond = {
            "Module": self.submodule_obj.module,
            "SubModule": self.submodule_obj.submodule,
            "StatusDescription": self.submodule_obj.alert_status,
        }
        for key, value in filter_cond.items():
            all_questionnaire_data = all_questionnaire_data.loc[
                all_questionnaire_data[key] == value
            ]

        if len(all_questionnaire_data.index) == 0:
            Status.FAILED(
                "No questionnaire data found",
                self.submodule_obj,
                alert_status=self.submodule_obj.alert_status,
            )
            return pd.Series(), pd.Series()

        # get concern and non concern keys
        concern_question_data = self.__filter_by_responses(
            all_questionnaire_data, self.submodule_obj.concern_questionnaire
        )
        no_concern_question_data = self.__filter_by_responses(
            all_questionnaire_data, self.submodule_obj.no_concern_questionnaire
        )

        return (
            concern_question_data[self.index].drop_duplicates(),
            no_concern_question_data[self.index].drop_duplicates(),
        )

    def __get_all_questionnaire_data(self) -> Optional[pd.DataFrame]:
        """This function is used to get questionnaire data from database"""
        Status.INFO("Downloading all questionnaire data", self.submodule_obj)

        instance = GlobalSettings.instance_by_id(self.submodule_obj.instance_id)
        if not instance:
            Status.FAILED(
                "Instance not found",
                self.submodule_obj,
                alert_status=self.submodule_obj.alert_status,
            )
            return None
        instance_db = instance.settings.projectdb

        # Getting Table names
        query = config.get("QUESTIONNAIRE", "QUERY")
        ddf = instance_db.download_table_or_query(query=query, demo=False)
        if ddf is None or len(ddf.index) == 0:
            Status.FAILED(
                "No questionnaire data found",
                self.submodule_obj,
                alert_status=self.submodule_obj.alert_status,
            )
            return None
        return ddf.compute()

    def __filter_by_responses(
        self,
        all_questionnaire_data: pd.DataFrame,
        questionnaire: List[TemplateQuestion],
    ):
        """This function is used to filter questionnaire data based on user input"""
        training_data = pd.DataFrame(columns=all_questionnaire_data.columns)
        questionnaire_text = config.get("QUESTIONNAIRE", "QUESTIONNAIRE_TEXT_COLUMN")
        response_text = config.get("QUESTIONNAIRE", "RESPONSE_TEXT_COLUMN")

        for q in questionnaire:
            question_data = all_questionnaire_data.loc[
                (all_questionnaire_data[questionnaire_text] == q.question)
                & (all_questionnaire_data[response_text].isin(q.options))
            ]
            training_data = pd.concat([training_data, question_data])

        return training_data
