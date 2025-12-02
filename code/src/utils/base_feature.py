# # Copyright (C) KonaAI - All Rights Reserved
"""This module provide a base ML feature structure"""
import abc
import sys
from typing import List

import dask.dataframe as dd
from src.tools.dask_tools import compute
from src.tools.dask_tools import optimize_dtypes
from src.tools.dask_tools import validate_column
from src.utils.file_mgmt import file_handler
from src.utils.notification import EmailNotification
from src.utils.status import Status
from src.utils.submodule import Submodule
from tqdm import tqdm


class BaseFeatureBuilder(metaclass=abc.ABCMeta):
    """
    BaseFeatureBuilder is an abstract base class that provides a standardized interface and utility methods for building machine learning feature classes. It manages feature metadata, input/output validation, data loading, and feature processing for ML pipelines.
    Attributes:
        _data_path (str): Path to the input data file.
        _index (str): Name of the index column.
        _risk_level (str): Risk level of the feature ("High", "Medium", "Low").
        _description (str): Description of the feature.
        _deprecated (bool): Indicates if the feature is deprecated.
        _input_cols (List): List of input columns required for feature generation.
        _feature_names (List): List of feature names.
        _feature_types (List): List of feature types ("Boolean", "Date", "Categorical", "Numerical").
        _submodule (str): Name of the submodule associated with the feature.
    """

    def __init__(self, input_data_path: str, index_column: str) -> None:
        self._data_path: str = input_data_path
        self._index: str = index_column
        self._risk_level: str = "Medium"
        self._description: str = None
        self._deprecated: bool = False
        self._input_cols: List = []
        self._feature_names: List = []
        self._feature_types: List = []
        self._submodule: str = None

    @classmethod
    def class_type(cls) -> str:
        """class type

        Returns:
                str: Static value 'feature'
        """
        return "feature"

    @property
    def submodule(self):
        """Submodule Property"""
        return self._submodule

    @submodule.setter
    def submodule(self, value: str):
        """Submodule Setter

        Args:
                value (str): Submodule name
        """
        self._submodule = value

    @property
    def risk_level(self) -> str:
        """Risk Level Property"""
        return self._risk_level

    @risk_level.setter
    def risk_level(self, value: str) -> None:
        """Risk Level Setter

        Args:
                value (str): Risk Level

        Raises:
                ValueError: Only allows pre-defined values
        """
        valid_values = ["High", "Medium", "Low"]
        if value not in valid_values:
            Status.FAILED(
                f"Risk level can only be one of {valid_values} for {self.__class__.__name__}"
            )
            raise ValueError()
        self._risk_level = value

    @risk_level.deleter
    def risk_level(self) -> None:
        """Risk Level Deleter

        Raises:
                AttributeError: Stop from deleting Risk Level
        """
        raise AttributeError("Attribute can not be deleted")

    @property
    def description(self) -> str:
        """Feature Description property"""
        return self._description

    @description.setter
    def description(self, value: str):
        """Description Setter

        Args:
                value (str): Feature Description
        """
        self._description = value

    @property
    def deprecated(self):
        """Deprecated feature property"""
        return self._deprecated

    @deprecated.setter
    def deprecated(self, value: bool):
        """Allows changing deprecated setting

        Args:
                value (bool): Feature deprecated
        """
        self._deprecated = value

    @property
    def input_cols(self):
        """List of columns to use to generate features"""
        return self._input_cols

    @input_cols.setter
    def input_cols(self, value: list):
        """Sets list of columns to use to generate features

        Args:
                value (list): List of columns from dataframe
        """
        self._input_cols = value

    @property
    def features_types(self):
        """Feature Types property"""
        return self._feature_types

    @features_types.setter
    def feature_types(self, values: list):
        """Feature Type Property Setter

        Args:
                values (list): List of Feature Types from pre-defined values

        Raises:
                ValueError: Error when input values are not from pre-defined values
        """
        valid_values = ["Boolean", "Date", "Categorical", "Numerical"]
        is_valid_values = [False for value in values if value not in valid_values]
        if all(value is True for value in is_valid_values):
            self._feature_types = values
        else:
            Status.FAILED(
                f"Feature type can only be from {valid_values} for {self.__class__.__name__}"
            )
            raise ValueError()

    def __add_feature_type_prefix__(self, values: list):
        """Adds a prefix based on type of feature

        Args:
                values (list): List of feature names

        Returns:
                list: Updated list of feature names
        """
        prefix = ""
        for idx, feature_name in enumerate(values):
            feature_type = self.feature_types[idx]
            if feature_type == "Boolean":
                prefix = "b:"
            elif feature_type == "Date":
                prefix = "d:"
            elif feature_type == "Categorical":
                prefix = "c:"
            elif feature_type == "Numerical":
                prefix = "n:"
            values[idx] = prefix + str(feature_name)
        return values

    @property
    def features_names(self):
        """Feature Names Property

        Returns:
                list: List of updated feature names
        """
        result = self._feature_names.copy()
        try:
            # Check if feature types are assigned or not
            if len(self._feature_names) == len(self._feature_types):
                # add feature type prefix to feature names
                # values = self.__add_feature_type_prefix__(result)

                # add feature class name prefix to feature names
                values = [f"{self.__class__.__name__}_{value}" for value in result]

                result = values
            else:
                Status.FAILED(
                    f"First assign feature types before feature names for {self.__class__.__name__}"
                )
                raise TypeError("First assign feature types before feature names")
        except BaseException as _e:
            Status.FAILED(
                f"Can not assign feature names for feature {self.__class__.__name__}",
                error=str(_e),
            )
        return result

    @features_names.setter
    def feature_names(self, values: List):
        """Feature Names setter

        Args:
                values (list): List of input feature names
        """
        self._feature_names = values

    def __feature_validation__(self):
        """Basic sanity checks"""
        # Check if feature is deprecated
        if self.deprecated is True:
            Status.FAILED(
                f"This feature builder is deprecated: {self.__class__.__name__}"
            )

        valid_feature = (
            self._data_path is None
            # or self._index is None
            or self.description is None
            or len(self.input_cols) == 0
            or len(self._feature_names) == 0
            or len(self._feature_types) == 0
        )

        # Check if mandatory attributes and data is present
        if valid_feature:
            # create a list of invalid attributes
            invalid_attributes = []
            if self._data_path is None:
                invalid_attributes.append("data_path")
            # if self._index is None:
            #     invalid_attributes.append("index")
            if self.description is None:
                invalid_attributes.append("description")
            if self.input_cols is None or len(self.input_cols) == 0:
                invalid_attributes.append("input_cols")
            if len(self._feature_names) == 0:
                invalid_attributes.append("feature_names")
            if len(self._feature_types) == 0:
                invalid_attributes.append("feature_types")

            # log error and raise exception
            Status.FAILED(
                f"Following Attributes can not be empty: {invalid_attributes} for {self.__class__.__name__}"
            )
            raise ValueError(f"Invalid Attributes for {self.__class__.__name__}")

    def optimize(self, _df):
        """This function optimizes dataframe data types for memory efficiency

        Args:
                df (dataframe): Input dataframe

        Returns:
                dataframe: Modified dataframe
        """
        return optimize_dtypes(_df)

    def load_data(self, submodule: Submodule) -> dd.DataFrame:
        """
        Loads data from a parquet file, validates the feature, and processes the data.
        This method performs the following steps:
        1. Validates the feature instance.
        2. Reads specific columns from the parquet file located at `self._data_path`.
        3. Checks if the required input columns are present in the data.
        4. Validates and sets the index column.
        5. Resets the index to a column to remove duplicates based on the index.
        6. Drops duplicates and sets the index back.
        7. Returns the processed DataFrame with the specified input columns.
        Returns:
            dd.DataFrame: The processed Dask DataFrame with the specified input columns.
        Raises:
            ValueError: If the required input columns or index column are not found in the data.
            BaseException: If there is an error loading the data.
        Logs:
            Logs various steps and errors during the data loading and processing.
        """
        try:
            return self._load_data(submodule=submodule)
        except ValueError as _e:
            Status.NOT_FOUND("Required column not found, skipping", error=str(_e))
        except BaseException as _e:
            Status.FAILED(
                f"Can not load data for feature {self.__class__.__name__}",
                error=str(_e),
            )
        return None

    def _load_data(self, submodule: Submodule) -> dd.DataFrame:
        # run validation on new instance
        self.__feature_validation__()

        # Read only specific columns from the parquet file
        _df: dd.DataFrame = dd.read_parquet(self._data_path)

        if _df is None or len(_df) == 0:
            raise ValueError("No data found")

        if not_present := [col for col in self.input_cols if col not in _df.columns]:
            s = Status.NOT_FOUND(
                f"Required input columns not found in the data for feature {self.__class__.__name__}",
                submodule,
                not_present=not_present,
                feature_names=self.feature_names,
            )
            notifier = EmailNotification(instance_id=submodule.instance_id)
            notifier.add_content(
                "Required input columns not found",
                content=s.to_dict(),
            )
            notifier.send(
                subject=f"Required input columns not found for {submodule.instance_id}"
            )

            raise ValueError(f"Required input columns not found : {not_present}")

        # # find out correct index column
        self._index = validate_column(self._index, _df)
        if self._index is None:
            raise ValueError("Index column name not defined.")

        _df = _df.set_index(self._index)

        # move index as column
        # We need to do this since we have to drop duplicates based on this index
        _df = _df.reset_index(drop=False)

        # remove duplicates by index and keep the first one
        _df = _df.drop_duplicates(subset=[self._index], keep="first")

        # move index as index
        _df = _df.set_index(self._index)

        # result = self.optimize(_df[self.input_cols].set_index(self._index))
        return compute(_df[self.input_cols])

    def output_validation(self, output_df: dd.DataFrame, submodule: Submodule) -> bool:
        """This fuction help to validate incoming index is right or not by checking index in feature class
        Retrurn:
            Boolean Value in True or False
        """
        try:
            self._output_validation(output_df, submodule)
        except BaseException as _e:
            Status.FAILED(
                f"Output validation failed for feature {self.__class__.__name__}",
                error=str(_e),
            )
            return False
        return True

    def _output_validation(self, output_df: dd.DataFrame, submodule: Submodule) -> bool:
        # check if incoming dataframe has an index of same name as the one defined in the feature class
        if self._index not in output_df.index.name:
            raise ValueError(
                f"Index name {output_df.index.name} does not match with the one defined in the feature class {self._index}"
            )
        input_df = self.load_data(submodule)

        # Check number of record
        if len(output_df) != len(input_df):
            raise ValueError("Input and Output length mismatch")

        min_index = input_df.index.min().compute()
        max_index = input_df.index.max().compute()

        # Check input and outputs min and max value of index
        if (
            output_df.index.min().compute() != min_index
            and output_df.index.max().compute() != max_index
        ):
            raise ValueError("Input and output index min and max value mismatch")

        return True

    def write_output(self, _df: dd.DataFrame, submodule: Submodule) -> str:
        """This function write the feature output to a parquet file with pre-defined index

        Args:
                df (dataframe): input dataframe

        Returns:
                str: path of output parquet file
        """
        Status.INFO("Saving feature output")

        if _df is None or len(_df) == 0:
            Status.FAILED("Output dataframe is empty")
            return None

        out_file = None
        if not self.output_validation(_df, submodule):
            return None

        # Rename columns
        column_mapper = dict(zip(self._feature_names, self.feature_names))

        _df = _df.rename(columns=column_mapper)

        # drop columns that are not in feature names
        _df = _df.drop(
            columns=[col for col in _df.columns if col not in self.feature_names]
        )

        # confirm if all feature names are present in the dataframe
        if any(col not in _df.columns for col in self.feature_names):
            Status.FAILED(
                f"Feature names are not present in the output dataframe for {self.__class__.__name__}"
            )
            return None

        # Save dataframe to the common context folder
        # get new file name
        _, file_path = file_handler.get_new_file_name("parquet")
        result = _df.to_parquet(
            file_path,
            engine="pyarrow",
            compression="snappy",
            write_index=True,
            compute=True,
        )

        if result is None:
            out_file = file_path
        else:
            Status.FAILED(f"Feature output file can not be saved to {file_path}")
        Status.SUCCESS("Feature output saved to", path=out_file)

        return out_file

    def impute(self, _df: dd.DataFrame, values: List) -> dd.DataFrame:
        """This function imputes missing values for features

        Args:
                df (dataframe): input dataframe
                values (list): list of impute value for each feature in the same order

        Returns:
                dataframe: output dataframe
        """

        # check if values are list
        if not isinstance(values, list):
            Status.FAILED(
                f"Imputation values must be a list for {self.__class__.__name__}"
            )
            raise TypeError()

        # check if imputation items equal to total features
        if len(self._feature_names) != len(values):
            Status.FAILED(
                f"Total imputation items do not match with total features for {self.__class__.__name__}"
            )
            raise IndexError()

        try:
            # fill missing values
            # impute values one by one
            for idx, feature_name in tqdm(
                enumerate(self._feature_names),
                desc="Imputing feature",
                total=len(self._feature_names),
                file=sys.stdout,
            ):
                try:
                    _df[feature_name] = _df[feature_name].fillna(values[idx])
                except BaseException as _e:
                    Status.FAILED(f"Imputation failed for {feature_name}. Error - {_e}")

        except BaseException as _e:
            Status.FAILED(f"Imputation failed for {self._feature_names}", error=str(_e))
            _df = None
        return _df

    @abc.abstractmethod
    def run(self, _df):
        """mandatory implementation function in inherited class to
        automatically run it when initiated"""
        pass
