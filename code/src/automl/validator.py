# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to create the submodule class"""
from typing import Tuple

import dask.dataframe as dd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from src.automl.utils import config
from src.utils.status import Status
from src.utils.submodule import Submodule


class DataValidator(BaseEstimator, TransformerMixin):
    """
    DataValidator is a scikit-learn compatible transformer for validating and preparing Dask DataFrames and Series for machine learning workflows.
    This class provides methods to:
    - Ensure the input data has the correct index column set, as specified in the configuration.
    - Validate that the input data (X) and target (y) are compatible in terms of length and partitioning, and automatically adjust partitions if necessary.

    """

    def __init__(self, submodule: Submodule) -> None:
        self.submodule = submodule

    def validate_index(self, X: dd.DataFrame) -> dd.DataFrame:
        """
        Validates and sets the index of the given Dask DataFrame.
        This method checks if the index of the DataFrame is set to the column specified
        in the configuration. If not, it sets the specified column as the index.
        Parameters:
        -----------
        X : dd.DataFrame
            The Dask DataFrame to validate and set the index for.
        Returns:
        --------
        dd.DataFrame
            The DataFrame with the specified column set as the index.
        Raises:
        -------
        ValueError
            If the specified index column is not found in the DataFrame.
        """

        # get index column name
        index = config.get("DATA", "INDEX")

        # check if index is already set as index
        if not X.index.name or X.index.name.lower() != index.lower():
            # if index is not already set, ensure index column is present in data
            if index.lower() not in [col.lower() for col in X.columns]:
                raise ValueError(f"Index column {index} not found in data")

            # get the index column name irrespective of case
            index = [col for col in X.columns if col.lower() == index.lower()]
            if len(index) > 1:
                index = index[0]
            else:
                raise ValueError(f"Index column {index} not found in data")

            # set index column as index
            Status.INFO(f"Setting column {index} as index", self.submodule)
            X = X.set_index(index)
        return X

    def validate_data(
        self, X: dd.DataFrame, y: dd.Series
    ) -> Tuple[dd.DataFrame, dd.Series]:
        """
        Validates the input data and target to ensure they are compatible for machine learning tasks.
        Parameters:
        -----------
        X : dd.DataFrame
            The input data to be validated.
        y : dd.Series
            The target data to be validated.
        Returns:
        --------
        Tuple[dd.DataFrame, dd.Series]
            The validated input data and target, potentially repartitioned to ensure compatibility.
        Raises:
        -------
        ValueError
            If the input data and target have different lengths.
        Notes:
        ------
        - If the input data and target have different numbers of partitions, the function will attempt to repartition
          them to match each other.
        - If either the input data or target has known divisions, the function will use those divisions to repartition
          the other.
        - If validation fails, a warning status will be logged.
        """

        try:
            # ensure that X and y are of same length
            if len(X) != len(y):
                raise ValueError("Data and target are of different lengths")

            # if both do not have known divisions, ensure that they have same number of partitions
            if X.npartitions != y.npartitions:
                Status.INFO(
                    "Data and target have different number of partitions. Repartitioning data to match target",
                    self.submodule,
                )

                # ensure both have known divisions
                known_divisions = X.known_divisions and y.known_divisions
                if not known_divisions:
                    # check which one has known divisions
                    if X.known_divisions:
                        # repartition y
                        y = y.repartition(divisions=X.divisions)
                    elif y.known_divisions:
                        # repartition X
                        X = X.repartition(divisions=y.divisions)
        except BaseException as _e:
            Status.WARNING("Validation failed for input data and target")
        return X, y

    def fit(self, X, y):  # pylint: disable=unused-argument
        """This function is used to fit the data"""
        return self

    def transform(self, X, y):
        """This function is used to transform the data"""
        return self.validate_data(X, y)
