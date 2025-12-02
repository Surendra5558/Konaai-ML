# # Copyright (C) KonaAI - All Rights Reserved
"""This module handles datetime encoding process for the AutoML pipeline."""
import sys
from typing import List
from typing import Optional
from typing import Self

import dask.dataframe as dd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from src.automl.numeric_encoder import NumericEncoder
from src.tools.dask_tools import compute
from src.utils.status import Status
from tqdm import tqdm


class DateTimeEncoder(BaseEstimator, TransformerMixin):
    """This class handles datetime encoding process for the AutoML pipeline."""

    quarter_suffix = "_Qtr"
    month_suffix = "_Month"
    day_suffix = "_DayOfMonth"
    weekday_suffix = "_WkDay"

    scaler: NumericEncoder = None

    def __init__(self, columns: List[str], unknown_replacement: str) -> None:
        self.columns: List[str] = columns
        self.feature_names_: List[str] = []
        self.derived_categories: List[str] = []
        self.unknown_replacement = unknown_replacement

    def get_feature_names_out(self):
        """
        Get the names of the generated polynomial features.
        Returns:
        List[str]: A list of feature names.
        """
        return list(set(self.feature_names_ + self.derived_categories))

    def _create_suffix_feature_name(self, column_name: str, suffix: str) -> str:
        column_name = str(column_name).strip()
        return f"{column_name}{suffix}"

    def _extract_date_time_features(self, X: dd.DataFrame, column: str) -> dd.DataFrame:
        """
        Extract datetime features from a column and add them to the DataFrame.

        Parameters:
            X (dd.DataFrame): The input Dask DataFrame
            column (str): The datetime column to process

        Returns:
            dd.DataFrame: DataFrame with additional datetime features
        """
        # Ensure the column is in datetime format
        X[column] = dd.to_datetime(X[column], errors="coerce")

        # Extract year, month, day, weekday, and quarter
        qtr_feature = self._create_suffix_feature_name(column, self.quarter_suffix)
        X[qtr_feature] = X[column].dt.quarter
        X[qtr_feature] = X[qtr_feature].astype("string[pyarrow]")

        month_feature = self._create_suffix_feature_name(column, self.month_suffix)
        X[month_feature] = X[column].dt.month_name()
        X[month_feature] = X[month_feature].astype("string[pyarrow]")

        weekday_feature = self._create_suffix_feature_name(column, self.weekday_suffix)
        X[weekday_feature] = X[column].dt.day_name()
        X[weekday_feature] = X[weekday_feature].astype("string[pyarrow]")

        # keeping day feature as numeric to avoid large feature space
        # Also to avoid overfitting
        # Day feature will be a numeric feature and others will be categorical
        day_feature = self._create_suffix_feature_name(column, self.day_suffix)
        X[day_feature] = X[column].dt.day

        return compute(X)  # Compute the result before returning X

    def fit(
        self,
        X: dd.DataFrame,
        y: Optional[dd.Series] = None,  # pylint: disable=unused-argument
    ) -> Self:
        """
        Fit the DateTimeEncoder to the data.

        Parameters:
            X (dd.DataFrame): The input data as a Dask DataFrame.
            y (Optional[dd.Series]): The target variable, not used in this encoder.

        Returns:
            Self: Returns the fitted instance of DateTimeEncoder.
        """
        if len(self.columns) == 0:
            Status.NOT_FOUND("No columns to onehot encode")
        else:
            Status.INFO(f"Fitting datetime encoder for {len(self.columns)} columns")

        for column in self.columns:
            if column not in X.columns:
                Status.NOT_FOUND(f"Column {column} not found in the dataset")
                continue

            self.feature_names_.extend(
                [
                    self._create_suffix_feature_name(column, self.quarter_suffix),
                    self._create_suffix_feature_name(column, self.month_suffix),
                    self._create_suffix_feature_name(column, self.day_suffix),
                    self._create_suffix_feature_name(column, self.weekday_suffix),
                ]
            )
            self.derived_categories.extend(
                [
                    self._create_suffix_feature_name(column, self.quarter_suffix),
                    self._create_suffix_feature_name(column, self.month_suffix),
                    self._create_suffix_feature_name(column, self.weekday_suffix),
                ]
            )

        return self

    def transform(self, X: dd.DataFrame) -> dd.DataFrame:
        """
        Transform the input data by extracting date and time features.

        Parameters:
            X (dd.DataFrame): The input data as a Dask DataFrame.

        Returns:
            dd.DataFrame: A Dask DataFrame with the transformed date and time features.
        """
        if len(self.columns) == 0:
            Status.NOT_FOUND("Found empty datetime encoder")
            return X

        # Create datetime features
        for column_name in tqdm(
            self.columns, desc="Creating datetime columns", file=sys.stdout
        ):
            X = self._extract_date_time_features(X, column_name)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        # drop the original datetime columns
        for column_name in self.columns:
            X = X.drop(column_name, axis=1)

        return compute(X)
