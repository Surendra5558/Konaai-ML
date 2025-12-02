# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to encode numerical data"""
import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Self
from typing import Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from src.tools.dask_tools import compute
from src.utils.status import Status
from tqdm import tqdm


# Custom Scaler class to store the median and iqr values
# We had to do this because dask-ml RobustScalar has numpy version dependency issues
# Also using Robust Scalar methodology to scale the data as it is robust to outliers
class Scaler:
    """
    A class used to scale numeric data using median and interquartile range (IQR).
    Attributes
    ----------
    median : Dict
        A dictionary containing the median values for each feature.
    iqr : Dict
        A dictionary containing the interquartile range values for each feature.
    Methods
    -------
    transform(X: dd.DataFrame) -> dd.DataFrame
        Scales the input DataFrame X using the stored median and IQR values.
    """

    def __init__(self):
        self.median: Dict = None
        self.iqr: Dict = None

    def fit(self, X: dd.DataFrame) -> Self:
        """
        Fits the transformer to the input Dask DataFrame by computing the median and interquartile range (IQR) for each column.

        Parameters:
            X (dd.DataFrame): Input Dask DataFrame containing numeric features.

        Returns:
            Self: Returns the fitted transformer instance with computed median and IQR values stored as attributes.

        Notes:
            - The IQR for each column is calculated as the difference between the 75th and 25th percentiles.
            - If the IQR for any column is zero, it is replaced with a small epsilon (1e-9) to avoid division by zero errors.
            - The median is computed using an approximate method for efficiency.
        """

        # Calculate the IQR for each column
        q1 = X.quantile(0.25).compute()
        q3 = X.quantile(0.75).compute()
        iqr = (q3 - q1).to_dict()
        # when iqr is 0, replace it with epsilon to avoid division by zero
        iqr = {k: v if v != 0 else 1e-9 for k, v in iqr.items()}

        # Calculate the median for each column
        median = X.median_approximate().compute().to_dict()

        # store the median and iqr values
        self.median = median
        self.iqr = iqr

        return self

    def transform(self, X: dd.DataFrame) -> dd.DataFrame:
        """Transform the input DataFrame X using the stored median and IQR values"""
        return (X - self.median) / self.iqr

    def reverse_transform(self, X: dd.DataFrame) -> dd.DataFrame:
        """Reverse transform the input DataFrame X using the stored median and IQR values"""
        return (X * self.iqr) + self.median

    def reverse_transform_single(self, x: float, column_name: str) -> float:
        """Reverse transform a single value X using the stored median and IQR values"""
        return (x * self.iqr[column_name]) + self.median[column_name]


class NumericEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer for encoding numeric columns by binning and normalizing the data,
    and generating polynomial features.
    """

    range_suffix = "_Range"
    power_sign = "^"

    def __init__(self, columns: List[str], unknown_replacement: str) -> None:
        self.columns: List[str] = columns
        self.unknown_replacement: str = unknown_replacement
        self.derived_categories: List[str] = []
        self.feature_names_: List[str] = None
        self.scaler: Scaler = None

    def _create_bin_edges(self, X: dd.DataFrame, column_name: str) -> List[int]:
        """
        Generate bin edges for a given column in a Dask DataFrame.
        This method creates bin edges based on the maximum value in the specified column.
        It starts with an initial step and limit, then increases these values by a factor
        of 10 until the limit exceeds the maximum value in the column. The bin edges are
        generated for each range and returned as a list of floats.
        Args:
        -----
            X (dd.DataFrame): The input Dask DataFrame.
            column_name (str): The name of the column for which to create bin edges.
        Returns:
            List[int]: A list of bin edges.
        """
        step_ranges = []
        step = 1
        limit = 10
        max_value = X[column_name].max().compute()

        # generate step ranges
        while limit <= max_value:
            step_ranges.append((step, limit))
            step = limit
            limit *= 10

        # add the last range
        step_ranges.append((step, step * 10))
        # Initial bin edges
        bin_edges = []

        # Add bin edges for each range
        for step, max_range in step_ranges:
            current_edge = bin_edges[-1] if bin_edges else 0
            while current_edge <= max_range:
                if current_edge in bin_edges:
                    current_edge += step
                    continue

                bin_edges.append(current_edge)
                current_edge += step
        return bin_edges

    def create_bins(self, X: dd.DataFrame, column_name: str) -> dd.DataFrame:
        """
        Create bins for a specified column in a Dask DataFrame and map the values to bin ranges.
        Parameters:
        -----------
        X : dd.DataFrame
            The input Dask DataFrame.
        column_name : str
            The name of the column to be binned.
        Returns:
        --------
        dd.DataFrame
            The Dask DataFrame with an additional column containing the binned ranges.
        Notes:
        ------
        - The method creates bin edges and labels based on the specified column.
        - The original DataFrame is copied to avoid modifying the input data.
        - The new binned column is added to the DataFrame with a name derived from the original column name and a suffix.
        - The derived column name is appended to the list of derived categories.
        """
        Status.INFO(f"Creating bins for column: {column_name}")

        # during the fitting, X memory location will be updated hence making a copy of it
        X = X.copy()
        bin_edges = self._create_bin_edges(X, column_name)
        # create bin labels as a list of strings
        bin_labels = [
            f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)
        ]

        binned_column_name = f"{column_name}{self.range_suffix}"

        # Map the bin numbers to bin ranges
        X[binned_column_name] = X[column_name].map_partitions(
            lambda s: pd.cut(s, bins=bin_edges, labels=bin_labels, include_lowest=True),
            meta=(binned_column_name, "string[pyarrow]"),
        )

        # Add the derived column to the list of derived categories
        self.derived_categories.append(binned_column_name)
        return compute(X)

    def get_feature_names_out(self):
        """
        Get the names of the generated polynomial features.
        Returns:
        List[str]: A list of feature names.
        """
        return list(set(list(self.feature_names_) + self.derived_categories))

    def create_polynomial_features(
        self, X: dd.DataFrame, degree: int = 2
    ) -> Tuple[dd.DataFrame, List[str]]:
        """
        Generate polynomial features for the given DataFrame.
        Parameters:
        -----------
        X : dd.DataFrame
            The input DataFrame containing the features to be transformed.
        degree : int, optional, default=2
            The degree of the polynomial features to be generated.
        Returns:
        --------
        Tuple[dd.DataFrame, List[str]]
            A tuple containing the transformed DataFrame with polynomial features and a list of the new feature names.
        """
        Status.INFO("Creating polynomial features")

        data = compute(X[self.columns])
        if isinstance(data, dd.Series):
            data = data.to_frame()

        # for each column, create polynomial features
        feature_names = []
        for col in self.columns:
            for d in range(1, degree + 1):
                feature_name = f"{col}{self.power_sign}{d}"
                data[feature_name] = data[col] ** d
                feature_names.append(feature_name)

        # update dtypes for the new columns
        return data.astype(float), feature_names

    def get_polynomial_value(self, x: float, column_name: str) -> Optional[float]:
        """
        Calculates the value of a polynomial feature for a given input.
        If the column name indicates a power feature (contains `self.power_sign`), this method extracts the power from the column name,
        raises the input `x` to that power, and returns the result rounded to two decimal places. If the column name does not indicate
        a power feature or if `x` is falsy (e.g., 0 or None), returns None.
        Args:
        ----
            x (float): The input value to be raised to a power.
            column_name (str): The name of the column, expected to contain the power sign and the exponent.
        Returns:
            Optional[float]: The computed power value rounded to two decimals, or None if not applicable.
        """
        if (self.power_sign not in column_name) or not x:
            return None

        nth_power = int(column_name.split(self.power_sign)[1])
        value = np.power(x, nth_power)
        return round(value, 2)

    def fit(self, X: dd.DataFrame, y=None):  # pylint: disable=unused-argument
        """
        Fits the transformer to the provided Dask DataFrame by filling missing values in specified numeric columns with 0,
        generating polynomial features, and fitting a scaler to the data.
        Parameters:
        ----------
            X (dd.DataFrame): Input Dask DataFrame containing the features to be transformed.
            y (optional): Ignored. Present for compatibility.
        Returns:
            self: Returns the fitted transformer instance.
        """
        # fill missing values with 0 for numeric columns
        # Its necessary as polynomial features cannot handle missing values
        # Also 0 is a good replacement for missing values in numeric columns since we are considering only amount and anomaly columns
        X[self.columns] = X[self.columns].fillna(0)

        # create polynomial features
        _, feature_names = self.create_polynomial_features(X)
        if len(feature_names) > 0:
            self.feature_names_ = feature_names

        # fit the scaler
        self.scaler = Scaler().fit(X[self.columns])

        return self

    def transform(
        self, X: dd.DataFrame
    ) -> dd.DataFrame:  # pylint: disable=unused-argument
        """
        Transforms the input Dask DataFrame by binning, normalizing, and creating polynomial features.
        Parameters:
        ----------
        X (dd.DataFrame): The input Dask DataFrame to be transformed.
        Returns:
        dd.DataFrame: The transformed Dask DataFrame with binned, standardized, and polynomial features.
        """
        # fill missing values with 0 for numeric columns
        # Its necessary as polynomial features cannot handle missing values
        # Also 0 is a good replacement for missing values in numeric columns since we are considering only amount and anomaly columns
        X[self.columns] = X[self.columns].fillna(0)

        # bin the columns
        for column_name in tqdm(
            self.columns, desc="Creating numerical bins", file=sys.stdout
        ):
            X = self.create_bins(X, column_name)

        # Create polynomial features
        X_scaled = None
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X[self.columns])

        # if there is scaled data, create polynomial features
        if X_scaled is not None and len(X_scaled) > 0:
            result, _ = self.create_polynomial_features(X_scaled)

            # drop the original columns
            X = X.drop(self.columns, axis=1)
            # merge the dataframes
            X = X.merge(result, left_index=True, right_index=True)

        return compute(X)
