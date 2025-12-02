# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides functions to normalize data"""
from enum import Enum
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from src.utils.status import Status


class DistributionType(Enum):
    """
    Enumeration representing different types of distributions.
    """

    SYMMETRIC = "Symmetric"
    NORMAL = "Normal"
    UNIFORM = "Uniform"
    LOGNORMAL = "Lognormal"
    RIGHT_SKEWED = "Right skewed"
    LEFT_SKEWED = "Left skewed"
    MULTIMODAL = "Multimodal"
    DEGENERATE = "Degenerate"
    UNKNOWN = "Unknown"


class DistributionProcessor:
    """
    A class that provides methods for normalizing data and classifying the distribution type.

    Methods:
        standardize(data: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
            Standardize the given data based on its distribution type.

        classify_distribution(data: Union[np.ndarray, pd.Series, List]) -> DistributionType:
            Classify the distribution type of the given data.

    """

    @staticmethod
    def standardize(
        data: Union[np.ndarray, pd.Series, List], verbose: bool = False
    ) -> np.ndarray:
        """
        Standardize the given data based on its distribution type.

        Args:
        ----
            data (Union[np.ndarray, pd.Series, List]): The data to be normalized.
            verbose (bool): Whether to print verbose output.

        Returns:
            np.ndarray: The normalized data.

        """

        if len(data) == 0:
            return np.array([])

        dt = DistributionProcessor.classify_distribution(data)
        if verbose:
            Status.INFO(f"Standardizing distribution type: {dt.value}")

        try:
            if dt == DistributionType.DEGENERATE:
                # If the distribution is degenerate, return the data as is
                return np.array(data)

            if dt in (
                DistributionType.UNKNOWN,
                DistributionType.SYMMETRIC,
                DistributionType.UNIFORM,
            ):
                # Apply StandardScaler
                data = StandardScaler().fit_transform(np.array(data).reshape(-1, 1))
            elif dt in (
                DistributionType.LOGNORMAL,
                DistributionType.RIGHT_SKEWED,
                DistributionType.LEFT_SKEWED,
            ):
                # Apply PowerTransformer with yeo-johnson
                data = PowerTransformer(method="yeo-johnson").fit_transform(
                    np.array(data).reshape(-1, 1)
                )
            elif dt == DistributionType.MULTIMODAL:
                # Apply QuantileTransformer to normalize the data
                data = QuantileTransformer(output_distribution="normal").fit_transform(
                    np.array(data).reshape(-1, 1)
                )
        except Exception as _e:
            # if failed to normalize, use standard scaler
            data = StandardScaler().fit_transform(np.array(data).reshape(-1, 1))

        # Convert to numpy array
        return np.array(data)

    @staticmethod
    def classify_distribution(  # pylint: disable=too-many-return-statements
        data: Union[np.ndarray, pd.Series, List],
    ) -> DistributionType:
        """
        Classify the distribution type of the given data.

        Args:
        -----
            data (Union[np.ndarray, pd.Series, List]): The data to be classified.

        Returns:
            DistributionType: The classified distribution type.

        """
        # remove all nan values
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return DistributionType.UNKNOWN

        try:
            data = np.array(data, dtype=float)
        except (ValueError, TypeError):
            return DistributionType.UNKNOWN

        # Helper functions
        def check_normality(data) -> bool:
            """
            Check if the given data follows a normal distribution.
            Uses Shapiro-Wilk test for small datasets (<=5000), and D'Agostino and Pearson's test for larger datasets.

            Parameters:
                data (array-like): The data to be tested for normality.

            Returns:
                bool: True if the data follows a normal distribution False otherwise.
            """
            n = len(data)
            if n < 3:
                return False
            if n <= 5000:
                _, p = stats.shapiro(data)
                return p > 0.05
            # For n > 5000
            _, p = stats.normaltest(data)
            return p > 0.05

        def check_uniform(data) -> bool:
            """
            Check if the given data follows a uniform distribution.

            This function uses the Kolmogorov-Smirnov test to determine if the provided data
            comes from a uniform distribution. The test compares the empirical distribution
            of the data with the cumulative distribution function (CDF) of a uniform distribution.

            Args:
            ----
                data (array-like): The data to be tested for uniformity.

            Returns:
                bool: True if the data follows a uniform distribution (p-value > 0.05), False otherwise.
            """
            # Scale data to [0, 1] before applying the test
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max == data_min:
                return False  # Avoid division by zero
            scaled_data = (data - data_min) / (data_max - data_min)
            _, p = stats.kstest(scaled_data, stats.uniform.cdf, method="approx")
            return p > 0.05

        def check_skewness(data) -> DistributionType:
            """
            Determine the skewness of the given data.
            This function calculates the skewness of the provided data using the `stats.skew` method.
            It then classifies the distribution as right-skewed, left-skewed, or symmetric based on the skewness value.
            Args:
            ----
                data (array-like): The data for which skewness is to be calculated.
            Returns:
                DistributionType: An enumeration indicating whether the data is right-skewed, left-skewed, or symmetric.
            """
            skewness = stats.skew(data)
            if skewness > 0.5:
                return DistributionType.RIGHT_SKEWED
            if skewness < -0.5:
                return DistributionType.LEFT_SKEWED

            # check if symmetric
            return DistributionType.SYMMETRIC

        def check_multimodal(data) -> bool:
            """
            Check if the given data is multimodal.

            This function uses Kernel Density Estimation (KDE) with a Gaussian kernel
            to estimate the probability density function of the input data. It then
            identifies peaks in the estimated density function to determine if the
            data is multimodal.

            Parameters:
            data (array-like): The input data to be checked for multimodality.

            Returns:
            bool: True if the data is multimodal (i.e., has more than one peak),
                  False otherwise.
            """
            kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(
                np.array(data).reshape(-1, 1)
            )
            s = np.linspace(min(data), max(data), 1000)
            e = kde.score_samples(s.reshape(-1, 1))
            peaks, _ = find_peaks(np.exp(e))
            return len(peaks) > 1

        # check if all values are the same
        if len(set(data)) == 1:
            # It is a degenerate distribution
            return DistributionType.DEGENERATE

        # Classification
        if check_normality(data):
            return DistributionType.NORMAL

        skewness = check_skewness(data)
        # check if symmetric and normal distribution
        if skewness == DistributionType.SYMMETRIC and check_uniform(data):
            return DistributionType.UNIFORM

        # check if multimodal distribution
        if check_multimodal(data):
            return DistributionType.MULTIMODAL

        return (
            skewness
            if isinstance(skewness, DistributionType)
            else DistributionType.UNKNOWN
        )
