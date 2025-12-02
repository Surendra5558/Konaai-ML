# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains functions to calculate Benford's Law anomaly."""
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from src.utils.status import Status


def first_digits(value: Union[int, float], test_digits=2) -> int:
    """
    Extracts the first 'test_digits' digits from the given number.
    Parameters:
    ---------
    value (Union[int, float]): The number from which to extract the first digits.
    test_digits (int, optional): The number of digits to extract. Defaults to 2.

    Returns:
    int: The first 'test_digits' digits of the absolute value of the number.
         Returns 0 if an exception occurs or if the absolute value of the number
         is less than 10 ** (test_digits - 1).
    """
    try:
        number = float(value)

        # Should not be less than 10 ** (test_digits - 1)
        if abs(number) < 10 ** (test_digits - 1):
            return int(abs(number))

        # Take only the number before the decimal point and mark absolute value
        number_str = str(abs(number)).split(".", maxsplit=1)[0]
        return int(number_str[:test_digits])
    except Exception:
        return 0


# Calculate the standard error
def calc_std_error(prob, n) -> float:
    """
    Calculate the standard error for a given probability and sample size.

    The standard error is a measure of the statistical accuracy of an estimate,
    calculated as the standard deviation of the sampling distribution of a statistic,
    most commonly of the mean.

    Parameters:
    prob (float): The probability value (between 0 and 1).
    n (int): The sample size.

    Returns:
    float: The standard error of the probability.
    """
    return np.sqrt(prob * (1 - prob) / n)


def null_hypothesis_test(
    s: pd.Series, alpha: float = 0.01, test_digits: int = 2
) -> bool:
    """
    Perform a chi-square test to determine if the observed frequencies of the
    leading digits in a dataset follow Benford's Law.
    Parameters:
    ----------
    s (pd.Series): A pandas Series containing the data to be tested.
    alpha (float): The significance level for the test. Default is 0.01.
    test_digits (int): The number of leading digits to test. Default is 2.

    Returns:
    bool: True if the null hypothesis (that the data follows Benford's Law)
          is rejected, False otherwise.
    """
    s = s.dropna().sort_index()
    digits_range = range(10 ** (test_digits - 1), 10**test_digits)
    s = s.loc[s.index.isin(digits_range)]

    observed_freq = s.to_numpy()
    expected_freq = np.array([sum(s) * np.log10(1 + 1 / i) for i in s.index])

    # Normalize observed and expected frequencies to ensure sums match
    observed_freq_sum = observed_freq.sum()
    expected_freq_sum = expected_freq.sum()

    # If the sum of observed and expected frequencies do not match, normalize the frequencies
    if observed_freq_sum != expected_freq_sum:
        observed_freq = observed_freq / observed_freq_sum
        expected_freq = expected_freq / expected_freq_sum

    # Perform the chi-square test
    _, p_value = chisquare(observed_freq, expected_freq)

    return p_value < alpha


def calc_benford_anomaly(s: pd.Series, test_digits=2) -> pd.Series:
    """
    Calculate Benford's Law anomaly for a given pandas Series.
    Parameters:
    ---------
    s (pd.Series): The input series containing numerical values.
    test_digits (int): The number of leading digits to test (default is 2).

    Returns:
    pd.Series: A series indicating the Benford anomaly score for each value in the input series.
               The score is normalized between 0 and 1, where 0 indicates no anomaly and 1 indicates the highest anomaly.

    Raises:
    Status.INVALID_INPUT: If Benford's law is not applicable for the distribution of the input series.
    Status.FAILED: If there is an error while calculating Benford's Law anomaly.
    """
    try:
        # Create DataFrame from the series
        value = "value"
        df = s.to_frame(name=value)

        # Filter out zero and negative values
        df = df[df[value] > 0]

        # Check minimum sample size
        if len(df) < 50:
            return pd.Series([0.0] * len(s), index=s.index).astype(float)

        # Calculate the first digit
        df["first_digits"] = (
            df[value].map(lambda x: first_digits(x, test_digits)).astype("int8")
        )
        digits_range = range(10 ** (test_digits - 1), 10**test_digits)

        # Calculate the observed frequency and probabilities
        first_digits_frequency = df["first_digits"].value_counts().astype("int8")

        if null_hypothesis_test(first_digits_frequency, test_digits=test_digits):
            # Status.INVALID_INPUT(
            #     "Benford's law is not applicable for this distribution"
            # )
            return pd.Series([0.0] * len(s), index=s.index).astype(float)

        df["first_digits_probs"] = (
            (df["first_digits"].map(first_digits_frequency) / len(df))
            .fillna(0)
            .astype(float)
        )

        # Calculate Benford probabilities
        benford_probs = {x: np.log10(1 + 1 / x) for x in digits_range}
        df["benford_probs"] = (
            df["first_digits"].map(benford_probs).fillna(0).astype(float)
        )

        # Calculate standard error and upper bound
        std_errors = {
            x: calc_std_error(benford_probs[x], len(df)) for x in digits_range
        }
        df["std_error"] = df["first_digits"].map(std_errors).fillna(0)
        df["upper_bound"] = (
            df["benford_probs"] + 2.3263 * df["std_error"]
        )  # 99% confidence interval

        # Determine if there is a Benford anomaly
        df["benford_anomaly"] = np.maximum(
            0, df["first_digits_probs"] - df["upper_bound"]
        )

        # Additional filter: only consider anomalies if the digit frequency is significantly high
        min_freq_threshold = max(5, len(df) * 0.01)
        digit_counts = df["first_digits"].map(first_digits_frequency)
        df["benford_anomaly"] = df["benford_anomaly"].where(
            digit_counts >= min_freq_threshold, 0
        )

        # find max value of benford_anomaly and divide all values by it to normalize
        max_anomaly = df["benford_anomaly"].max()
        if max_anomaly > 0:  # Avoid division by zero
            df["benford_anomaly"] = (
                df["benford_anomaly"] / max_anomaly
            ) ** 2  # decrease sensitivity
            df["benford_anomaly"] = df["benford_anomaly"].round(3)
        else:
            df["benford_anomaly"] = 0

        # if first_digits other than the range, set anomaly to 0
        df["benford_anomaly"] = df["benford_anomaly"].where(
            df["first_digits"].isin(digits_range), 0
        )

        # create result series
        result = pd.Series([0.0] * len(s), index=s.index, dtype=float)
        result.loc[df.index] = df["benford_anomaly"]

        return result.astype(float)
    except BaseException as _e:
        Status.FAILED(f"Error while calculating Benford's Low anomaly: {_e}")
    return pd.Series([False] * len(df)).astype(float)
