# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides outlier detection functions"""
from typing import List
from typing import Union

import numpy as np
import pandas as pd


# Write a function that taken a series as input and returns a boolean list for IQR based outlier detection
def detect_outliers_iqr(_s: Union[List, pd.Series], with_confidence=False):
    """
    Detects outliers in a dataset using the Interquartile Range (IQR) method.
    Parameters:
    _s (Union[List, pd.Series]): The input data, either as a list or a pandas Series.
    with_confidence (bool, optional): If True, returns the confidence of the outlier. Defaults to False.
    Returns:
    List: A list where each element is either 0 (not an outlier) or 1 (outlier) if with_confidence is False.
          If with_confidence is True, returns a list where each element is the confidence value of the outlier.
    confidence (bool, optional): If True, returns the confidence of the outlier. Defaults to False.
    """
    if not isinstance(_s, (list, pd.Series)):
        raise TypeError("Input must be a list or a pandas Series.")

    if isinstance(_s, list):
        _s = pd.Series(_s)

    # convert all values to absolute
    _s = _s.abs()

    # convert all values a scale of 0 to 1
    _s = (_s / _s.max()).round(4)

    # calculate interquartile range
    _q1 = _s.quantile(0.25)
    _q3 = _s.quantile(0.75)
    iqr = _q3 - _q1

    # calculate lower and upper bounds
    lower_bound = _q1 - (1.5 * iqr)
    upper_bound = _q3 + (1.5 * iqr)

    if not with_confidence:
        return [1 if x < lower_bound or x > upper_bound else 0 for x in _s]

    mean = np.mean(_s)
    return [
        abs(mean - x) if x <= lower_bound else abs(x - mean) if x >= upper_bound else 0
        for x in _s
    ]
