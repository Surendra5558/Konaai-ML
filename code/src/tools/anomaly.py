# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides anomaly detection functions

Returns:
    _type_: _description_
"""
from typing import List
from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from src.tools.distributions import DistributionProcessor
from src.tools.scaling import scale_zero_one


def detect_anomaly_lof(
    data: Union[np.ndarray, pd.Series, List, dd.Series], with_confidence=True
) -> np.ndarray:
    """
    Detects anomalies in a dataset using the Local Outlier Factor (LOF) algorithm.
    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, List, dd.Series]
        The input data to analyze for anomalies. Must be a 1-dimensional array-like structure.
    with_confidence : bool, optional (default=True)
        If True, returns anomaly scores scaled between 0 and 1, indicating the degree of outlierness.
        If False, returns a binary array where 1 indicates an anomaly and 0 indicates a normal point.
    Returns
    -------
    np.ndarray
        An array of anomaly scores (if with_confidence=True) or binary anomaly labels (if with_confidence=False).
    Raises
    ------
    TypeError
        If the input data is not a numpy array, pandas Series, list, or dask Series.
    Notes
    -----
    - The function standardizes the input data and handles NaN values by replacing them with the mean.
    - If the data has insufficient variance or too few samples, it returns an array of zeros (no anomalies).
    - Uses a dynamic number of neighbors for LOF based on the input data size.
    """
    if not isinstance(data, (np.ndarray, pd.Series, List, dd.Series)):
        raise TypeError(
            "Input data must be a numpy array, pandas Series, list, or dask Series."
        )

    if len(data) < 5:
        return np.zeros(len(data))

    # Convert and standardize data
    data = DistributionProcessor.standardize(data).reshape(-1, 1)

    # Handle NaN values
    non_null = data[~np.isnan(data)]
    mean_val = np.mean(non_null) if len(non_null) > 0 else 0
    data[np.isnan(data)] = mean_val

    # Check for variance threshold
    std = np.std(non_null, ddof=0, dtype=np.float64)

    # check if mean is too small to calculate co-efficient of variance
    if abs(mean_val) < 0.1 * std:
        return np.zeros(len(data))

    # calculate co-efficent of variance
    # This is done to ensure that low variance values do not throw unwanted anomalies.
    cv = std / mean_val
    if cv <= 0.5:
        return np.zeros(len(data))

    # Calculate LOF - use reasonable number of neighbors
    n_neighbors = min(20, max(2, len(data) // 2))  # Better neighbor selection
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.01)

    if not with_confidence:
        # Binary prediction (-1=outlier, 1=inlier)
        outliers = lof.fit_predict(data)
        # Convert to 0 (inlier) 1 (outlier)
        return (outliers == -1).astype(float)

    # Confidence scores (higher = more outlier)
    lof.fit_predict(data)
    scores = -lof.negative_outlier_factor_
    return scale_zero_one(scores)
