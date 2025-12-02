# # Copyright (C) KonaAI - All Rights Reserved
"""This module generates core features for the P2P dataset"""
import secrets

import numpy as np
import pandas as pd


def generate_core_features(records):
    """
    Generate synthetic core features for P2P dataset modeling.

    This function creates:
        - A binary target variable.
        - Boolean informative features (True/False).
        - Redundant features: numerical features correlated with the target.
        - Repeated features: exact copies of some informative features.

    Boolean columns are converted to string representation to mimic categorical input.

    Args:
    -----
        records (int): Number of records (rows) to generate.

    Returns:
        pandas.DataFrame: A DataFrame with a 'target' column and multiple engineered feature columns.
    """

    n_informative = 30
    n_redundant = 20
    n_repeated = 10

    target = np.random.choice([0, 1], size=records, p=[0.5, 0.5])

    X = pd.DataFrame(target, columns=["target"])

    for i in range(n_informative):
        # create a features with true and false values
        X[f"P2PFM{i:03d}"] = np.random.choice([True, False], size=records, p=[0.5, 0.5])

    for i in range(n_redundant):
        # create a redundant feature that is like target with some noise
        X[f"P2PFM{i+n_informative:03d}"] = target + 0.1

    for i in range(n_repeated):
        # create a repeated feature that is like target with some noise
        X[f"P2PFM{i+n_informative+n_redundant:03d}"] = X[f"P2PFM{i:03d}"]

    # convert boolean columns to string
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype("string[pyarrow]")

    # find all P2PFM columns
    p2pfm_cols = [col for col in X.columns if col.startswith("P2PFM")]
    # Add _Tran_Score suffixed columns
    for col in p2pfm_cols:
        col_name = f"{col}_Tran_Score"
        X[col_name] = secrets.randbelow(100)
        X[col_name] = X[col_name].astype(int)

    return X
