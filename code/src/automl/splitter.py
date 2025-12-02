# # Copyright (C) KonaAI - All Rights Reserved
""" Module to split the data into train, test and validation sets. """ ""
from collections import Counter
from typing import Generator
from typing import Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import StratifiedShuffleSplit
from src.automl.ml_params import MLParameters
from src.utils.status import Status

params = MLParameters()


def split_train_test(  # pylint: disable=too-many-positional-arguments
    X: dd.DataFrame,
    y: dd.Series,
    test_size: int = params.test_size,
    n_splits: int = params.n_splits,
    synthentic_data: bool = params.need_synthetic,
    random_state: int = 42,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None, None]:
    """
    Splits the given data into train and test sets using stratified shuffle split.
    Parameters:
    ----------
    X (dd.DataFrame): The input features as a Dask DataFrame.
    y (dd.Series): The target variable as a Dask Series.
    test_size (int, optional): The percentage of data to be used as the test set. Default is params.test_size.
    n_splits (int, optional): The number of re-shuffling & splitting iterations. Default is params.n_splits.
    synthentic_data (bool, optional): Whether to generate synthetic data. Default is params.need_synthetic.
    random_state (int, optional): The seed used by the random number generator. Default is 42.

    Returns:
    Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], None, None]:
    A generator yielding tuples containing the train-test split of X and y.
    """

    # convert to pandas dataframe
    X = X.compute().reset_index(drop=True)
    y = y.compute().reset_index(drop=True)

    if synthentic_data:
        Status.INFO("Synthetic data generation is enabled")
        X, y = generate_synthetic_data(X, y)

    Status.INFO(
        "Splitting the data into train and test sets",
        test_size=f"{test_size}%",
        n_splits=n_splits,
    )

    # Stratified Train-Test Split
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size / 100, random_state=random_state
    )
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        yield X_train, X_test, y_train, y_test


def generate_synthetic_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic data to address class imbalance in the dataset.
    This function checks the class imbalance in the target variable `y` and applies
    different resampling techniques based on the level of imbalance:
    - If the imbalance is between 1% and 5%, it oversamples the minority class using SMOTE.
    - If the imbalance is less than 1%, it oversamples the minority class using SMOTE and
        undersamples the majority class using RandomUnderSampler.
    - If the imbalance is greater than or equal to 5%, no resampling is performed.

    Parameters:
    X (pd.DataFrame): The feature matrix.
    y (pd.Series): The target variable.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: The resampled feature matrix and target variable.
    """
    random_state = 42

    # find out minority and majority class
    counter = Counter(y)
    minority_class = min(counter, key=counter.get)
    majority_class = max(counter, key=counter.get)

    # adjust k_neighbors based on the number of samples in the minority class
    k_neighbors = min(5, counter[minority_class] - 1)

    # check class imbalance
    imbalance = y.sum() / len(y)
    Status.INFO(
        f"Imbalance in the split training dataset: {imbalance}",
        concern=y.sum(),
        no_concern=(len(y) - y.sum()),
    )

    if imbalance > 0.05:
        Status.INFO("No need to generate synthetic data")
        return X, y

    r = 0.05  # minimum desired ratio of minority class over majority class
    sampling_strategy = {
        minority_class: np.ceil(r * counter[majority_class]).astype(int),
        majority_class: counter[majority_class],
    }

    # over sample the minority class
    sm = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
    )

    # if imbalance is between 1% and 5%
    if 0.01 < imbalance <= 0.05:
        Status.INFO("Over sampling the minority class")

        # over sample the minority class
        X_train, y_train = sm.fit_resample(X, y)

        return X_train, y_train

    # if imbalance is less than 1%
    Status.INFO(
        "Over sampling the minority class and under sampling the majority class"
    )

    # Define SMOTE-ENN
    resample = SMOTEENN(
        enn=EditedNearestNeighbours(sampling_strategy="all"),
        smote=sm,
        random_state=random_state,
        sampling_strategy="auto",
    )

    # resample the data
    X_train, y_train = resample.fit_resample(X, y)

    return X_train, y_train
