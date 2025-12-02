# # Copyright (C) KonaAI - All Rights Reserved
import dask.dataframe as dd
import pandas as pd
import pytest
from src.automl.ml_params import MLParameters
from src.automl.splitter import split_train_test


@pytest.fixture
def sample_data():
    X = pd.DataFrame({"feature1": range(100), "feature2": range(100, 200)})
    y = pd.Series([0] * 90 + [1] * 10)
    return dd.from_pandas(X, npartitions=1), dd.from_pandas(y, npartitions=1)


def test_split_train_test(sample_data):
    X, y = sample_data
    params = MLParameters()
    params.test_size = 20
    params.n_splits = 1
    params.need_synthetic = False

    splits = list(
        split_train_test(
            X,
            y,
            test_size=params.test_size,
            n_splits=params.n_splits,
            synthentic_data=params.need_synthetic,
        )
    )
    assert len(splits) == 1

    X_train, X_test, y_train, y_test = splits[0]
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_train_test_with_synthetic_data(sample_data):
    X, y = sample_data
    params = MLParameters()
    params.test_size = 20
    params.n_splits = 1
    params.need_synthetic = True

    # Scenario 1: Class imbalance greater than 0.05
    y_imbalanced = [0] * 90 + [1] * 10
    y_imbalanced = dd.from_pandas(pd.Series(y_imbalanced), npartitions=X.npartitions)
    imbalance = y.sum().compute() / len(y)
    assert imbalance > 0.05
    splits = list(
        split_train_test(
            X,
            y_imbalanced,
            test_size=params.test_size,
            n_splits=params.n_splits,
            synthentic_data=params.need_synthetic,
        )
    )
    assert len(splits) == 1
    X_train, X_test, y_train, y_test = splits[0]
    assert len(X_train) == 80  # synthetic data increases the size
    assert len(y_train) == 80
    assert len(X_test) == 20
    assert len(y_test) == 20
    assert len(y_train[y_train == 1]) == 8
    assert len(y_test[y_test == 1]) == 2

    # Scenario 2: Class imbalance greater than 0.01 and less than 0.05
    X = dd.concat([X, X], axis=0)  # double it to make it 200
    y_imbalanced = [0] * 193 + [1] * 7
    y_imbalanced = dd.from_pandas(pd.Series(y_imbalanced), npartitions=X.npartitions)
    imbalance = y_imbalanced.sum().compute() / len(y_imbalanced)
    assert 0.01 < imbalance < 0.05
    splits = list(
        split_train_test(
            X,
            y_imbalanced,
            test_size=params.test_size,
            n_splits=params.n_splits,
            synthentic_data=params.need_synthetic,
        )
    )
    assert len(splits) == 1
    X_train, X_test, y_train, y_test = splits[0]
    assert len(X_train) == 162
    assert len(y_train) == 162
    assert len(X_test) == 41
    assert len(y_test) == 41
    assert len(y_train[y_train == 1]) == 8
    assert len(y_test[y_test == 1]) == 2

    # Scenario 3: Class imbalance less than 0.01
    X = dd.from_pandas(
        pd.DataFrame({"feature1": range(700), "feature2": range(100, 800)})
    )
    y_imbalanced = [0] * 695 + [1] * 5
    y_imbalanced = dd.from_pandas(pd.Series(y_imbalanced), npartitions=X.npartitions)
    imbalance = float(y_imbalanced.sum().compute() / len(y_imbalanced))
    assert imbalance < 0.01
    splits = list(
        split_train_test(
            X,
            y_imbalanced,
            test_size=params.test_size,
            n_splits=params.n_splits,
            synthentic_data=params.need_synthetic,
        )
    )
    assert len(splits) == 1
    X_train, X_test, y_train, y_test = splits[0]
    assert len(X_train) == 583  # synthetic data increases the size
    assert len(y_train) == 583
    assert len(X_test) == 146
    assert len(y_test) == 146
    assert len(y_train[y_train == 1]) == 28
    assert len(y_test[y_test == 1]) == 7
