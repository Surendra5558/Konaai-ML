# # Copyright (C) KonaAI - All Rights Reserved
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from src.automl.onehot_encoder import CustomOneHotEncoder


@pytest.fixture(scope="module")
def sample_data():
    # Create a sample Dask DataFrame for testing
    data = {
        "col1": ["A", "B", "C", "D", "E", np.nan, "C"],
        "col2": ["X", "Y", "Z", "X", "Y", np.nan, "Z"],
        "col3": [True, True, False, False, True, False, True],
    }
    X = dd.from_pandas(pd.DataFrame(data), npartitions=2)

    # create a y variable with values 0 and 1
    y = np.random.randint(0, 2, len(data["col1"]))
    # create dask series from y
    y = dd.from_pandas(pd.Series(y), npartitions=2)

    return X, y


@pytest.fixture(scope="module")
def encoder(sample_data):
    X, y = sample_data
    ohe = CustomOneHotEncoder(columns=["col1", "col2"], unknown_replacement="missing")
    return ohe.fit(X, y)


def test_encode(encoder):
    assert isinstance(encoder.categories_, dict), "categories should be a dictionary"
    assert encoder.feature_names_ == [
        "col1_a",
        "col1_b",
        "col1_c",
        "col1_d",
        "col1_e",
        "col1_missing",
        "col2_missing",
        "col2_x",
        "col2_y",
        "col2_z",
    ], "feature names should be equal"
    assert encoder.categories_["col2"] == [
        "missing",
        "x",
        "y",
        "z",
    ], "categories should be equal"


def test_transform(sample_data, encoder):
    X, y = sample_data
    transformed_data = encoder.transform(X)
    assert transformed_data.shape[1] == 9, "transformed data should have 11 columns"
    assert transformed_data["col2_x"].compute().tolist() == [
        1,
        0,
        0,
        1,
        0,
        0,
        0,
    ], "transformed data should be equal"
