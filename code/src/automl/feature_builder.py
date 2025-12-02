# # Copyright (C) KonaAI - All Rights Reserved
"""This module handles the feature building process for the AutoML pipeline."""
import sys
from typing import Dict

import dask.dataframe as dd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from src.automl.datetime_encoder import DateTimeEncoder
from src.automl.feature_encoder import CustomFeatureEncoder
from src.automl.numeric_encoder import NumericEncoder
from src.automl.onehot_encoder import CustomOneHotEncoder
from src.automl.preprocess import PreProcess
from src.tools.dask_tools import compute
from src.utils.status import Status
from src.utils.submodule import Submodule
from tqdm import tqdm


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    FeatureBuilder is a custom transformer that preprocesses and encodes features for machine learning models.
    Attributes:
    ---------
        prefix_separator (str): Separator used in prefixing encoded features.
    Methods:
    -------
        __init__(submodule: Submodule, preprocessor: PreProcess) -> None:
            Initializes the FeatureBuilder with a submodule and a preprocessor.
        _label_transform(X: dd.DataFrame) -> dd.DataFrame:
        fit(X: dd.DataFrame, y=None) -> "FeatureBuilder":
            Fits the encoder to the data.
        transform(X: dd.DataFrame) -> dd.DataFrame:
            Transforms the data using the fitted encoders.
    """

    prefix_separator = "="

    def __init__(
        self,
        submodule: Submodule,
        preprocessor: PreProcess,
        encoder: CustomFeatureEncoder = None,
    ) -> None:
        self.submodule = submodule
        self.preprocessor = preprocessor or PreProcess(submodule=submodule)
        self.encoders = encoder or CustomFeatureEncoder()

    def fit(
        self, X: dd.DataFrame, y=None  # pylint: disable=unused-argument
    ) -> "FeatureBuilder":
        """
        Fits the FeatureBuilder by encoding numeric and categorical columns in the input Dask DataFrame.
        This method performs the following steps:
        1. Encodes numeric columns using a NumericEncoder, if present.
        2. Identifies rare labels for categories derived from numeric columns.
        3. Encodes categorical columns (including those derived from numeric columns) using a CustomOneHotEncoder, if present.
        """
        # Encode the numeric columns
        ne = None
        if numeric_columns := self.preprocessor.numeric_columns:
            ne = NumericEncoder(
                columns=numeric_columns,
                unknown_replacement=self.preprocessor.unknown_category,
            )
            X = ne.fit_transform(X)
            self.encoders.numeric_encoder = ne

            # identify rare labels for numeric derived categories
            self.preprocessor.identify_rare_labels(
                X, y, self.encoders.numeric_encoder.derived_categories
            )

        # check what datetime columns are present in the data
        de = None
        if datetime_columns := self.preprocessor.datetime_columns:
            de = DateTimeEncoder(
                columns=datetime_columns,
                unknown_replacement=self.preprocessor.unknown_category,
            )
            X = de.fit_transform(X)
            self.encoders.datetime_encoder = de

            # Important:
            # not removing date time rare label because we want to keep them for better learning

        # check what category columns are present in the data
        # if there are any, encode them
        category_columns = self.preprocessor.categorical_columns
        category_columns = (
            category_columns + de.derived_categories + ne.derived_categories
        )

        # encode the category columns
        if category_columns := ([col for col in category_columns if col in X.columns]):
            ohe = CustomOneHotEncoder(
                columns=category_columns,
                unknown_replacement=self.preprocessor.unknown_category,
                prefix_sep=self.prefix_separator,
            ).fit(X, y)
            self.encoders.onehot_encoder = ohe

        return self

    def transform(self, X: dd.DataFrame) -> dd.DataFrame:
        """
        Transforms the input Dask DataFrame `X` using the appropriate encoding strategy based on the type of encoder(s) present.
        The transformation process includes:
        - Label encoding if `self.encoders` is a dictionary (backward compatibility).
        - One-hot encoding if `self.encoders` is a `CustomOneHotEncoder` (backward compatibility).
        - Numeric and one-hot encoding if `self.encoders` is a `CustomFeatureEncoder`, including imputation of rare labels and removal of columns with unknown categories.
        - Removal of columns containing unknown categories after encoding.
        - Selection of only the available features as per the encoder's output feature names.
        Args:
        ----
            X (dd.DataFrame): Input Dask DataFrame to be transformed.
        Returns:
            dd.DataFrame: Transformed Dask DataFrame with encoded features.
        """

        # when only label encoder is present
        if isinstance(self.encoders, Dict):
            # This is for backward compatability, since we had label encoders in previous versions
            X = self._label_transform(X)
        # when only onehot encoder is present
        elif isinstance(self.encoders, CustomOneHotEncoder):
            # This is for backward compatability, since we had onehot encoders in previous versions
            # encode the category columns
            X = self.encoders.transform(X)

            if unknown_columns := [
                col
                for col in tqdm(
                    X.columns, desc="Removing missing data columns", file=sys.stdout
                )
                if self.preprocessor.unknown_category in col
            ]:
                X = compute(X.drop(unknown_columns, axis=1))
        # when both encoders are present
        elif isinstance(self.encoders, CustomFeatureEncoder):
            # encode the numeric columns
            if self.encoders.numeric_encoder is not None:
                X = self.encoders.numeric_encoder.transform(X)
                X = self.preprocessor.impute_rare_labels(X)

            # encode the datetime columns
            # Checking for datetime encoder for backward compatibility
            if (
                hasattr(self.encoders, "datetime_encoder")
                and self.encoders.datetime_encoder is not None
            ):
                X = self.encoders.datetime_encoder.transform(X)

            # encode the category columns
            if self.encoders.onehot_encoder is not None:
                X = self.encoders.onehot_encoder.transform(X)

            # remove the columns with unknown category
            if unknown_columns := [
                col
                for col in tqdm(
                    X.columns, desc="Removing missing data columns", file=sys.stdout
                )
                if self.preprocessor.unknown_category in col
            ]:
                X = compute(X.drop(unknown_columns, axis=1))

            # keep only the columns that are present in the feature names
            available_features = [
                feature
                for feature in self.encoders.get_feature_names_out()
                if feature in X.columns
            ]
            X = X[available_features]

        Status.INFO(f"Total {len(X.columns)} features ready.", self.submodule)
        return compute(X)
