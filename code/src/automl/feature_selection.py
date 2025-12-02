# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the feature selection transformer"""
import sys

import dask.dataframe as dd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from src.automl.feature_builder import FeatureBuilder
from src.automl.ml_params import MLParameters
from src.automl.woe_calculator import WoECalculator
from src.tools.dask_tools import compute
from src.utils.conf import Setup
from src.utils.status import Status
from src.utils.submodule import Submodule
from tqdm import tqdm


params = MLParameters()


class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    FeatureSelection is a scikit-learn compatible transformer for automated feature selection in machine learning pipelines.
    This class implements several feature selection strategies, including:
    - Removal of features with no impact (e.g., features that do not contribute to the target class).
    - Removal of constant features (features with the same value across all samples).
    - Removal of no-impact test patterns (e.g., false values in one-hot encoded test patterns).
    - Filtering features based on mutual information using a WoECalculator.

    Attributes
    ----------
    correlation_threshold : float
        The threshold for feature correlation, derived from submodule parameters.
    _features_to_keep : list of str
        List of feature names to retain after selection.
    _features_to_drop : list of str
        List of feature names to drop after selection.
    """

    def __init__(
        self,
        submodule: Submodule,
        feature_builder: FeatureBuilder,
    ):
        self.submodule = submodule
        self.correlation_threshold = submodule.ml_params.correlation_threshold / 100
        self._feature_builder = feature_builder
        self._features_to_keep = []
        self._features_to_drop = []

    def get_features_to_keep(self):
        """
        Returns a sorted list of feature names to keep, excluding the feature named "target".

        Returns:
            List[str]: A sorted list of feature names (as strings) to retain, with "target" removed if present.
        """
        return sorted(
            list(
                {feature for feature in self._features_to_keep if feature != "target"}
            ),
            key=str,
        )

    def get_features_to_drop(self):
        """
        Returns a sorted list of features that are marked to be dropped.

        The features are deduplicated and sorted alphabetically as strings.

        Returns:
            List[str]: A sorted list of feature names to drop.
        """
        return sorted(list(set(self._features_to_drop)), key=str)

    def _remove_no_impact_features(self, X: dd.DataFrame, y: dd.Series) -> dd.DataFrame:
        # objective is to remove those features that do not have any value of 1 for y = 1
        X_copy = X.assign(target=y).copy()
        X_copy = X_copy[X_copy["target"] == 1]
        X_copy = compute(X_copy.drop("target", axis=1))

        # now remove features that do not have any value of 1
        no_impact_features = []
        no_impact_features.extend(
            feature
            for feature in tqdm(
                X_copy.columns, desc="Evaluating no impact features", file=sys.stdout
            )
            if X_copy[feature].sum().compute() == 0
        )

        if no_impact_features:
            Status.INFO(
                f"Removing {len(no_impact_features)} no impact features",
                self.submodule,
                dropped_features=no_impact_features,
            )
            return compute(X.drop(no_impact_features, axis=1))
        return compute(X)

    def _remove_no_impact_test_patterns(self, X: dd.DataFrame) -> dd.DataFrame:
        """
        Removes the no impact test patterns from the input DataFrame.
        This is used when the test patterns are One-Hot Encoded.
        False values of the test patterns are removed to avoid false impact on the model.
        Parameters:
            X (dd.DataFrame): The input DataFrame.
        Returns:
            dd.DataFrame: The DataFrame with the no impact test patterns removed.
        """
        Status.INFO("Evaluating no impact test patterns", self.submodule)
        pattern_df = self.submodule.get_test_patterns()
        if pattern_df is None:
            Status.INFO(
                "No test patterns found, skipping no impact test patterns removal",
                self.submodule,
            )
            return X

        # get the pattern id column
        pattern_id_col = (
            Setup().global_constants.get("TEST_PATTERNS", {}).get("PATTERN_ID_COLUMN")
        )
        test_pattern_cols = pattern_df[pattern_id_col].unique().tolist()
        no_impact_patterns = [
            f"{col}{self._feature_builder.prefix_separator}false".lower()
            for col in test_pattern_cols
        ]

        if no_impact_features := [
            col
            for col in tqdm(
                X.columns, desc="Evaluating no impact test patterns", file=sys.stdout
            )
            if col.lower() in no_impact_patterns
        ]:
            Status.INFO(
                f"Removing {len(no_impact_features)} no impact test patterns",
                self.submodule,
                dropped_features=no_impact_features,
            )

            return compute(X.drop(no_impact_features, axis=1))
        return X

    def _remove_constant(self, X):
        """
        Remove features with constant value from the input data.
        Parameters:
        - X: pandas DataFrame or Dask DataFrame
            The input data.
        Returns:
        - List[str]
            A list of column names that have constant values.
        """
        Status.INFO(
            "Removing features with constant value",
            self.submodule,
        )

        # Compute unique values for each column
        unique_counts = X.nunique().compute()

        # Identify constant columns
        constant_columns = unique_counts.index[unique_counts == 1]
        if not constant_columns.size:
            Status.INFO(
                "No constant features found",
                self.submodule,
            )
            return X

        Status.INFO(
            f"Found {len(constant_columns)} constant features",
            self.submodule,
            dropped_features=constant_columns.tolist(),
        )

        return compute(X.drop(columns=constant_columns, axis=1))

    def fit(self, X, y=None):
        """
        Fits the feature selection transformer to the provided dataset.

        This method applies a sequence of feature selection steps to the input DataFrame `X`:
        1. Removes test patterns with no impact.
        2. Removes features with no impact based on the target `y`.
        3. Removes constant features.
        4. Filters features using mutual information via a WoECalculator.
        Updates internal lists of features to keep and drop based on the filtering process.
        Args:
        ---
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series or None, optional): The target variable for supervised feature selection. Default is None.
        Returns:
            self: Returns the fitted transformer instance.
        """
        Status.INFO(
            "Fitting feature selection transformer",
            self.submodule,
            total_features=len(X.columns),
        )
        X = self._remove_no_impact_test_patterns(X)
        X = self._remove_no_impact_features(X, y)
        X = self._remove_constant(X)

        # filter by mutual information
        mi_filter = WoECalculator(self.submodule)
        mi_filter.fit(X, y)

        # update features to keep
        self._features_to_keep = [
            f for f in X.columns if f in mi_filter.get_features_to_keep()
        ]

        # update features to drop
        self._features_to_drop = [
            f for f in X.columns if f not in self._features_to_keep
        ]

        Status.INFO(
            f"Finalized {len(self._features_to_keep)} features",
            self.submodule,
            features=X.columns.tolist(),
        )

        return self

    def transform(self, X):
        """
        Transforms the input data by selecting and retaining only the features determined to be kept.

        Parameters:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data containing only the selected features.

        Logs:
            An informational message indicating the number of features being retained.
        """
        Status.INFO(
            f"Preparing data for {len(self.get_features_to_keep())} features",
            self.submodule,
        )

        # Drop the selected features from the training data
        return compute(X[self.get_features_to_keep()])
