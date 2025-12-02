# # Copyright (C) KonaAI - All Rights Reserved
"""Weight of Evidence Calculator"""
import numpy as np
import pandas as pd
from dask import dataframe as dd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from src.tools.dask_tools import compute
from src.tools.dask_tools import discretize
from src.utils.status import Status
from src.utils.submodule import Submodule
from tqdm import tqdm


class WoECalculator(BaseEstimator, TransformerMixin):
    """
    WoECalculator is a custom transformer that calculates the Weight of Evidence (WoE) for features in a dataset
    and filters features based on mutual information with the target variable.
    Attributes:
    ----------
        woe_suffix (str): Suffix to append to feature names after WoE transformation.
        submodule (Submodule): Submodule containing machine learning parameters.
        correlation_threshold (float): Threshold for filtering features based on mutual information.
        _features_to_keep (list): List of features to keep after filtering.
        _features_to_drop (list): List of features to drop after filtering.
    """

    woe_suffix = "_WoE"

    def __init__(
        self,
        submodule: Submodule,
    ):
        self.submodule = submodule
        self.correlation_threshold = submodule.ml_params.correlation_threshold / 100
        self._features_to_keep = []
        self._features_to_drop = []

    def get_features_to_keep(self):
        """
        Returns a sorted list of feature names to keep, excluding the 'target' feature.

        This method processes the internal set of features marked to be kept, removes the 'target' feature if present,
        and returns the remaining feature names sorted as strings.

        Returns:
            List[str]: A sorted list of feature names to keep, excluding 'target'.
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

        Returns:
            List[str]: A sorted list of feature names to drop.
        """
        return sorted(list(set(self._features_to_drop)), key=str)

    def calculate_woe(self, df: dd.DataFrame, feature_name: str, target_name: str):
        """
        Calculate the Weight of Evidence (WoE) for a given feature in a Dask DataFrame.
        Parameters:
        -----------
        df : dd.DataFrame
            The input Dask DataFrame containing the feature and target columns.
        feature_name : str
            The name of the feature column for which WoE is to be calculated.
        target_name : str
            The name of the target column.
        Returns:
        --------
        dd.DataFrame
            The DataFrame with an additional column containing the WoE values for the specified feature.
        Notes:
        ------
        - The function discretizes the feature into bins, calculates the proportion of target=1 and target=0 in each bin,
          and then computes the WoE for each bin.
        - An epsilon value is added to the proportions to avoid division by zero.
        - The resulting WoE values are added as a new column to the DataFrame.
        """
        # convert to bins
        bin_col_name = f"{feature_name}_bins"
        df[bin_col_name] = discretize(df[feature_name])

        # Group by bins and calculate statistics
        grouped = df.groupby(bin_col_name)[target_name].agg(["count", "sum"])
        grouped.columns = [
            "total",
            "concern",
        ]  # "concern" refers to the count of target=1
        grouped["no_concern"] = grouped["total"] - grouped["concern"]
        grouped = grouped.compute()

        # Calculate proportions
        grouped["concern_pct"] = grouped["concern"] / grouped["concern"].sum()
        grouped["no_concern_pct"] = grouped["no_concern"] / grouped["no_concern"].sum()

        # Compute WoE (adding epsilon to avoid division by zero)
        epsilon = 1e-6
        grouped[self.woe_suffix] = np.log(
            (grouped["no_concern_pct"] + epsilon) / (grouped["concern_pct"] + epsilon)
        )

        woe_col_name = f"{feature_name}{self.woe_suffix}"
        df[woe_col_name] = df[bin_col_name].map(
            grouped[self.woe_suffix], meta=("float64")
        )
        return compute(df.drop(columns=[bin_col_name]))

    def filter_by_mutual_info(self, df: dd.DataFrame, target_name: str):
        """
        Filters features in the dataframe by calculating their mutual information with each other and the target variable.
        Features that are highly correlated with each other are identified, and only the feature with the highest mutual
        information with the target variable is kept.

        Parameters:
        df (dd.DataFrame): The input dataframe containing features to be filtered.
        target_name (str): The name of the target variable in the dataframe.

        Returns:
        list: A list of features to be dropped from the dataframe.
        """
        Status.INFO(
            "Filtering features by Mutual Information. This may take a while...",
            threshold=f"{self.correlation_threshold*100}%",
        )
        features_to_drop = [target_name]
        mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

        for idx, feature_1 in enumerate(
            tqdm(df.columns, desc="Calculating Mutual Information")
        ):
            # skip if feature is already dropped or target
            if feature_1 in features_to_drop:
                continue
            remaining_cols = [
                col for col in df.columns[idx:] if col not in features_to_drop
            ]
            for feature_2 in tqdm(remaining_cols, leave=False):
                # skip if feature is already dropped or target
                if feature_2 in features_to_drop:
                    continue

                # do not exclude same feature comparison
                # because we will use self mutual information to normalize
                # since mutual information is not bound between 0 and 1
                mi_matrix.loc[feature_1, feature_2] = mutual_info_regression(
                    df[feature_1].values.compute().reshape(-1, 1),
                    df[feature_2].values.compute(),
                )[0]

            # identify highly correlated features
            row = mi_matrix.loc[feature_1]

            # skip if row is all null
            if row.isnull().all():
                continue

            # normalize row
            epsilon = 1e-6
            row = (row + epsilon) / (
                row.max() + epsilon
            )  # Avoid division by zero by adding epsilon
            # find highly correlated features
            high_corr = row[row > self.correlation_threshold]

            # if no highly correlated features, continue
            if len(high_corr) == 0:
                continue

            # if there are highly correlated features, display them and break
            correlated_features = high_corr.index.tolist()
            target_mi = {
                feature: mutual_info_regression(
                    df[feature].values.compute().reshape(-1, 1),
                    df[target_name].values.compute(),
                )[0]
                for feature in correlated_features
            }
            # find feature with highest mutual information with target
            feature_to_keep = max(target_mi, key=target_mi.get)
            # drop all other features
            features_to_drop.extend(
                [
                    feature
                    for feature in correlated_features
                    if feature != feature_to_keep
                ]
            )
            correlated_features.remove(feature_to_keep)
            if len(correlated_features) > 0:
                Status.INFO(
                    f"Found {len(correlated_features)} correlated features",
                    keeping=self._clean_feature_name(feature_to_keep),
                    dropped=[
                        self._clean_feature_name(feature)
                        for feature in correlated_features
                    ],
                )

        return features_to_drop

    def _clean_feature_name(self, feature_name: str):
        """
        Cleans the feature name by removing the Weight of Evidence (WoE) suffix if it exists.

        Args:
            feature_name (str): The name of the feature to be cleaned.

        Returns:
            str: The cleaned feature name without the WoE suffix.
        """
        if feature_name.endswith(self.woe_suffix):
            return feature_name[: -len(self.woe_suffix)]
        return feature_name

    def fit(self, X: dd.DataFrame, y: dd.Series):
        """
        Fits the WoECalculator to the provided data by calculating the Weight of Evidence (WoE) for each feature,
        filtering features based on mutual information, and determining which features to keep or drop.
        Parameters
        ----------
        X : dd.DataFrame
            The input features as a Dask DataFrame.
        y : dd.Series
            The target variable as a Dask Series.
        Returns
        -------
        self : WoECalculator
            Returns the fitted instance of the WoECalculator.
        """
        # calculate weight of evidence for all features
        incoming_features = X.columns.tolist()
        Status.INFO(
            f"Calculating Weight of Evidence for {len(incoming_features)} features"
        )
        X["target"] = y
        pbar = tqdm(total=len(incoming_features), desc="Calculating WoE")
        for feature in X.columns:
            if feature == "target":
                continue
            pbar.set_postfix_str(f"Calculating WoE for {feature}")
            X = self.calculate_woe(X, feature, "target")
            pbar.update()
        pbar.close()
        Status.INFO(
            f"Finished calculating Weight of Evidence for {len(incoming_features)} features"
        )

        # filter features by mutual information
        X_WoE: dd.DataFrame = X[
            [col for col in X.columns if col.endswith(self.woe_suffix)] + ["target"]
        ]
        woe_features_to_drop = self.filter_by_mutual_info(X_WoE, "target")
        woe_features_to_keep = [
            feature for feature in X_WoE.columns if feature not in woe_features_to_drop
        ]

        # remove suffix from feature names
        self._features_to_keep = [
            self._clean_feature_name(feature) for feature in woe_features_to_keep
        ]
        self._features_to_drop = [
            self._clean_feature_name(feature) for feature in woe_features_to_drop
        ]

        # remove target from features to keep
        if "target" in self._features_to_keep:
            self._features_to_keep.remove("target")

        # remove target from features to drop
        if "target" in self._features_to_drop:
            self._features_to_drop.remove("target")

        # remove target from X
        X = X.drop(columns=["target"])

        return self

    def transform(self, X: dd.DataFrame):
        """Transform the data by keeping only the features that were selected during fit."""
        return compute(
            X[
                [
                    col
                    for col in X.columns
                    if self._clean_feature_name(col) in self._features_to_keep
                ]
            ]
        )
