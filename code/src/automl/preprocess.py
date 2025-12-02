# # Copyright (C) KonaAI - All Rights Reserved
"""
This file contains the preprocessing steps for the automl pipeline
"""
import sys
from typing import Any
from typing import Dict
from typing import List

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from src.automl.feature_encoder import CustomFeatureEncoder
from src.automl.ml_params import MLParameters
from src.automl.utils import config
from src.tools.dask_tools import compute
from src.tools.dask_tools import is_boolean
from src.tools.dask_tools import is_datetime
from src.tools.dask_tools import is_string
from src.utils.conf import Setup
from src.utils.status import Status
from src.utils.submodule import Submodule
from tqdm import tqdm

params = MLParameters()


class PreProcess(BaseEstimator, TransformerMixin):
    """
    PreProcess is a scikit-learn compatible transformer class designed to preprocess data for the automl submodule.
    It provides a comprehensive suite of data cleaning and transformation utilities tailored for large-scale, distributed data processing using Dask DataFrames.
    Key Features:
    -------------
    - Missing Value Handling: Summarizes, drops, and imputes missing values based on configurable thresholds.
    - High Cardinality Detection: Identifies and removes columns with excessive unique values or low-frequency patterns.
    - Large Text Column Removal: Detects and drops columns containing large text blobs.
    - Categorical and Numeric Feature Identification: Automatically detects and manages categorical and numeric columns.
    - Rare Label Handling: Identifies and imputes rare categorical labels to reduce model bias.
    - Constant Imputation: Fills missing values in specified columns with a constant value.
    - KNN Imputation (planned): Placeholder for KNN-based imputation for missing values.
    - Feature Reduction: Drops unnecessary columns based on configurable patterns and retains only required features.
    - Test Pattern Management: Ensures test pattern columns are preserved during preprocessing.
    - Alignment: Aligns feature and target data to ensure consistency.
    - Scikit-learn API: Implements fit and transform methods for pipeline compatibility.
    """

    unknown_category = "<||unknown||>"

    _y = None

    def __init__(
        self,
        submodule: Submodule,
        encoder: CustomFeatureEncoder = None,
    ):
        self.submodule = submodule
        self.missing_threshold = submodule.ml_params.missing_threshold
        self.categorical_columns: List[str] = []
        self.datetime_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.missing_summary: pd.DataFrame = None
        self.encoder = encoder
        self.rare_labels: Dict = {}

    def get_missing_summary(self):
        """
        Returns a summary of missing values in the dataset.

        Returns:
            pd.DataFrame or dict: A summary of missing values, typically showing the count or percentage of missing entries per column.
        """
        return self.missing_summary

    def missing_values_summary(self, X: dd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of missing values in a DataFrame.
        Parameters:
            X (dd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing the count and percentage of missing values for each column.
        """

        Status.INFO(
            "Building missing values summary.",
            self.submodule,
        )

        missing_values = pd.DataFrame(
            X.isna().sum().compute(), columns=["missing_count"]
        )

        missing_values["missing_percent"] = missing_values["missing_count"] / len(X)
        missing_values["column_name"] = missing_values.index
        missing_values = missing_values[
            missing_values["missing_count"] > 0
        ].sort_values(by="missing_count", ascending=False)

        return missing_values

    def drop_missing(self, X: dd.DataFrame, exclude: List[str] = None):
        """
        Drops columns from the input DataFrame `X` that have missing values greater than the specified threshold.
        Parameters:
        - X (dd.DataFrame): The input DataFrame.
        - exclude (List[str]): A list of column names to exclude from dropping.
        Returns:
        - dd.DataFrame: The modified DataFrame with dropped columns.
        """
        if exclude is None:
            exclude = []

        if self.missing_summary is not None or len(self.missing_summary.index) > 0:
            missing_values = self.missing_summary
        else:
            missing_values = self.missing_values_summary(X)

        # filter columns with missing values greater than threshold
        missing_values = missing_values[
            missing_values["missing_percent"] > self.missing_threshold / 100
        ]
        if len(missing_values) > 0:
            cols_to_drop = [col for col in missing_values.index if col not in exclude]
            X = X.drop(cols_to_drop, axis=1)

            Status.INFO(
                f"Dropped {len(cols_to_drop)} columns with more than {self.missing_threshold}% missing values.",
                remaining_columns=len(X.columns),
                # dropped_columns=cols_to_drop,
            )
        else:
            Status.INFO(
                f"No columns with more than {self.missing_threshold}% missing values found."
            )
        return compute(X)

    def drop_high_cardinality_columns(
        self, X: dd.DataFrame, ratio_threshold=0.5
    ) -> dd.DataFrame:
        """
        Drops columns from the DataFrame that have a high cardinality, i.e., columns where the ratio of unique values
        to the total number of rows exceeds the specified threshold.
        Parameters:
        ---------
        X (dd.DataFrame): The input Dask DataFrame from which high cardinality columns will be dropped.
        ratio_threshold (float, optional): The threshold ratio of unique values to total rows above which columns
                                           will be considered high cardinality and dropped. Default is 0.5.

        Returns:
        dd.DataFrame: The DataFrame with high cardinality columns removed.
        """
        Status.INFO(
            "Evaluating high cardinality columns",
            self.submodule,
        )

        unique_counts = X.map_partitions(
            lambda df: df.nunique(), meta=(None, "int64")
        ).compute()
        unique_count_ratio = unique_counts / len(X.index)

        if id_cols := unique_count_ratio[
            unique_count_ratio > ratio_threshold
        ].index.tolist():
            X = X.drop(id_cols, axis=1)
            Status.INFO(
                f"Dropped {len(id_cols)} columns",
                self.submodule,
                dropped_columns=id_cols,
            )
        else:
            Status.INFO(
                "No high cardinality columns found.",
                self.submodule,
            )
        return X

    def drop_high_cardinality_columns_by_proportion(
        self, X: dd.DataFrame, proportion_threshold=0.01
    ) -> dd.DataFrame:
        """
        Drops columns from a Dask DataFrame that have high cardinality, defined as all unique values in the column occurring with a proportion less than the specified threshold.
        This method is useful for removing columns such as voucher numbers or order numbers, which can lead to overfitting due to their high cardinality.
        Args:
        ----
            X (dd.DataFrame): The input Dask DataFrame.
            proportion_threshold (float, optional): The maximum allowed proportion for the most frequent value in a column. Columns where the most frequent value occurs less than this proportion will be dropped. Defaults to 0.01.

        Returns:
            dd.DataFrame: The DataFrame with high cardinality columns removed.
        """
        Status.INFO(
            "Evaluating high cardinality columns by proportion",
            self.submodule,
        )
        columns_to_drop = []
        for col in tqdm(
            X.columns, desc="Evaluating high cardinality columns", file=sys.stdout
        ):
            try:
                # get the value counts for the column
                max_value_count = (
                    X[col].dropna().value_counts(normalize=True).max().compute()
                )
                # if max_value_count is NaN, it means the column has no values
                if pd.isna(max_value_count):
                    continue

                # check if no value is greater than proportion_threshold of the total rows
                if max_value_count < proportion_threshold:
                    columns_to_drop.append(col)
            except Exception:  # nosec
                continue

        if columns_to_drop:
            X = compute(X.drop(columns_to_drop, axis=1))
            Status.INFO(
                f"Dropped {len(columns_to_drop)} columns",
                self.submodule,
                dropped_columns=columns_to_drop,
            )
        else:
            Status.INFO(
                "No high cardinality columns found.",
                self.submodule,
            )
        return X

    def drop_large_text_columns(
        self, X: dd.DataFrame, length_threshold=5
    ) -> dd.DataFrame:
        """
        Drops columns from a Dask DataFrame that contain large text based on a specified word count threshold.
        Parameters:
        -----------
        X : dd.DataFrame
            The input Dask DataFrame from which large text columns will be evaluated and potentially dropped.
        length_threshold : int, optional (default=5)
            The word count threshold above which a column is considered a large text column.
        Returns:
        --------
        dd.DataFrame
            The DataFrame with large text columns dropped if any were found.
        Notes:
        ------
        - A column is considered a large text column if any of its entries have more words than the specified length_threshold.
        - The function logs the process of evaluating and dropping large text columns using the Status.INFO method.
        """
        Status.INFO(
            "Evaluating large text columns",
            self.submodule,
        )

        def _is_large_text(text):
            return 1 if len(str(text).strip().split()) > length_threshold else 0

        # Check if the column is a large text column
        word_count_df = X.copy().map_partitions(
            lambda df: df.astype(str).map(_is_large_text), meta=X
        )

        text_cols = word_count_df.max().compute()
        text_cols = text_cols[text_cols > 0].index.tolist()

        if text_cols:
            X = X.drop(text_cols, axis=1)
            Status.INFO(
                f"Dropped {len(text_cols)} large text columns",
                self.submodule,
                dropped_columns=text_cols,
            )
        else:
            Status.INFO(
                "No large text columns found.",
                self.submodule,
            )
        return X

    def get_categorical_columns(self, X) -> List[str]:
        """
        Returns a list of categorical columns in the given DataFrame.
        Parameters:
        - X (DataFrame): The input DataFrame.
        Returns:
        - List[str]: A list of categorical column names.
        """
        # code implementation
        Status.INFO(
            "Evaluating categorical columns",
            self.submodule,
            columns=len(X.columns),
        )
        X = X.copy()

        # get categorical columns
        cat_cols = [
            col
            for col in tqdm(
                X.columns, desc="Evaluating categorical columns", file=sys.stdout
            )
            if (is_string(X[col]) or is_boolean(X[col]))
        ]
        Status.INFO(
            f"Total categorical columns found: {len(cat_cols)}",
            self.submodule,
            columns=cat_cols,
        )

        X = self.drop_high_cardinality_columns(X[cat_cols])
        X = self.drop_high_cardinality_columns_by_proportion(X)
        X = self.drop_large_text_columns(X)
        return X.columns.tolist()

    def get_datetime_columns(self, X) -> List[str]:
        """
        Returns a list of datetime columns in the given DataFrame.
        Parameters:
        - X (DataFrame): The input DataFrame.
        Returns:
        - List[str]: A list of datetime column names.
        """
        Status.INFO(
            "Evaluating datetime columns",
            self.submodule,
            columns=len(X.columns),
        )
        X = X.copy()
        # get datetime columns
        datetime_cols = [
            col
            for col in tqdm(
                X.columns, desc="Evaluating datetime columns", file=sys.stdout
            )
            if is_datetime(X[col])
        ]
        Status.INFO(
            f"Total datetime columns found: {len(datetime_cols)}",
            self.submodule,
            columns=datetime_cols,
        )
        return datetime_cols

    def reduce_to_required(self, X) -> dd.DataFrame:
        """
        Reduce the input data frame `X` to only include the required columns.
        Parameters:
        - X: pandas DataFrame
            The input data frame to be reduced.
        Returns:
        - X: pandas DataFrame
            The reduced data frame with only the required columns.
        """

        Status.INFO(
            "Reducing data to required columns.",
            self.submodule,
        )
        total_columns = len(X.columns)

        # ETL process generates some columns which should be Dropped
        # Also some other columns which should be Dropped such as description and phase
        patterns_to_remove = config.get("REMOVABLE PATTERNS", "PATTERNS").split(",")
        patterns_to_remove = [pattern.strip().lower() for pattern in patterns_to_remove]
        # remove columns with patterns
        cols_to_remove = [
            col
            for col in X.columns
            if any(pattern in col.lower() for pattern in patterns_to_remove)
        ]

        # update categorical columns
        self.categorical_columns = [
            col for col in self.categorical_columns if col not in cols_to_remove
        ]

        # update numeric columns
        self.numeric_columns = [
            col for col in self.numeric_columns if col not in cols_to_remove
        ]

        # update datetime columns
        self.datetime_columns = [
            col for col in self.datetime_columns if col not in cols_to_remove
        ]

        # dont drop any columns that are in the encoder
        features_to_keep = self.encoder.get_feature_names_out()

        if cols_to_remove:
            X = X.drop(cols_to_remove, axis=1)
            Status.INFO(
                f"Dropped {len(cols_to_remove)} columns with removable patterns {patterns_to_remove}.",
                remaining_columns=len(X.columns),
                total_columns=total_columns,
            )
            Status.INFO(f"Dropped columns: {cols_to_remove}")

        # find index column
        index_col = config.get("DATA", "INDEX").lower()
        index_col = [col for col in X.columns if index_col == col.lower()]

        # slide data
        X = X[
            self.categorical_columns
            + index_col
            + self.numeric_columns
            + self.datetime_columns
            + features_to_keep
        ]
        Status.INFO(
            f"Reduced data to {len(X.columns)} columns.",
            self.submodule,
        )

        return compute(X)

    def replaces_empty_string_with_nan(self, _df):
        """
        Replaces all empty strings or strings containing only whitespace in the given DataFrame with NaN values.

        Parameters:
            _df (pandas.DataFrame): The input DataFrame to process.

        Returns:
            pandas.DataFrame: A DataFrame with empty or whitespace-only strings replaced by NaN.
        """

        return _df.replace(r"^\s*$", np.nan, regex=True)

    # Currently not being used
    # TODO: implement KNN imputation for dask based evaluation
    def knn_impute(self, X: dd.DataFrame, n_neighbors=5) -> dd.DataFrame:
        """
        Perform KNN imputation for missing data.
        Args:
            X (dd.DataFrame): The input DataFrame with missing values.
            n_neighbors (int, optional): The number of neighbors to consider for imputation. Defaults to 5.
        Returns:
            dd.DataFrame: The DataFrame with imputed values.
        """

        Status.INFO(
            "Starting KNN imputation for missing data",
            self.submodule,
        )

        # Function to apply KNN imputation to each partition
        def impute_partition(partition: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_values = imputer.fit_transform(partition)
            return pd.DataFrame(imputed_values, columns=partition.columns)

        index = X.index.to_series()

        # Apply the imputation function to each partition for given columns
        X = X.map_partitions(
            impute_partition, n_neighbors=n_neighbors, meta=X._meta
        ).set_index(index, drop=True)

        return X

    def identify_rare_labels(self, X: dd.DataFrame, y: dd.Series, cols: List) -> None:
        """
        Identify rare labels in specified columns of a Dask DataFrame.
        This method identifies rare labels in the specified columns of the input DataFrame `X`.
        A label is considered rare if its frequency is less than 1% of the total values in the column.
        Additionally, only those rare labels which are present exclusively in the negative class (target = 0)
        are considered to avoid bias in the model.
        Parameters:
        -----------
        X : dd.DataFrame
            The input Dask DataFrame containing the features.
        y : dd.Series
            The target variable corresponding to the input DataFrame `X`.
        cols : List
            A list of column names in which to identify rare labels.
        Returns:
        --------
        None
            This method updates the `self.rare_labels` dictionary with the identified rare labels for each column.
        """
        X_copy = X[cols].copy()  # copy the required columns
        X_copy = X_copy.map_partitions(
            lambda partition: partition.map(
                lambda x: x.lower().strip() if isinstance(x, str) else x
            ),
            meta=X_copy._meta,
        )
        X_copy["target"] = y
        X_copy = compute(X_copy)

        pbar = tqdm(cols, desc="Evaluating rare values", file=sys.stdout)
        for col in cols:
            pbar.set_postfix_str(f"Evaluating rare values for {col}")

            # get value counts
            value_counts = (
                X_copy[col].astype(str).value_counts(normalize=True).compute()
            )

            # index is always unique
            if rare_labels := value_counts[value_counts < 0.01].index.tolist():
                # we consider only those items as rare which are present only in target = 0
                # This is done to avoid any bias in the model
                # Downside is that we may miss some rare values which are present in target = 1
                # As it may increase the false positive rate but since we expect positive class to be much smaller than negative class
                # It is a trade off we are willing to make
                # This is as aligned with Krishna's suggestion
                positive_class_labels = (
                    X_copy.loc[X_copy["target"] == 1, col].unique().compute()
                )

                if rare_labels := [
                    str(label).strip()
                    for label in rare_labels
                    if label not in positive_class_labels
                ]:
                    self.rare_labels[col] = rare_labels

                    Status.INFO(
                        f"Found {len(rare_labels)} rare values for column {col}",
                        self.submodule,
                        rare_values=rare_labels,
                    )
            pbar.update(1)
        pbar.close()

    def impute_rare_labels(self, X: dd.DataFrame) -> dd.DataFrame:
        """
        Impute rare labels in the given Dask DataFrame.

        This method replaces rare labels in the specified columns of the DataFrame
        with a predefined unknown category. The columns to be processed and the
        rare labels for each column are specified in the `self.rare_labels` dictionary.

        Parameters:
        -----------
        X : dd.DataFrame
            The input Dask DataFrame containing the data to be processed.

        Returns:
        --------
        dd.DataFrame
            The DataFrame with rare labels imputed with the unknown category.

        Notes:
        ------
        - This method uses the `tqdm` library to display a progress bar for the columns being processed.
        - The `compute` function is called after processing each column to ensure the changes are applied.
        """

        def _replace_value(x: Any, rare_labels: List):
            return self.unknown_category if str(x).strip() in rare_labels else x

        for col in tqdm(
            self.rare_labels.keys(), desc="Imputing rare values", file=sys.stdout
        ):
            if col not in X.columns:
                continue

            X[col] = X[col].astype(str)
            X[col] = X[col].apply(
                _replace_value,
                args=(self.rare_labels[col],),
                meta=(col, "string[pyarrow]"),
            )

            X = compute(X)
        return X

    def constant_imputation(
        self,
        X: dd.DataFrame,
        cols: List[str],
        impute_value: Any,
    ) -> dd.DataFrame:
        """
        Impute missing values in specified columns with a constant value.
        Parameters:
        -----------
        X : dd.DataFrame
            The input dataframe containing the data to be imputed.
        cols : List[str]
            List of column names in which missing values need to be imputed.
        impute_value : Any
            The constant value to be used for imputing missing values.
        Returns:
        --------
        dd.DataFrame
            The dataframe with missing values imputed.
        Raises:
        -------
        BaseException
            If there is an error during the imputation process for any column.
        Notes:
        ------
        This method uses Dask DataFrame for handling large datasets and performs the imputation in parallel.
        """

        Status.INFO(
            f"Imputing missing values with constant {impute_value}.",
            self.submodule,
            columns_to_impute=len(cols),
        )

        # Impute other columns with a constant value
        for col in tqdm(cols, desc="Imputing columns", file=sys.stdout):
            try:
                X[col] = X[col].fillna(impute_value)
            except BaseException as _e:
                Status.FAILED(
                    f"Failed to fill missing values for {col}. Impute value {impute_value}",
                    self.submodule,
                    error=_e,
                )

        return compute(X)

    def datetime_imputation(
        self,
        X: dd.DataFrame,
        cols: List[str],
    ) -> dd.DataFrame:
        """
        Imputes missing values in specified datetime columns of a Dask DataFrame using the column median.

        This method iterates over the provided list of column names, converts each column to datetime,
        computes the median of non-missing values, and fills missing values with this median.
        Progress and status messages are logged throughout the process.

        Args:
            X (dd.DataFrame): The input Dask DataFrame containing the columns to be imputed.
            cols (List[str]): List of column names (strings) in X to perform datetime imputation on.

        Returns:
            dd.DataFrame: The DataFrame with missing datetime values imputed in the specified columns.
        """
        Status.INFO(
            "Imputing missing datetime values with column median.",
            self.submodule,
            columns_to_impute=len(cols),
        )

        for col in tqdm(cols, desc="Imputing datetime columns", file=sys.stdout):
            try:

                # convert to datetime
                X[col] = dd.to_datetime(X[col], errors="coerce")

                median_value = X[col].dropna().median_approximate().compute()

                X[col] = compute(X[col].fillna(median_value))
                Status.INFO(
                    f"Imputed {col} with median value: {median_value}",
                    self.submodule,
                )
            except BaseException as _e:
                Status.FAILED(
                    f"Failed to fill missing datetime values for {col}.",
                    self.submodule,
                    error=_e,
                    traceback=False,
                )

        return X

    def get_test_patterns(self, X) -> List[str]:
        """
        Retrieves the test patterns from the submodule and returns the feature columns.
        Parameters:
            X (pandas.DataFrame): The input DataFrame.
        Returns:
            List[str]: A list of feature columns.
        """

        pattern_df = self.submodule.get_test_patterns()
        if pattern_df is None or len(pattern_df) == 0:
            Status.INFO(
                "No test patterns found.",
                self.submodule,
            )
            return []

        # get the pattern id column
        pattern_id_col = (
            Setup().global_constants.get("TEST_PATTERNS", {}).get("PATTERN_ID_COLUMN")
        )

        return (
            [col.strip() for col in feature_name_patterns if col.strip() in X.columns]
            if (feature_name_patterns := pattern_df[pattern_id_col].unique().tolist())
            else []
        )

    def align_X_y(self, X: dd.DataFrame, y: dd.Series) -> dd.DataFrame:
        """
        Aligns the features DataFrame (X) with the target Series (y) by keeping only the rows in X that have corresponding indices in y.

        Parameters:
        -----------
        X : dd.DataFrame
            The features DataFrame.
        y : dd.Series
            The target Series.

        Returns:
        --------
        dd.DataFrame
            The aligned features DataFrame with only the rows that have corresponding indices in y.
        """
        # keep only those X rows that have y index
        if y is not None:
            # join X and y so that only the rows with target are retained
            X = X.merge(y.to_frame(), left_index=True, right_index=True).drop(
                columns=[y.name]
            )
            X = compute(X)
        return X

    def fit(self, X, y):  # pylint: disable=unused-argument
        """This function fits the data for the submodule"""
        # set y, we will use it in transform
        self._y = y
        # get amount column
        if amount_col := self.submodule.get_amount_column():
            amount_col = [
                col for col in X.columns if col.lower() == amount_col.lower()
            ][0]
            self.numeric_columns.append(amount_col)

        # get datetime columns
        # remove amount and categorical columns from datetime evaluation
        remaining_columns = [col for col in X.columns if col != amount_col]
        self.datetime_columns = self.get_datetime_columns(X[remaining_columns])

        # get categorical columns
        remaining_columns = [
            col for col in remaining_columns if col not in self.datetime_columns
        ]
        self.categorical_columns = self.get_categorical_columns(X[remaining_columns])
        
        # align X and y
        X = self.align_X_y(X, y)

        # reduce to required columns
        X = self.reduce_to_required(X)

        # identify rare labels
        test_patterns = self.get_test_patterns(X)
        cols_to_evaluate = [
            col for col in self.categorical_columns if col not in test_patterns
        ]
        self.identify_rare_labels(X, y, cols_to_evaluate)

        return self

    def transform(self, X):
        """
        Transforms the input data `X` by applying a series of preprocessing steps required for the submodule.

        The transformation process includes:
        - Logging the start of preprocessing.
        - Aligning `X` with the target variable `y` if available, and releasing `y` to free memory.
        - Reducing `X` to only the required columns.
        - Replacing empty strings in the data with NaN values.
        - Computing a summary of missing values.
        - Dropping rows with missing values, excluding specified test patterns.
        - Imputing missing values for categorical columns with a constant value and handling rare labels.
        - Imputing missing values for numeric columns with zero.
        Returns:
            The transformed data after all preprocessing steps.
        """
        Status.INFO(
            "Preprocessing data for submodule",
            self.submodule,
        )

        # align X and y
        if self._y is not None:
            X = self.align_X_y(X, self._y)
            # reset y to release memory
            self._y = None

        # reduce to required columns
        X = self.reduce_to_required(X)

        Status.INFO("Replacing empty strings with NaN", self.submodule)
        X = X.map_partitions(self.replaces_empty_string_with_nan, meta=X._meta)
        X = compute(X)

        # calculate missing values summary
        self.missing_summary = self.missing_values_summary(X)
        missing_percents = self.missing_summary["missing_percent"].to_dict()
        missing_percents = {k: f"{v:.2f}%" for k, v in missing_percents.items()}
        Status.INFO(
            "Missing values summary", self.submodule, missing_percent=missing_percents
        )

        # impute for test patterns
        # This is being done to make sure that test patterns are not dropped during missing value assessment
        test_patterns = self.get_test_patterns(X)
        X = self.drop_missing(X, exclude=test_patterns)

        # update categorical columns based on available columns
        self.categorical_columns = [
            col for col in self.categorical_columns if col in X.columns
        ]

        # For categorical columns, change the data type to string
        for col in self.categorical_columns:
            X[col] = X[col].astype("string[pyarrow]")

        if self.categorical_columns:
            Status.INFO(
                "Imputing missing values for categorical columns",
                self.submodule,
                total_columns=len(self.categorical_columns),
            )
            X = self.constant_imputation(
                X,
                cols=self.categorical_columns,
                impute_value=self.unknown_category,
            )
            X = self.impute_rare_labels(X)

        # update numeric columns
        self.numeric_columns = [col for col in self.numeric_columns if col in X.columns]
        if self.numeric_columns:
            Status.INFO(
                "Imputing missing values for numeric columns",
                self.submodule,
                total_columns=len(self.numeric_columns),
            )
            X = self.constant_imputation(X, cols=self.numeric_columns, impute_value=0)

        # update datetime columns
        self.datetime_columns = [
            col for col in self.datetime_columns if col in X.columns
        ]
        if self.datetime_columns:
            Status.INFO(
                "Imputing missing values for datetime columns",
                self.submodule,
                total_columns=len(self.datetime_columns),
            )
            X = self.datetime_imputation(X, cols=self.datetime_columns)

        return compute(X)
