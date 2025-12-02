# # Copyright (C) KonaAI - All Rights Reserved
"""This module handles one hot encoding process for the AutoML pipeline."""
import re
import sys
import unicodedata
from typing import Dict
from typing import List

import dask.dataframe as dd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from src.tools.dask_tools import compute
from src.utils.status import Status
from tqdm import tqdm


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom one-hot encoder transformer.

        columns (List[str]): The list of columns to encode.
        unknown_replacement (str): The value to replace unseen categories with.
        prefix_sep (str, optional): The separator to use between column name and category name in the encoded features. Defaults to "_".
        categories_ (Dict): A dictionary mapping each column to its unique categories.
    """

    def __init__(
        self, columns: List[str], unknown_replacement: str, prefix_sep: str = None
    ) -> None:
        self.columns = columns or []
        self.unknown_replacement = unknown_replacement
        self.prefix_sep = prefix_sep or "_"
        self.categories_: Dict = {}
        self.feature_names_ = None

    def _replace_unseen_categories(
        self, s: dd.Series, categories: set[str]
    ) -> dd.Series:
        """
        Replaces unseen categories in a Series with a specified replacement value.
        Args:
            s (dd.Series): The input Series.
            categories (set[str]): The set of categories to replace.
        Returns:
            dd.Series: The Series with replaced unseen categories.
        """

        return s.mask(~s.isin(categories), self.unknown_replacement)

    def _create_feature_name(self, col: str, category: str) -> str:
        # sanitize category name so that it can be utilized as a pandas column name
        category = (
            unicodedata.normalize("NFKD", category)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        # remove special characters
        category = re.sub(r"[^a-zA-Z0-9]+", "_", category).strip()

        return f"{col}{self.prefix_sep}{category}"

    def process_column(self, X: dd.DataFrame, col: str) -> dd.DataFrame:
        """
        Processes a specified column in a Dask DataFrame by performing one-hot encoding.
        Parameters:
        ----------
        X (dd.DataFrame): The input Dask DataFrame.
        col (str): The name of the column to be processed.

        Returns:
        dd.DataFrame: The transformed Dask DataFrame with the specified column one-hot encoded and the original column removed.

        Raises:
        KeyError: If the specified column is not found in the input DataFrame.

        Notes:
        - Unseen categories in the new data are replaced.
        - All values in the column are converted to lower case.
        - The unknown category is removed from the list of categories.
        - Each category is processed and added as a new column with binary values indicating the presence of the category.
        """
        if col not in X.columns:
            raise KeyError(
                f"Column {col} not found in input data. Cannot transform, all columns must be present for transformation."
            )

        # make sure all values are lower case
        # This should be done before replacing unseen categories
        # to ensure that unseen categories are replaced correctly
        X[col] = X[col].astype("string[pyarrow]").str.strip().str.lower()
        X = compute(X)

        # replace unseen categories in new data
        X[col] = self._replace_unseen_categories(X[col], self.categories_[col])

        # remove unknown category from categories
        categories = [c for c in self.categories_[col] if c != self.unknown_replacement]
        pbar_desc = f"Processing column {col}"
        # create new columns for each category
        for category in tqdm(categories, desc=pbar_desc, leave=False, file=sys.stdout):
            feature_name = self._create_feature_name(col, category)
            X[feature_name] = (X[col] == category).astype("int8")
            X = compute(X)

        return compute(X.drop(col, axis=1))

    def _build_categories(self, s: dd.Series) -> List[str]:
        """
        Build a list of unique categories from a Dask Series.
        This method processes the input Dask Series to generate a list of unique
        category values. It ensures that the categories are case-insensitive and
        whitespace-trimmed. Additionally, it includes a specified unknown replacement
        value if it is not already present in the list of unique categories.
        Args:
            s (dd.Series): The input Dask Series from which to extract unique categories.
        Returns:
            List[str]: A sorted list of unique category values as strings.
        """
        unique_values = s.dropna().unique().compute().tolist()
        # make sure unique values are left irrespective of case
        unique_values = list({str(val).strip().lower() for val in unique_values})
        # add unknown replacement to categories if not present
        if self.unknown_replacement not in unique_values:
            unique_values.append(self.unknown_replacement)

        # we need to sort as text to ensure the same order of categories
        return sorted(unique_values, key=str)

    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names for transformation.

        This method returns the feature names after one-hot encoding, excluding any
        feature names that contain the unknown replacement value.

        Returns:
            List[str]: A list of feature names after one-hot encoding, excluding
                       those with the unknown replacement value.
        """
        # remove unknown data columns from feature names
        return [
            name for name in self.feature_names_ if self.unknown_replacement not in name
        ]

    def fit(
        self, X: dd.DataFrame, y=None  # pylint: disable=unused-argument
    ) -> "CustomOneHotEncoder":
        """
        Fits the CustomOneHotEncoder to the input Dask DataFrame by determining the valid columns and their unique categories for one-hot encoding.
        Parameters:
        ---------
            X (dd.DataFrame): The input Dask DataFrame containing the features to be encoded.
            y (optional): Target values (not used in fitting, included for compatibility).

        Returns:
            CustomOneHotEncoder: Returns self, fitted to the input data.

        Notes:
            - Columns with fewer than 2 unique categories or with a number of categories equal to the length of y are excluded from encoding.
            - Updates internal attributes `categories_`, `columns`, and `feature_names_` based on the fitting process.
        """
        if len(self.columns) == 0:
            Status.NOT_FOUND("No columns to onehot encode")
        else:
            Status.INFO(f"Fitting onehot encoder for {len(self.columns)} columns")

        valid_columns = []
        for col in tqdm(self.columns, desc="Fitting onehot encoder", file=sys.stdout):
            categories = self._build_categories(X[col]) or []
            # if categories are equal to 1 or length of y, then skip this column
            if len(categories) < 2 or len(categories) == len(y):
                # remove from columns
                continue
            self.categories_[col] = categories
            valid_columns.append(col)

        self.columns = valid_columns
        # create feature names
        self.feature_names_ = [
            self._create_feature_name(col, category)
            for col in self.columns
            for category in self.categories_[col]
        ]

        return self

    def transform(self, X: dd.DataFrame) -> dd.DataFrame:
        """
        Transforms the input Dask DataFrame by applying one-hot encoding to the specified columns.
        Parameters:
        ---------
            X (dd.DataFrame): The input Dask DataFrame to be transformed.

        Returns:
            dd.DataFrame: The transformed DataFrame with one-hot encoded features for the specified columns.

        Notes:
            - If no columns are specified for encoding, the original DataFrame is returned unchanged.
            - Progress is displayed using tqdm for each column being encoded.
        """
        if len(self.columns) == 0:
            Status.NOT_FOUND("Found empty encoder")
            return X

        Status.INFO(f"Creating onehot encoded features for {len(self.columns)} columns")

        for col in tqdm(
            self.columns, desc="Creating onehot encoded features", file=sys.stdout
        ):
            X = self.process_column(X, col)

        return X
