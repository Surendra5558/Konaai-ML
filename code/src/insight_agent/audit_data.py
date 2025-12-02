# # Copyright (C) KonaAI - All Rights Reserved
"""Audit Data Module"""
from typing import List
from typing import Optional

import dask.dataframe as dd
import pandas as pd
from src.automl.explainer import ExplainationOutput
from src.automl.explainer import PredictionExplainer
from src.insight_agent import constants
from src.utils.conf import Setup
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule


class AuditData:
    """Audit Data Handler for fetching patterns and audit-related data."""

    submodule_obj: Submodule = None
    transaction_id: int = None

    def __init__(self, submodule: Submodule, transaction_id: int):
        self.submodule_obj = submodule
        self.transaction_id = transaction_id

    def fetch_automl_explanation(self) -> Optional[ExplainationOutput]:
        """
        Fetch an AutoML explanation for the current transaction.

        This method instantiates a PredictionExplainer using the instance's
        self.submodule_obj and self.transaction_id, invokes its explain()
        method, and returns the explanation if one is produced. Falsy
        explanation values are normalized to None.

        Returns:
            Optional[ExplainationOutput]: The explanation produced by the explainer,
            or None if no explanation is available.

        Raises:
            Any exceptions raised by PredictionExplainer construction or its
            explain() method are propagated to the caller.

        Notes:
            - Requires that self.submodule_obj and self.transaction_id are set and
              valid for creating a PredictionExplainer.
            - The method does not perform additional validation or transformation
              of the returned explanation beyond converting falsy values to None.
        """
        # sourcery skip: use-named-expression
        explainer = PredictionExplainer(
            submodule_obj=self.submodule_obj, transaction_id=self.transaction_id
        )

        explanation: ExplainationOutput = explainer.explain()
        return explanation or None

    def fetch_patterns(self) -> Optional[pd.DataFrame]:
        """
        Fetch and return CSV-formatted information about test patterns that produced hits
        for the current transaction.

        Behavior summary
        - Resolves the target instance via GlobalSettings using self.submodule_obj.instance_id.
        - Loads the test patterns DataFrame from self.submodule_obj.get_test_patterns().
        - Loads transaction data (Dask DataFrame) for self.transaction_id from the project DB
            and computes it to a pandas DataFrame.
        - Identifies pattern score columns by appending the suffix "_Tran_Score" to pattern IDs
            obtained from the test patterns configuration.
        - Selects only those patterns whose transaction score column contains at least one
            positive value (> 0). Patterns without hits are ignored.
        - Filters and orders the patterns DataFrame by configured weightage (descending),
            renames columns to standardized output names, and adds an "Additional Test Info"
            column populated from matching transaction-level columns (if present).
        - Returns the resulting table serialized as CSV (string) with all fields quoted.

        Return value
        - Returns a pandas DataFrame containing details of patterns that produced hits
        - Returns None when any of the following conditions occur (and Status.NOT_FOUND is
            invoked with contextual information):
            - the target instance cannot be found,
            - no test patterns data is available,
            - no transaction data is available for the given transaction_id,
            - no patterns produced hits in the transaction data.

        Side effects and dependencies
        - Calls:
                GlobalSettings.instance_by_id(...)
                self.submodule_obj.get_test_patterns()
                self.submodule_obj.get_data_table_name()
                instance.settings.projectdb.download_table_or_query(...)
                Status.NOT_FOUND(...) to report missing resources
                Setup().global_constants to read configuration keys:
                    "TEST_PATTERNS" -> "PATTERN_ID_COLUMN", "WEIGHTAGE_COLUMN",
                                                         "PATTERN_CATEGORY_COLUMN", "PATTERN_DESCRIPTION_COLUMN"
        - Assumes the transaction DataFrame contains columns named "<pattern_id>_Tran_Score"
            for pattern scores and may contain other columns containing the pattern id as a
            substring that are used to populate "Additional Test Info".
        - Can raise downstream exceptions from database access, Dask compute, or pandas
            indexing if the expected columns/configuration are absent.

        Notes
        - Only patterns with a strictly positive score are considered "hits".
        - The returned CSV is fully quoted (csv.QUOTE_STRINGS) and does not include an index.
        """
        instance = GlobalSettings.instance_by_id(
            instance_id=self.submodule_obj.instance_id
        )
        if not instance:
            Status.NOT_FOUND(
                "Instance not found", instance_id=self.submodule_obj.instance_id
            )
            return None

        patterns_data: pd.DataFrame = self.submodule_obj.get_test_patterns()
        if patterns_data is None or len(patterns_data) == 0:
            Status.NOT_FOUND(
                "No test patterns data found for audit agent", self.submodule_obj
            )
            return None

        # we get enabled pattern ids sorted by weightage
        pattern_id_col = (
            Setup().global_constants.get("TEST_PATTERNS", {}).get("PATTERN_ID_COLUMN")
        )
        pattern_ids: List[str] = patterns_data[pattern_id_col].tolist()

        db = instance.settings.projectdb
        data_query = constants.TRANSACTION_DATA_QUERY.format(
            table_name=self.submodule_obj.get_data_table_name(),
            transaction_id=self.transaction_id,
        )
        dff: dd.DataFrame = db.download_table_or_query(query=data_query)
        if dff is None or len(dff) == 0:
            Status.NOT_FOUND(
                "No transaction data found for audit agent",
                self.submodule_obj,
                transaction_id=self.transaction_id,
            )
            return None

        transaction_df: pd.DataFrame = dff.compute()

        # Filter columns based on hit patterns
        tran_score_sfx = "_Tran_Score"
        pattern_score_cols = [f"{p}{tran_score_sfx}" for p in pattern_ids]
        pattern_score_data = transaction_df[pattern_score_cols]
        pattern_score_data = pattern_score_data[pattern_score_data > 0].dropna(
            axis=1, how="all"
        )
        valid_patterns = [
            col.replace(tran_score_sfx, "")
            for col in pattern_score_data.columns.tolist()
        ]
        if not valid_patterns:
            Status.NOT_FOUND(
                "No patterns with hits found in transaction data for audit agent",
                self.submodule_obj,
                transaction_id=self.transaction_id,
            )
            return None

        # valid_patterns now has only those patterns which have hit scores
        patterns_data = patterns_data[
            patterns_data[pattern_id_col].isin(valid_patterns)
        ]
        weitage_col = (
            Setup().global_constants.get("TEST_PATTERNS", {}).get("WEIGHTAGE_COLUMN")
        )
        category_col = (
            Setup()
            .global_constants.get("TEST_PATTERNS", {})
            .get("PATTERN_CATEGORY_COLUMN")
        )
        desc_col = (
            Setup()
            .global_constants.get("TEST_PATTERNS", {})
            .get("PATTERN_DESCRIPTION_COLUMN")
        )
        cols_required = [
            pattern_id_col,
            desc_col,
            category_col,
            weitage_col,
        ]

        patterns_data = patterns_data[cols_required]
        patterns_data = patterns_data.sort_values(by=weitage_col, ascending=False)
        patterns_data = patterns_data.rename(
            columns={
                pattern_id_col: "PatternID",
                desc_col: "Pattern Description",
                category_col: "Pattern Category",
                weitage_col: "Weightage",
            }
        )

        add_info_col = "Additional Test Info"
        patterns_data[add_info_col] = None

        for pattern in valid_patterns:
            if additional_column := [
                col
                for col in transaction_df.columns
                if pattern in col and tran_score_sfx not in col and col != pattern
            ]:
                patterns_data.loc[
                    patterns_data["PatternID"] == pattern, add_info_col
                ] = transaction_df[additional_column[0]].values[0]

        return patterns_data
