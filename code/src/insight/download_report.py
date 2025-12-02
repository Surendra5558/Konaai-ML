# # Copyright (C) KonaAI - All Rights Reserved
"""
This module builds Excel reports containing various insight data
for a given project instance. It aggregates instance metadata,
transaction summaries, risk patterns, concern records, and AutoML
experiment results into structured Excel sheets.
"""
from io import BytesIO

import pandas as pd
from src.insight.fetch_insightdata import get_amount_count_data
from src.insight.fetch_insightdata import get_automl_report_data
from src.insight.fetch_insightdata import get_concern_report
from src.insight.fetch_insightdata import get_top_risk_patterns
from src.utils.global_config import GlobalSettings


def get_info_df(instance_id: str) -> pd.DataFrame:
    """
    Builds a DataFrame containing metadata information for a given project instance.
    Args:
        instance_id (str): The unique identifier for the project instance.
    Returns:
        pd.DataFrame: A DataFrame with two columns, "Field" and "Value", containing
        the client name, project name, and instance ID. Returns an empty DataFrame
        if the instance is not found.
    """
    instance = GlobalSettings().instance_by_id(instance_id)
    if not instance:
        return pd.DataFrame()

    info_dict = {
        "Client Name": instance.client_name,
        "Project Name": instance.project_name,
        "Instance ID": instance.instance_id,
    }

    # Convert dict into two-column DataFrame (field, value)
    return pd.DataFrame(list(info_dict.items()), columns=["Field", "Value"])


def excel_report(instance_id: str) -> BytesIO:
    """
    Generates an Excel report for a given instance, including an Info sheet and metadata columns in all report sheets.
    Args:
        instance_id (str): The unique identifier for the instance to generate the report for.
    Returns:
        BytesIO: An in-memory binary stream containing the generated Excel report.
    The report includes the following sheets:
        - Info: General information about the instance.
        - Amount and Count Report: Data with metadata columns prepended.
        - Top Risk PatternIds Report: Data with metadata columns prepended.
        - Concern Records: Data with metadata columns prepended.
        - AutoML Report: Data with metadata columns prepended.
    Each report sheet includes the following metadata columns at the beginning:
        - Client Name
        - Project Name
        - Instance ID
    If any DataFrame is empty, its corresponding sheet is omitted from the report.
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")

    # Build Info DataFrame
    info_df = get_info_df(instance_id)

    # Build report DataFrames
    amount_count_df = pd.DataFrame(get_amount_count_data(instance_id))
    top_risk_df = pd.DataFrame(get_top_risk_patterns(instance_id))
    concern_df = pd.DataFrame(get_concern_report(instance_id))
    automl_df = pd.DataFrame(get_automl_report_data(instance_id))

    # Write Info sheet
    if not info_df.empty:
        info_df.to_excel(writer, sheet_name="Info", index=False)

    # Fetch instance metadata for column injection
    instance = GlobalSettings().instance_by_id(instance_id)
    meta_cols = {
        "Client Name": instance.client_name if instance else "",
        "Project Name": instance.project_name if instance else "",
        "Instance ID": instance.instance_id if instance else "",
    }

    def write_with_metadata(sheet_name, df):
        if df.empty:
            return

        df = df.copy()
        for col, val in meta_cols.items():
            df[col] = val

        # Reorder so metadata columns appear first
        cols = list(meta_cols.keys()) + [c for c in df.columns if c not in meta_cols]
        df = df[cols]

        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Write sheets with metadata columns
    write_with_metadata("Amount and Count Report", amount_count_df)
    write_with_metadata("Top Risk PatternIds Report", top_risk_df)
    write_with_metadata("Concern Records", concern_df)
    write_with_metadata("AutoML Report", automl_df)

    writer.close()
    output.seek(0)
    return output
