# # Copyright (C) KonaAI - All Rights Reserved
"""
This module provides P2P entity reduction functions
"""
import dask.dataframe as dd
import pandas as pd
from src.tools.dask_tools import compute
from src.tools.dask_tools import shuffle_and_transform
from src.utils.notification import EmailNotification
from src.utils.status import Status
from src.utils.submodule import Submodule


CREDIT_VALUES = ["H-Credit", "H"]
DEBIT_VALUES = ["S-Debit", "S"]
credit_debit_col = "DebitCreditIndicator"


def notify_for_invalid_values(submodule: Submodule, ddf: dd.DataFrame):
    """
    Checks for invalid values in the specified credit/debit column of a Dask DataFrame and sends an email notification if any are found.
    Args:
    -----
        submodule (Submodule): The submodule instance containing configuration and context, including the instance ID.
        ddf (dd.DataFrame): The Dask DataFrame to be checked for invalid values.

    Behavior:
        - Checks if the credit/debit column exists in the DataFrame.
        - Identifies values in the column that are not present in the allowed CREDIT_VALUES or DEBIT_VALUES.
        - If invalid values are found, sends an email notification with details about the invalid entries.

    Raises:
        None
    """
    notifier = EmailNotification(instance_id=submodule.instance_id)

    if credit_debit_col in ddf.columns:
        Status.INFO(
            f"Checking for invalid values in {credit_debit_col} for {submodule}"
        )
        unique_values = ddf[credit_debit_col].unique().compute().tolist()
        invalid_values = [
            value
            for value in unique_values
            if value not in (CREDIT_VALUES + DEBIT_VALUES)
        ]

        if invalid_values:
            s = Status.WARNING(
                f"Invalid values found in {credit_debit_col}: {invalid_values}",
                submodule,
            )
            notifier.add_content(
                f"Invalid values found in {credit_debit_col}",
                content=s.to_dict(),
            )
            notifier.send(
                subject=f"Invalid values in {credit_debit_col} for {submodule.instance_id}"
            )


def create_hash(_pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a hash for a pandas DataFrame.

    Args:
    ----
        _pdf (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with a generated hash.

    """
    return pd.util.hash_pandas_object(_pdf, index=False).astype(str)


def reduce_invoice_level(_df: dd.DataFrame):
    """
    Aggregates invoice line item data to the vendor and invoice level.
    This function processes a Dask DataFrame by:
    - Converting the amount column to float and filling missing values with 0.
    - Creating vendor and invoice level identifiers.
    - Validating the credit/debit column for unexpected values.
    - Creating separate columns for credit and debit amounts, ensuring debit amounts are negative.
    - Calculating the net amount for each row.
    - Dropping intermediate columns used for calculation.
    - Aggregating the data by vendor and invoice level, summing the net amounts.
    - Returning both the aggregated DataFrame and a version with duplicate vendor-invoice pairs removed.
    """
    Status.INFO("Aggregating data to vendor and invoice level")

    amount_column = "AmountExcl_RC"

    # convert amount column to float and fill missing values with 0
    _df[amount_column] = _df[amount_column].astype(float).fillna(0)
    _df["vendor_level"] = _df["VendorNumber"].astype("string[pyarrow]")
    _df["invoice_level"] = _df.map_partitions(
        lambda x: create_hash(x[["CompanyCode", "FiscalYear", "SystemInvoiceNo"]]),
        meta=("invoice_level", "string[pyarrow]"),
    ).astype("string[pyarrow]")

    # confirm that create column has only two unique values
    invalid_values = [
        value
        for value in _df[credit_debit_col].unique().compute()
        if value not in (CREDIT_VALUES + DEBIT_VALUES)
    ]
    if invalid_values:
        Status.WARNING(f"Invalid values found in {credit_debit_col}: {invalid_values}")

    # create column for "H-Credit" and "S-Debit"
    _df["credit_amount"] = _df[amount_column].where(
        _df[credit_debit_col].isin(CREDIT_VALUES + invalid_values), 0
    )
    _df["debit_amount"] = _df[amount_column].where(
        _df[credit_debit_col].isin(DEBIT_VALUES), 0
    )
    # make sure debit amount is always negative
    _df["debit_amount"] = _df["debit_amount"].where(
        _df["debit_amount"] < 0, -_df["debit_amount"]
    )

    # net amount is either credit amount or debit amount whichever is not zero
    _df["net_amount"] = _df["credit_amount"].where(
        _df["credit_amount"] != 0, _df["debit_amount"]
    )

    _df = _df.drop(
        columns=[amount_column, credit_debit_col, "credit_amount", "debit_amount"]
    )

    _df = shuffle_and_transform(
        _df,
        ["vendor_level", "invoice_level"],
        "net_amount",
        f"Total_{amount_column}",
        float,
        "sum",
    )
    _df[f"Total_{amount_column}"] = _df[f"Total_{amount_column}"].astype(float).abs()

    # drop duplicate rows
    _df_ow_duplicate = _df.drop_duplicates(
        subset=["invoice_level", "vendor_level"], keep="first"
    )
    _df_ow_duplicate = compute(_df_ow_duplicate)
    Status.INFO("Data reduction completed")

    return _df, _df_ow_duplicate


def reduce_payment_level(_df: dd.DataFrame):
    """
    Aggregates invoice line items to the vendor and payment level, computes net amounts, and removes duplicates.
    Parameters:
    ----------
        _df (dd.DataFrame): Input Dask DataFrame containing invoice line items. Must include columns:
            - "AmountExcl_RC": Amount column to aggregate.
            - "VendorNumber": Vendor identifier.
            - "CompanyCode", "FiscalYear", "SystemPaymentNo": Used to generate payment-level hash.
            - credit_debit_col: Column indicating credit or debit type.
    Returns:
        Tuple[dd.DataFrame, dd.DataFrame]:
            - The transformed DataFrame with aggregated net amounts at vendor and payment level.
            - A DataFrame with duplicate (vendor_level, payment_level) rows dropped.
    """
    Status.INFO("Aggregating data to vendor and payment level")

    amount_column = "AmountExcl_RC"

    # convert amount column to float and fill missing values with 0
    _df[amount_column] = _df[amount_column].astype(float).fillna(0)
    _df["vendor_level"] = _df["VendorNumber"].astype("string[pyarrow]")
    _df["payment_level"] = _df.map_partitions(
        lambda x: create_hash(x[["CompanyCode", "FiscalYear", "SystemPaymentNo"]]),
        meta=("payment_level", "string[pyarrow]"),
    ).astype("string[pyarrow]")

    # confirm that create column has only two unique values
    invalid_values = [
        value
        for value in _df[credit_debit_col].unique().compute()
        if value not in (CREDIT_VALUES + DEBIT_VALUES)
    ]
    if invalid_values:
        Status.WARNING(f"Invalid values found in {credit_debit_col}: {invalid_values}")

    # create column for "Credit" and "Debit"
    _df["credit_amount"] = _df[amount_column].where(
        _df[credit_debit_col].isin(CREDIT_VALUES + invalid_values), 0
    )
    _df["debit_amount"] = _df[amount_column].where(
        _df[credit_debit_col].isin(DEBIT_VALUES), 0
    )
    # make sure debit amount is always negative
    _df["debit_amount"] = _df["debit_amount"].where(
        _df["debit_amount"] < 0, -_df["debit_amount"]
    )

    # net amount is either credit amount or debit amount whichever is not zero
    _df["net_amount"] = _df["credit_amount"].where(
        _df["credit_amount"] != 0, _df["debit_amount"]
    )
    _df = _df.drop(columns=["credit_amount", "debit_amount"])

    _df = shuffle_and_transform(
        _df,
        ["vendor_level", "payment_level"],
        "net_amount",
        f"Total_{amount_column}",
        float,
        "sum",
    )
    _df[f"Total_{amount_column}"] = _df[f"Total_{amount_column}"].astype(float).abs()

    # drop duplicate rows
    _df_ow_duplicate = _df.drop_duplicates(
        subset=["payment_level", "vendor_level"], keep="first"
    )

    _df_ow_duplicate = compute(_df_ow_duplicate)
    Status.INFO("Data reduction completed")

    return _df, _df_ow_duplicate


def reduce_po_level(_df: dd.DataFrame):
    """
    Aggregates purchase order line items to the vendor and purchase order (PO) level.
    This function performs the following steps:
    1. Converts the 'AmountRC' column to float and fills missing values with 0.
    2. Creates a 'vendor_level' column as a string representation of 'VendorNumber'.
    3. Generates a unique 'po_level' identifier by hashing 'CompanyCode' and 'PONumber'.
    4. Groups the data by 'vendor_level' and 'po_level', summing the 'AmountRC' for each group.
    5. Merges the aggregated totals back into the original DataFrame.
    6. Removes duplicate rows based on 'po_level' and 'vendor_level', keeping the first occurrence.
    7. Computes the final deduplicated DataFrame.
    Args:
    -----
        _df (dd.DataFrame): Input Dask DataFrame containing purchase order line items.

    Returns:
        Tuple[dd.DataFrame, pd.DataFrame]:
            - The original DataFrame with aggregated total amounts per vendor and PO.
            - A deduplicated DataFrame at the vendor and PO level.
    """
    Status.INFO("Aggregating data to vendor and PO level")

    amount_column = "AmountRC"

    # convert amount column to float and fill missing values with 0
    _df[amount_column] = _df[amount_column].astype(float).fillna(0)

    _df["vendor_level"] = _df["VendorNumber"].astype("string[pyarrow]")
    _df["po_level"] = _df.map_partitions(
        lambda x: create_hash(x[["CompanyCode", "PONumber"]]),
        meta=("po_level", "string[pyarrow]"),
    ).astype("string[pyarrow]")

    grouped = (
        _df.groupby(["vendor_level", "po_level"])[amount_column].sum().reset_index()
    )
    grouped.columns = ["vendor_level", "po_level", f"Total_{amount_column}"]

    _df = _df.merge(grouped, on=["vendor_level", "po_level"], how="left")

    # drop duplicate rows
    _df_ow_duplicate = _df.drop_duplicates(
        subset=["po_level", "vendor_level"], keep="first"
    )

    _df_ow_duplicate = compute(_df_ow_duplicate)
    Status.INFO("Data reduction completed")

    return _df, _df_ow_duplicate
