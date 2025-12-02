# # Copyright (C) KonaAI - All Rights Reserved
"""This file is used to generate data for the expense dataset."""
import secrets
from datetime import datetime

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from data_generators.generate_master import generate_transaction_master
from data_generators.generate_master import generate_vendor_master
from data_generators.p2p_features import generate_core_features
from data_generators.utils import config
from data_generators.utils import set_random_to_null
from faker import Faker

fake = Faker()

vendor_master = generate_vendor_master(
    records=int(config.get("DATA", "MASTER_RECORDS"))
)
transaction_master = generate_transaction_master()

start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-11-22", "%Y-%m-%d")


def generate_expense_data():
    """
    Generate a single synthetic expense transaction record.

    The record includes hierarchical employee data, transaction metadata,
    merchant information, monetary fields (amounts, taxes, reimbursements),
    and status indicators. Data is derived from pre-generated vendor and
    transaction masters with additional randomization using Faker and NumPy.

    Returns:
        dict: A dictionary representing a single expense transaction.
    """
    vendor = vendor_master[secrets.randbelow(len(vendor_master) - 1)]

    return {
        "CorpID": fake.random_number(digits=5),
        "HierarchyLevel1": fake.random_number(digits=4),
        "HierarchyLevel1_Name": vendor[2],
        "HierarchyLevel2": fake.random_number(digits=4),
        "HierarchyLevel2_Name": vendor[2],
        "HierarchyLevel3": fake.random_number(digits=4),
        "HierarchyLevel3_Name": vendor[2],
        "HierarchyLevel4": fake.random_number(digits=4),
        "HierarchyLevel4_Name": vendor[2],
        "AccountNumber(Short)": fake.random_number(digits=4),
        "CardholderLastName": fake.last_name(),
        "CardholderFirstName": fake.first_name(),
        "EmployeeID": vendor[3],
        "AccountStatus": np.random.choice(transaction_master["status"]),
        "MerchantAcceptorID": fake.random_number(digits=5),
        "MerchantName": vendor[2],
        "MerchantCity": vendor[5],
        "MerchantCountry": vendor[1],
        "MCC": fake.random_number(digits=4),
        "LineItemDescription": np.random.choice(
            transaction_master["LineItemDescription"]
        ),
        "TransactionReferenceNumber": fake.random_number(digits=5),
        "TransactionDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "TransactionPostDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "TransactionType": np.random.choice(["regular", "child"]),
        "TransactionAmount": secrets.randbelow(5000000),
        "DBReferenceNumber": fake.random_number(digits=5),
        "MerchantStreetAddress": vendor[7],
        "MerchantState": vendor[6],
        "MerchantZip": fake.random_number(digits=5),
        "MerchantCountryCode": vendor[1],
        "TransactionNumber": fake.random_number(digits=5),
        "TransactionCurrencyCode": vendor[4],
        "TransactionBillingCurrency": vendor[1],
        "TransactionBillingAmount": secrets.randbelow(5000000),
        "SalesTax": secrets.randbelow(5000000),
        "ItemQuant": fake.random_number(digits=1),
        "ItemUnitCost": secrets.randbelow(5000),
        "EmployeeName": vendor[2],
        "SPT_Source": fake.sentence(nb_words=10, variable_nb_words=True),
        "DefaultExpenseReportApprover": vendor[2],
        "Payment Type": np.random.choice(transaction_master["payment_type"]),
        "Default Expense Report Approver ID": fake.random_number(digits=5),
        "Default Expense Report Approver": vendor[2],
        "Approval Status": np.random.choice(transaction_master["Approval Status"]),
        "Payment Status": np.random.choice(["Paid", "Not paid"]),
        "Created Date": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "Expense Location": vendor[6],
        "Expense Country": vendor[1],
        "Is Personal Expense": np.random.choice(["Y", "N"]),
        "Reimbursement Amount": secrets.randbelow(5000000),
        "Reimbursement Currency": vendor[4],
        "First Submitted Date": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "From Location": vendor[7],
        "Approved Amount": secrets.randbelow(5000000),
        "Reimbursement Amount USD": secrets.randbelow(500000),
        "Transaction Line Type": np.random.choice(
            transaction_master["Transaction Line Type"]
        ),
        "Report ID": fake.random_number(digits=5),
        "Payment Date": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "Approved Date": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
    }


def generate_expense_dataframe(records=10):
    """
    Generate a Dask dataframe containing synthetic expense records with features and labels.

    Each record is created using `generate_expense_data()` and then enhanced with core features
    from `generate_core_features()`. Random null values are introduced to simulate real-world data.
    Binary labels (`y`) are extracted from the 'target' column of the core features.

    Args:
    -----
        records (int, optional): Number of records to generate. Defaults to 10.

    Returns:
        tuple:
            dask.dataframe.DataFrame: The feature dataframe with synthetic transaction records.
            dask.dataframe.Series: The binary label series derived from core features.
    """
    # Create a list of delayed objects
    delayed_data = [dask.delayed(generate_expense_data)() for _ in range(records)]
    delayed_data = dask.delayed(list)(delayed_data)
    delayed_data = dask.delayed(pd.DataFrame)(delayed_data)

    # Create a meta dataframe for the delayed dataframe
    meta = pd.DataFrame(generate_expense_data(), index=[0])
    # Create a dask dataframe from the delayed dataframe
    df = dd.from_delayed(delayed_data, meta=meta)

    core_features = generate_core_features(records)
    core_features = core_features.set_index(df.index.compute())
    core_features = dd.from_pandas(core_features, npartitions=df.npartitions)

    # merge core features with transaction data
    df = df.merge(core_features, left_index=True, right_index=True)

    # generate labels, generate 1 where sum of all features is greater than 50
    y = df["target"]
    # drop target column from X
    df = df.drop(columns=["target"])

    # introduce null values
    df = df.map_partitions(set_random_to_null, fraction=0.1)

    # set index columns
    index_col = config.get("DATA", "INDEX")
    df[index_col] = df.index
    # df = df.set_index(index_col)
    df = df.reset_index(drop=True)

    return df, y


if __name__ == "__main__":
    # generate dataframe with multiple of 100 records
    generate_expense_dataframe(records=1000)
