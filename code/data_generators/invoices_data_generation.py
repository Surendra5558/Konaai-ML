# # Copyright (C) KonaAI - All Rights Reserved
"""This module generates transaction data for invoices"""
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


def get_boolean_string(true_chance=50):
    """Return 'True' or 'False' string based on probability"""
    return "True" if secrets.randbelow(100) < true_chance else "False"


start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-11-22", "%Y-%m-%d")


def generate_transaction_data():
    """
    Generate a single synthetic invoice transaction record.

    Combines vendor information with randomized financial and metadata fields
    to simulate a complete invoice line. Fields include invoice amounts in
    multiple currencies, dates, payment information, and general ledger mappings.

    Returns:
        dict: A dictionary representing a single invoice transaction record.
    """
    vendor = vendor_master[secrets.randbelow(len(vendor_master) - 1)]

    return {
        "VendorNumber": vendor[3],
        "VendorName": vendor[2],
        "VendorCountry": vendor[1],
        "VendorCountryName": vendor[1],
        "VendorCategory": vendor[8],
        "OneTimeVendorFlag": get_boolean_string(10),
        "VendorCurrency": vendor[4],
        "CPIScore": secrets.randbelow(100),
        "CPIRanking": secrets.randbelow(150),
        "CompanyCode": vendor[0],
        "CompanyCodeName": vendor[0],
        "TransactionType": "",
        "PhysicalInvoiceNo": str(
            secrets.randbelow(100000)
        ),  # fake.random_number(digits=5),
        "FiscalYear": (2020 + secrets.randbelow(3)),
        "RefNo": str(secrets.randbelow(100000)),  # fake.random_number(digits=5),
        "SystemInvoiceNo": str(
            secrets.randbelow(100000)
        ),  # fake.random_number(digits=5),
        "InvoiceStatus": np.random.choice(transaction_master["status"]),
        "AmountVat_VC": secrets.randbelow(10000),
        "AmountIncl_VC": secrets.randbelow(5000000),
        "AmountExcl_VC": secrets.randbelow(5000000),
        "AmountVat_RC": secrets.randbelow(10000),
        "AmountIncl_RC": secrets.randbelow(5000000),
        "AmountExcl_RC": secrets.randbelow(5000000),
        "InvoiceDescription": fake.sentence(nb_words=15, variable_nb_words=True),
        "DebitCreditIndicator": np.random.choice(transaction_master["credit_debit"]),
        "PostingKey": np.random.choice(transaction_master["posting_keys"]),
        "InvoiceType": np.random.choice(transaction_master["invoice_type"]),
        "ReferenceCurrency": vendor[4],
        "PaymentType": np.random.choice(transaction_master["payment_type"]),
        "CapturedBy": np.random.choice(transaction_master["roles"]),
        "AuthorizedBy": np.random.choice(transaction_master["roles"]),
        "GLIndicator": np.random.choice(transaction_master["general_ledger"][1]),
        "InvoiceLineRef": str(1 + secrets.randbelow(9)),
        "DateDocumented": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "DateCaptured": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "DateAuthorized": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "PostingDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "DueDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "ScheduledPayDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "GLAccountRiskCategory": "",
        "GLAccountDescription": "",
        "GLAccountGroup": "",
        "PaymentOrClearingNumber": str(
            secrets.randbelow(100000)
        ),  # fake.random_number(digits=8),
        "PaymentOrClearingDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "PaymentTerms": np.random.choice(transaction_master["PaymentTerms"]),
        "IntercompanyFlag": get_boolean_string(50),
        "WhiteListedVendorFlag": get_boolean_string(50),
        "ModifiedOn": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "TotalInvoiceAmountRC": secrets.randbelow(10000000),
        "TotalInvoiceAmountVC": secrets.randbelow(10000000),
        "Asso#PR": 0,
        "AssoPRAmount_RC": 0,
        "AssoPRRiskScore": 0,
        "Asso#PO": 0,
        "AssoPOAmount_RC": 0,
        "AssoPORiskScore": 0,
        "Asso#PYMT": 0,
        "AssoPYMTAmount_RC": 0,
        "AssoPYMTRiskScore": 0,
        "LookbackFlag": get_boolean_string(50),
        "Region": fake.state(),
        "IsDuplicateFlagged": get_boolean_string(20),
        "Tests_Failed_Count": fake.random_int(min=0, max=20),
        "Tests_Failed_RiskCategory": "",
        "Tests_Failed": "",
        "Tests_FailedID": "",
        "Overall_Risk_Score": secrets.randbelow(100),
        "Overall_Tran_Risk_Score": secrets.randbelow(100),
        "CumulativeRiskScore": secrets.randbelow(100),
        "PhaseNo": 0,
        "Phase": "",
        "SPT_Score": secrets.randbelow(100),
        "Rank": secrets.randbelow(100),
        "RowNumber": 0,
        "PreviousTranRiskScore": 0,
        "PreviousRiskScore": 0,
        "ML_Risk_score": 0,
        "ML_Prediction": 0,
        "SPT_TablePrimaryKey": "",
        "SPT_InvoiceNumber": "",
        "SPT_VendorNumber": "",
        "SPT_VendorAddress": "",
        "SPT_PR": "",
        "SPT_PO": "",
        "SPT_InvoiceID": "",
        "SPT_RowID": fake.unique.random_number(digits=12),
        "P2PACIN101_Keyword_Translated": get_boolean_string(50),
        "P2PACIN101": get_boolean_string(50),
        "P2PFMIN211": get_boolean_string(50),
        "P2PFMIN212": get_boolean_string(50),
        "P2PFMIN213": get_boolean_string(50),
        "P2PFMIN220": get_boolean_string(50),  # Changed
        "P2PFMIN221": get_boolean_string(50),  # Changed
        "P2PFMIN222": get_boolean_string(50),  # Changed
        "P2PFMIN223": get_boolean_string(50),  # Changed
        "P2PFMIN224": get_boolean_string(50),  # Changed
        "P2PFMIN225": get_boolean_string(50),  # Changed
        "P2PFMIN226": get_boolean_string(50),  # Changed
        "P2PFMIN227": get_boolean_string(50),  # Changed
        "P2PFMIN254": get_boolean_string(50),  # Changed
        "P2PFMIN256": get_boolean_string(50),  # Changed
        "P2PFMIN261": get_boolean_string(50),  # Changed
        "P2PFMIN262": get_boolean_string(50),  # Changed
        "P2PFMIN263": get_boolean_string(50),  # Changed
        "P2PFMIN264": get_boolean_string(50),  # Changed
        "P2PFMIN271": get_boolean_string(50),  # Changed
        "P2PHRIN302": get_boolean_string(50),  # Changed
        "P2PHRIN303": get_boolean_string(50),  # Changed
        "P2PHRIN304": get_boolean_string(50),  # Changed
        "P2PHRIN305": get_boolean_string(50),  # Changed
        "P2PFMIN268": get_boolean_string(50),  # Changed
        "P2PICIN604": get_boolean_string(50),  # Changed
        "P2PICIN611": get_boolean_string(50),  # Changed
        "P2PSTIN990": get_boolean_string(50),  # Changed
        "P2PSTIN990_Min_Threshold": 0.0,
        "P2PSTIN990_Max_Threshold": 0.0,
        "P2PFMIN261_HolidayOrWeekends": "False",
        "P2PFMIN262_RoundedTo": "False",
        "P2PFMIN268_POCurrency": "False",
        "P2PHRIN302_OFACName": "False",
        "P2PHRIN303_PanamaName": "False",
        "P2PHRIN303_JurisdictionDescription": "False",
        "P2PHRIN304_PanamaAddress": "False",
        "P2PHRIN304_Country": "False",
        "P2PFMIN213_DifferenceInDays": 0,
        "P2PFMIN214_DifferenceInDays": 0,
        "P2PICIN604_DifferenceInDays": 0,
        "P2PFMIN220_GRD": 0,
        "P2PFMIN221_GRD": 0,
        "P2PFMIN222_GRD": 0,
        "P2PFMIN223_GRD": 0,
        "P2PFMIN224_GRD": 0,
        "P2PFMIN225_GRD": 0,
        "P2PFMIN226_GRD": 0,
        "P2PFMIN227_GRD": 0,
        "P2PFMIN254_GRD": 0,
        "P2PACIN101_Tran_Score": 0.0,
        "P2PFMIN211_Tran_Score": 0.0,
        "P2PFMIN212_Tran_Score": 0.0,
        "P2PFMIN213_Tran_Score": 0.0,
        "P2PFMIN214_Tran_Score": 0.0,
        "P2PFMIN220_Tran_Score": 0.0,
        "P2PFMIN221_Tran_Score": 0.0,
        "P2PFMIN222_Tran_Score": 0.0,
        "P2PFMIN223_Tran_Score": 0.0,
        "P2PFMIN224_Tran_Score": 0.0,
        "P2PFMIN225_Tran_Score": 0.0,
        "P2PFMIN226_Tran_Score": 0.0,
        "P2PFMIN227_Tran_Score": 0.0,
        "P2PFMIN254_Tran_Score": 0.0,
        "P2PFMIN256_Tran_Score": 0.0,
        "P2PFMIN261_Tran_Score": 0.0,
        "P2PFMIN262_Tran_Score": 0.0,
        "P2PFMIN263_Tran_Score": 0.0,
        "P2PFMIN264_Tran_Score": 0.0,
        "P2PFMIN271_Tran_Score": 0.0,
        "P2PFMIN268_Tran_Score": 0.0,
        "P2PHRIN302_Tran_Score": 0.0,
        "P2PHRIN303_Tran_Score": 0.0,
        "P2PHRIN304_Tran_Score": 0.0,
        "P2PHRIN305_Tran_Score": 0.0,
        "P2PICIN611_Tran_Score": 0.0,
        "P2PICIN604_Tran_Score": 0.0,
        "P2PSTIN990_Tran_Score": 0.0,
    }


def generate_invoice_dataframe(records=10):
    """
    Generate a Dask dataframe of synthetic invoice transactions with features and labels.

    This function creates a collection of randomized invoice records using
    `generate_transaction_data()`, augments them with engineered core features,
    introduces null values, and returns the result as a computed Dask dataframe.

    Args:
    -----
        records (int, optional): Number of invoice records to generate. Defaults to 10.

    Returns:
        tuple:
            pd.DataFrame: Feature dataframe with synthetic invoice records.
            pd.Series: Binary label series derived from the 'target' core feature.
    """
    # Create a list of delayed objects
    delayed_data = [dask.delayed(generate_transaction_data)() for _ in range(records)]
    delayed_data = dask.delayed(list)(delayed_data)
    delayed_data = dask.delayed(pd.DataFrame)(delayed_data)

    # Create a meta dataframe for the delayed dataframe
    meta = pd.DataFrame(generate_transaction_data(), index=[0])
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
    cols_to_null = ["DebitCreditIndicator", "PaymentType", "PaymentTerms"]
    df = df.map_partitions(
        set_random_to_null, fraction=0.1, include_columns=cols_to_null
    )

    # set index columns
    index_col = config.get("DATA", "INDEX")
    df[index_col] = df.index
    # df = df.set_index(index_col, drop=False)
    df = df.reset_index(drop=True)

    return df, y


if __name__ == "__main__":
    # generate dataframe with multiple of 100 records
    generate_invoice_dataframe(records=1000)
