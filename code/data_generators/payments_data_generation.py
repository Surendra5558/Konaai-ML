# # Copyright (C) KonaAI - All Rights Reserved
"""This file is used to generate transaction data for the project."""
import secrets
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
from data_generators.generate_master import generate_transaction_master
from data_generators.generate_master import generate_vendor_master
from data_generators.utils import config
from data_generators.utils import set_random_to_null
from faker import Faker
from src.tools.dask_tools import compute

seed_value = secrets.randbelow(1000)
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
    Generate a single synthetic payment transaction record.

    The transaction contains a variety of vendor-related fields including:
        - Vendor identifiers and addresses
        - Financial values (VAT, totals, invoice amounts)
        - Payment and invoice metadata
        - Bank and GL account info
        - Document dates and statuses

    Returns:
        dict: A dictionary representing one synthetic payment transaction.
    """

    vendor = vendor_master[secrets.randbelow(len(vendor_master) - 1)]

    return {
        "VendorNumber": vendor[3],
        "VendorName": vendor[2],
        "VendorCountry": vendor[1],
        "VendorCountryName": vendor[1],
        "VendorCountryCurrency": vendor[4],
        "VendorCategory": vendor[8],
        "OneTimeBendorFlag": get_boolean_string(10),
        "CPIScore": secrets.randbelow(100),
        "CPIRank": secrets.randbelow(100),
        "COITag:": secrets.randbelow(100),
        "CompanyCode": vendor[0],
        "CompanyCodeName": vendor[0],
        "SystemPaymentNo": str(secrets.randbelow(100000)),
        "AdditionalPaymentsID": str(secrets.randbelow(100000)),
        "PhysicalInvoiceNo": str(secrets.randbelow(100000)),
        "SystemInvoiceNo": str(secrets.randbelow(100000)),
        "FiscalYear": (2020 + secrets.randbelow(3)),
        "InvoiceLineRef": str(secrets.randbelow(100000)),
        "InvoiceType": np.random.choice(transaction_master["invoice_type"]),
        "PostingKey": np.random.choice(transaction_master["posting_keys"]),
        "PaymentDescription": fake.sentence(nb_words=15, variable_nb_words=True),
        "DebitCreditIndicator": np.random.choice(transaction_master["credit_debit"]),
        "AmountVat_RC": secrets.randbelow(10000),
        "AmountIncl_RC": secrets.randbelow(5000000),
        "AmountExcl_RC": secrets.randbelow(5000000),
        "VendorCurrency": vendor[4],
        "AmountVat_VC": secrets.randbelow(10000),
        "AmountIncl_VC": secrets.randbelow(5000000),
        "AmountExcl_VC": secrets.randbelow(5000000),
        "CapturedBy": np.random.choice(transaction_master["roles"]),
        "AuthorizedBy": np.random.choice(transaction_master["roles"]),
        "GLAccountRiskCategory": "",
        "GLAccountDescription": "",
        "GLAccountGroup": "",
        "RefNo": str(secrets.randbelow(100000)),
        "PaymentTerms": np.random.choice(transaction_master["PaymentTerms"]),
        "PaymentMethod": np.random.choice(transaction_master["payment_method"]),
        "IsManualPayment": get_boolean_string(20),
        "PayeeName": vendor[2],
        "PayeeAddress": vendor[7],
        "PayeeCity": vendor[5],
        "PayeeCountry": vendor[1],
        "PayeeRegion": vendor[6],  # doubtful
        "PayeePostalCode": str(fake.random_number(digits=4)),
        "PayeeBankNo": str(fake.random_number(digits=10)),
        "PayeeBankAccount": str(fake.random_number(digits=15)),
        "PayeeBankCountry": vendor[1],
        "PayeeSWIFT": "",
        "PayeeIBAN": (
            f'{fake.bothify(text="????").upper()}{fake.random_number(digits=5)}'
        ),
        "InterCompanyFlag": get_boolean_string(10),
        "WhiteListedVendorFlag": get_boolean_string(5),
        "HouseBankID": "",
        "HouseBankKey": "",
        "PhaseNo": 0,
        "Phase": "",
        "SPT_Score": secrets.randbelow(100),
        "SPT_VendorAddress": "",
        "SPT_VendorTaxID": "",
        "SPT_VendorContacts": "",
        "SPT_VendorEmail": "",
        "SPT_VendorBankAccounts": "",
        "SPT_PR": "",
        "SPT_PO": "",
        "SPT_InvoiceID": "",
        "TotalInvoiceAmountRC": secrets.randbelow(5000000),
        "TotalInvoiceAmountVC": secrets.randbelow(5000000),
        "SPT_PaymentID": "",
        "Asso#PR": 0,
        "AssoPRAmount_RC": 0.0,
        "AssoPRRiskScore": 0.0,
        "Asso#PO": 0,
        "AssoPOAmount_RC": 0.0,
        "AssoPORiskScore": 0.0,
        "Asso#INV": 0,
        "AssoINVAmount_RC": 0.0,
        "AssoINVRiskScore": 0.0,
        "Region": "",
        "LookbackFlag": get_boolean_string(10),
        "IsDuplicateFlagged": get_boolean_string(5),
        "ML_Prediction": 0,
        "ML_Risk_score": 0.0,
        "Tests_Failed": "",
        "Tests_Failed_RiskCategory": "",
        "Tests_Failed_Count": fake.random_int(min=0, max=20),
        "Tests_FailedID": 0,
        "Overall_Risk_Score": secrets.randbelow(100),
        "Overall_Tran_Risk_Score": secrets.randbelow(100),
        "PreviousRiskScore": secrets.randbelow(100),
        "PreviousTranRiskScore": secrets.randbelow(100),
        "CumulativeRiskScore": secrets.randbelow(100),
        "RowNumber": 0,
        "Rank": secrets.randbelow(100),
        "SPT_Reversal_Flag": "",
        "SPT_Reversal_Group": 0,
        "P2PACPY101": get_boolean_string(10),
        "P2PACPY101_Keyword_Translated": get_boolean_string(10),
        "P2PACPY101_Keyword_Category": get_boolean_string(10),
        "P2PACPY102": get_boolean_string(10),
        "P2PACPY102_GovernmentKeyword": get_boolean_string(10),
        "P2PCIPY146": get_boolean_string(10),
        "P2PCIPY146_MatchingEmployeeAdd": get_boolean_string(10),
        "P2PCIPY146_MatchingAddEmployees": get_boolean_string(10),
        "P2PCIPY151": get_boolean_string(10),
        "P2PCIPY151_GRD": secrets.randbelow(100),
        "P2PCIPY151_COIType": get_boolean_string(10),
        "P2PCIPY152": get_boolean_string(10),
        "P2PCIPY152_GRD": secrets.randbelow(100),
        "P2PCIPY152_COIType": get_boolean_string(10),
        "P2PCIPY153": get_boolean_string(10),
        "P2PCIPY153_GRD": secrets.randbelow(100),
        "P2PFMPY215_VendorCreationDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "P2PFMPY251": get_boolean_string(10),
        "P2PFMPY251_PODate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "P2PFMPY251_DifferenceInDays": secrets.randbelow(100),
        "P2PFMPY252": get_boolean_string(10),
        "P2PFMPY253": get_boolean_string(10),
        "P2PFMPY253_GRD": 0,
        "P2PFMPY255": get_boolean_string(10),
        "P2PFMPY256": get_boolean_string(10),
        "P2PFMPY256_Keyword": "",
        "P2PFMPY257": get_boolean_string(10),
        "P2PFMPY261": get_boolean_string(10),
        "P2PFMPY261_HolidayOrWeekends": "",
        "P2PFMPY262": get_boolean_string(10),
        "P2PFMPY262_RoundedTo": "",
        "P2PFMPY263": get_boolean_string(10),
        "P2PFMPY263_GRD": 0,
        "P2PFMPY264": get_boolean_string(10),
        "P2PFMPY264_GRD": 0,
        "P2PFMPY265": get_boolean_string(10),
        "P2PFMPY266": get_boolean_string(10),
        "P2PFMPY267": get_boolean_string(10),
        "P2PFMPY267_MismatchedVendorAdd": "",
        "P2PFMPY268": get_boolean_string(10),
        "P2PFMPY271": get_boolean_string(10),
        "P2PFMPY271_GRD": 0,
        "P2PFMPY272": get_boolean_string(10),
        "P2PFMPY273": get_boolean_string(10),
        "P2PFMPY273_GRD": 0,
        "P2PFMPY274": get_boolean_string(10),
        "P2PFMPY275": get_boolean_string(10),
        "P2PFMPY603": get_boolean_string(10),
        "P2PFMPY603_GRD": 0,
        "P2PFMPY603_TotalInvoiceAmount": secrets.randbelow(5000000),
        "P2PFMPY603_TotalPaymentAmount": secrets.randbelow(5000000),
        "P2PHRPY301": get_boolean_string(10),
        "P2PHRPY302": get_boolean_string(10),
        "P2PHRPY302_SanctionedProgram": "",
        "P2PHRPY302_OFACName": "",
        "P2PHRPY303": get_boolean_string(10),
        "P2PHRPY303_PanamaName": "",
        "P2PHRPY303_JurisdictionDescription": "",
        "P2PHRPY304": get_boolean_string(10),
        "P2PHRPY304_PanamaAddress": "",
        "P2PHRPY304_Country": "",
        "P2PHRPY305": get_boolean_string(10),
        "P2PHRPY305_AdditionalEmbargoDetails": "",
        "P2PHRPY305_IssuingAuthority": "",
        "P2PHRPY305_TypeandScopeofEmbargoorSanction": "",
        "P2PICPY601": get_boolean_string(10),
        "P2PICPY602": get_boolean_string(10),
        "P2PICPY602_GRD": 0,
        "P2PICPY602_TotalInvoiceAmount": secrets.randbelow(5000000),
        "P2PICPY602_TotalPaymentAmount": secrets.randbelow(5000000),
        "P2PICPY605": get_boolean_string(10),
        "P2PPVPY741": get_boolean_string(10),
        "P2PPVPY741_GRD": 0,
        "P2PCRPY220": get_boolean_string(10),
        "P2PCRPY220_GRD": 0,
        "P2PCRPY221": get_boolean_string(10),
        "P2PCRPY221_GRD": 0,
        "P2PCRPY222": get_boolean_string(10),
        "P2PCRPY222_GRD": 0,
        "P2PCRPY223": get_boolean_string(10),
        "P2PCRPY223_GRD": 0,
        "P2PCRPY224": get_boolean_string(10),
        "P2PCRPY224_GRD": 0,
        "P2PCRPY225": get_boolean_string(10),
        "P2PCRPY225_GRD": 0,
        "P2PCRPY226": get_boolean_string(10),
        "P2PCRPY226_GRD": 0,
        "P2PCRPY227": get_boolean_string(10),
        "P2PCRPY227_GRD": 0,
        "P2PCRPY228": get_boolean_string(10),
        "P2PSTPY990": get_boolean_string(10),
        "P2PSTIN990_Min_Threshold": 0.0,
        "P2PSTIN990_Max_Threshold": 0.0,
        "P2PACPY101_Tran_Score": 0.0,
        "P2PACPY102_Tran_Score": 0.0,
        "P2PCIPY146_Tran_Score": 0.0,
        "P2PCIPY151_Tran_Score": 0.0,
        "P2PCIPY152_Tran_Score": 0.0,
        "P2PFMPY005_Tran_Score": 0.0,
        "P2PFMPY215_Tran_Score": 0.0,
        "P2PFMPY251_Tran_Score": 0.0,
        "P2PFMPY252_Tran_Score": 0.0,
        "P2PFMPY253_Tran_Score": 0.0,
        "P2PFMPY255_Tran_Score": 0.0,
        "P2PFMPY256_Tran_Score": 0.0,
        "P2PFMPY257_Tran_Score": 0.0,
        "P2PFMPY261_Tran_Score": 0.0,
        "P2PFMPY262_Tran_Score": 0.0,
        "P2PFMPY264_Tran_Score": 0.0,
        "P2PFMPY266_Tran_Score": 0.0,
        "P2PFMPY267_Tran_Score": 0.0,
        "P2PFMPY268_Tran_Score": 0.0,
        "P2PFMPY271_Tran_Score": 0.0,
        "P2PFMPY272_Tran_Score": 0.0,
        "P2PFMPY273_Tran_Score": 0.0,
        "P2PHRPY301_Tran_Score": 0.0,
        "P2PHRPY302_Tran_Score": 0.0,
        "P2PHRPY303_Tran_Score": 0.0,
        "P2PHRPY304_Tran_Score": 0.0,
        "P2PHRPY305_Tran_Score": 0.0,
        "P2PICPY601_Tran_Score": 0.0,
        "P2PICPY602_Tran_Score": 0.0,
        "P2PICPY605_Tran_Score": 0.0,
        "P2PPVPY741_Tran_Score": 0.0,
        "P2PCRPY220_Tran_Score": 0.0,
        "P2PCRPY221_Tran_Score": 0.0,
        "P2PCRPY222_Tran_Score": 0.0,
        "P2PCRPY223_Tran_Score": 0.0,
        "P2PCRPY224_Tran_Score": 0.0,
        "P2PCRPY225_Tran_Score": 0.0,
        "P2PCRPY226_Tran_Score": 0.0,
        "P2PCRPY227_Tran_Score": 0.0,
        "P2PSTPY990_Tran_Score": 0.0,
        "P2PFMPY270": get_boolean_string(10),
        "P2PFMPY270_Tran_Score": 0.0,
        "P2PFMPY270_AccountNumber": "",
        "PaymentRunDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "PaymentOrClearingDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "PostingDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "DateDocumented": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "DateAuthorized": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "ModifiedOn": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
    }


def generate_payment_dataframe(records=10):
    """Generate payment dataframe"""
    # Genrate data directly without deplayed objects
    data = [generate_transaction_data() for _ in range(records)]
    df = pd.DataFrame(data)

    date_columns = [
        "P2PFMPY215_VendorCreationDate",
        "P2PFMPY251_PODate",
        "PaymentRunDate",
        "PostingDate",
        "DateDocumented",
        "PaymentOrClearingDate",
        "DateAuthorized",
        "ModifiedOn",
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = dd.from_pandas(df, npartitions=4)

    # set index columns
    index_col = config.get("DATA", "INDEX")
    df[index_col] = df.index
    df = df.reset_index(drop=True)
    # introduce null values
    df = df.map_partitions(set_random_to_null, fraction=0.1)

    # set index columns
    index_col = config.get("DATA", "INDEX")
    df[index_col] = df.index
    # df = df.set_index(index_col)
    df = df.reset_index(drop=True)

    return compute(df)


if __name__ == "__main__":
    # generate dataframe with multiple of 100 records
    generate_payment_dataframe(records=1000)
