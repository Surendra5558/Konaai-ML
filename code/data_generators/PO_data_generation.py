# # Copyright (C) KonaAI - All Rights Reserved
"""This module generates PO data"""
import secrets
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
from data_generators.generate_master import generate_transaction_master
from data_generators.generate_master import generate_vendor_master
from data_generators.utils import config
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
    Generate a single synthetic purchase order (PO) transaction record.

    The returned record includes various PO-related fields such as:
        - Quantity, price, material details
        - Vendor metadata
        - Dates of GRN, PO, PR, approvals, etc.
        - Inco terms, statuses, payment terms, and assignment categories

    Returns:
        dict: A dictionary containing one complete synthetic PO transaction.
    """
    vendor = vendor_master[secrets.randbelow(len(vendor_master) - 1)]

    return {
        "VendorNumber": vendor[3],
        "VendorName": vendor[2],
        "VendorCountry": vendor[1],
        "VendorCountryName": vendor[1],
        "VendorCategory": vendor[8],
        "OneTimeVendorFlag": get_boolean_string(10),
        "CPIScore": secrets.randbelow(100),
        "CPIRank": secrets.randbelow(100),
        "COITag": "",
        "CompanyCode": vendor[0],
        "CompanyCodeName": vendor[0],
        "Plant": "",
        "PlantName": "",
        "PONumber": str(secrets.randbelow(100000)),
        "POLineNo": str(secrets.randbelow(10000)),
        "PODescription": fake.sentence(nb_words=10, variable_nb_words=True),
        "POType": np.random.choice(transaction_master["po_type"]),
        "POStatus": "",
        "MaterialCode": str(secrets.randbelow(100000)),
        "MaterialDescription": fake.sentence(nb_words=5, variable_nb_words=True),
        "MaterialGroup": "",
        "DeletionIndicator": "",
        "PODate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "PORaisedBy": "",
        "Changedon": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "ChangedBy": "",
        "POApprovedOn": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "POApprovedBy": "",
        "UOM": "",
        "Quantity": secrets.randbelow(1000),
        "QuantityToBeDelivered": secrets.randbelow(1000),
        "PricePerUnit": secrets.randbelow(1000),
        "PricePerUnitRC": secrets.randbelow(1000),  # Random conversion rate
        "VendorCurrency": vendor[4],
        "AmountVC": secrets.randbelow(5000000),
        "AmountRC": secrets.randbelow(5000000),  # Random conversion rate
        "PODeliveryDueDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "IncoTerms": np.random.choice(transaction_master["inco_terms"]),
        "PurchasingOrg": "",
        "PurchasingGroup": "",
        "PRNumber": str(f"PR{secrets.randbelow(100000):05d}"),
        "PRDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "PRLineitem": "",
        "PRAmount": secrets.randbelow(100000),
        "PRCreatedBy": "",
        "PRApprovedBy": "",
        "PaymentTerm": np.random.choice(transaction_master["PaymentTerms"]),
        "GLAccountRiskCategory": "",
        "GLAccountDescription": "",
        "GLAccountGroup": "",
        "AccountingAssignmentCategory": "",
        "AccountingCategoryDescription": fake.sentence(
            nb_words=5, variable_nb_words=True
        ),
        "RFQStatus": "",
        "GRNNumber": "",
        "GRNMaterialCode": "",
        "GRNDate": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "GRNQuantity": "",
        "GRNCreatedby": "",
        "GRNAmountIncl_LC": secrets.randbelow(100000),
        "GRNAmount": secrets.randbelow(100000),
        "IntercompanyFlag": get_boolean_string(10),
        "WhiteListedVendorFlag": get_boolean_string(5),
        "ExcludeCancelledPOFlag": get_boolean_string(5),
        "ModifiedOn": fake.date_between_dates(
            date_start=start_date, date_end=end_date
        ).strftime("%Y-%m-%d"),
        "Phase": "",
        "PhaseNo": "",
        "SPT_Source": "",
        "SPT_RowId": 0,
        "SPT_TablePrimaryKey": "",
        "SPT_PR": "",
        "SPT_PO": "",
        "TotalPOAmountVC": secrets.randbelow(5000000),
        "TotalPOAmountRC": secrets.randbelow(5000000),
        "SPT_InvoiceID": "",
        "SPT_PaymentID": "",
        "SPT_Material_Code": "",
        "SPT_VendorNumber": vendor[3],
        "SPT_VendorAddress": vendor[7],
        "SPT_VendorBankAccounts": "",
        "SPT_VendorContacts": "",
        "SPT_VendorTaxID": "",
        "SPT_VendorEmail": "",
        "Asso#PR": 0,
        "Asso$PR": 0.0,
        "AssoPRRiskScore": 0.0,
        "Asso#INV": 0,
        "Asso$INV": 0.0,
        "AssoINVRiskScore": 0.0,
        "Asso#PYMT": 0,
        "Asso$PYMT": 0.0,
        "AssoPYMTRiskScore": 0.0,
        "LookbackFlag": get_boolean_string(10),
        "Region": "",
        "IsDuplicateFlagged": get_boolean_string(5),
        "ML_Prediction": 0,
        "ML_Risk_Score": 0.0,
        "Tests_Failed": "",
        "Tests_Failed_RiskCategory": "",
        "Tests_Failed_Count": fake.random_int(min=0, max=20),
        "Tests_FailedID": 0,
        "Overall_Risk_Score": secrets.randbelow(100),
        "Overall_Tran_Risk_Score": secrets.randbelow(100),
        "PreviousRiskScore": secrets.randbelow(100),
        "PreviousTranRiskScore": secrets.randbelow(100),
        "CummulativeRiskScore": secrets.randbelow(100),
        "RowNumber": 0,
        "Rank": secrets.randbelow(100),
        "P2PACPO101": get_boolean_string(10),
        "P2PACPO101_Keyword_Translated": get_boolean_string(10),
        "P2ACPCO101_Keyword_Category": get_boolean_string(10),
        "P2PACPO102": get_boolean_string(10),
        "P2PACPO102_GovernmentKeyword": get_boolean_string(10),
        "P2PFMPO742": get_boolean_string(10),
        "P2PFMPO742_GRD": secrets.randbelow(100),
        "P2PFMPO241": get_boolean_string(10),
        "P2PFMPO241_GRD": secrets.randbelow(100),
        "P2PFMPO241_PRVendorNo": vendor[3],
        "P2PFMPO242": get_boolean_string(10),
        "P2PFMPO242_GRD": secrets.randbelow(100),
        "P2PFMPO242_INVVendorNo": vendor[3],
        "P2PFMPO743": get_boolean_string(10),
        "P2PFMPO743_GRD": secrets.randbelow(100),
        "P2PFMPO244": get_boolean_string(10),
        "P2PFMPO244_GRD": secrets.randbelow(100),
        "P2PFMPO264": get_boolean_string(10),
        "P2PFMPO256": get_boolean_string(10),
        "P2PFMPO256_Keyword": "",
        "P2PFMPO261": get_boolean_string(10),
        "P2PFMPO261_HolidayOrWeekends": "",
        "P2PHRPO302": get_boolean_string(10),
        "P2PHRPO302_OFACName": get_boolean_string(10),
        "P2PHRPO302_SanctionedProgram": get_boolean_string(10),
        "P2PHRPO303": get_boolean_string(10),
        "P2PHRPO303_PanamaName": get_boolean_string(10),
        "P2PHRPO303_JurisdictionDescription": get_boolean_string(10),
        "P2PHRPO304": get_boolean_string(10),
        "P2PHRPO304_PanamaAddress": get_boolean_string(10),
        "P2PHRPO304_Country": get_boolean_string(10),
        "P2PHRPO305": get_boolean_string(10),
        "P2PHRPO305_AdditionalEmbargoDetails": get_boolean_string(10),
        "P2PHRPO305_IssuingAuthority": get_boolean_string(10),
        "P2PHRPO305_TypeandScopeofEmbargoorSanction": get_boolean_string(10),
        "P2PPVPO730": get_boolean_string(10),
        "P2PPVPO730_GRD": secrets.randbelow(100),
        "P2PPVPO731": get_boolean_string(10),
        "P2PPVPO731_GRD": secrets.randbelow(100),
        "P2PPVPO732": get_boolean_string(10),
        "P2PPVPO732_GRD": secrets.randbelow(100),
        "P2PPVPO733": get_boolean_string(10),
        "P2PPVPO733_GRD": secrets.randbelow(100),
        "P2PPVPO734": get_boolean_string(10),
        "P2PPVPO734_GRD": secrets.randbelow(100),
        "P2PPVPO735": get_boolean_string(10),
        "P2PPVPO735_GRD": secrets.randbelow(100),
        "P2PPVPO736": get_boolean_string(10),
        "P2PPVPO736_GRD": secrets.randbelow(100),
        "P2PPVPO737": get_boolean_string(10),
        "P2PPVPO737_GRD": secrets.randbelow(100),
        "P2PPVPO738": get_boolean_string(10),
        "P2PPVPO738_GRD": secrets.randbelow(100),
        "P2PPVPO739": get_boolean_string(10),
        "P2PPVPO739_GRD": secrets.randbelow(100),
        "P2PPVPO745": get_boolean_string(10),
        "P2PPVPO745_GRD": secrets.randbelow(100),
        "P2PPVPO740": get_boolean_string(10),
        "P2PPVPO740_GRD": secrets.randbelow(100),
        "P2PFMPO741": get_boolean_string(10),
        "P2PFMPO741_Threshold": "",
        "P2PFMPO741_GRD": secrets.randbelow(100),
        "P2PFMPO744": get_boolean_string(10),
        "P2PFMPO744_GRD": secrets.randbelow(100),
        "P2PSTPO990": get_boolean_string(10),
        "P2PSTPO990_Min_Threshold": 0,
        "P2PSTPO990_Max_Threshold": 0,
        "P2PACPO101_Tran_Score": 0.0,
        "P2PFMPO742_Tran_Score": 0.0,
        "P2PFMPO241_Tran_Score": 0.0,
        "P2PFMPO242_Tran_Score": 0.0,
        "P2PFMPO743_Tran_Score": 0.0,
        "P2PFMPO244_Tran_Score": 0.0,
        "P2PFMPO264_Tran_Score": 0.0,
        "P2PFMPO256_Tran_Score": 0.0,
        "P2PFMPO261_Tran_Score": 0.0,
        "P2PHRPO302_Tran_Score": 0.0,
        "P2PHRPO303_Tran_Score": 0.0,
        "P2PHRPO304_Tran_Score": 0.0,
        "P2PHRPO305_Tran_Score": 0.0,
        "P2PPVPO730_Tran_Score": 0.0,
        "P2PPVPO731_Tran_Score": 0.0,
        "P2PPVPO732_Tran_Score": 0.0,
        "P2PPVPO733_Tran_Score": 0.0,
        "P2PPVPO734_Tran_Score": 0.0,
        "P2PPVPO735_Tran_Score": 0.0,
        "P2PPVPO736_Tran_Score": 0.0,
        "P2PPVPO737_Tran_Score": 0.0,
        "P2PPVPO738_Tran_Score": 0.0,
        "P2PPVPO739_Tran_Score": 0.0,
        "P2PPVPO745_Tran_Score": 0.0,
        "P2PPVPO740_Tran_Score": 0.0,
        "P2PFMPO741_Tran_Score": 0.0,
        "P2PFMPO744_Tran_Score": 0.0,
        "P2PSTPO990_Tran_Score": 0.0,
        "P2PACPO102_Tran_Score": 0.0,
    }


def generate_po_dataframe(records=10):
    """Generate PO dataframe."""
    data = [generate_transaction_data() for _ in range(records)]
    df = pd.DataFrame(data)

    # convert date time columns
    date_columns = [
        "PODate",
        "Changedon",
        "POApprovedOn",
        "PODeliveryDueDate",
        "PRDate",
        "GRNDate",
        "ModifiedOn",
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # convert to dask dataframe
    df = dd.from_pandas(df, npartitions=4)

    # set index columns
    index_col = config.get("DATA", "INDEX")
    df[index_col] = df.index
    # df = df.set_index(index_col)
    df = df.reset_index(drop=True)

    # set index columns

    return compute(df)


if __name__ == "__main__":
    # generate dataframe with multiple of 100 records
    generate_po_dataframe(records=1000)
