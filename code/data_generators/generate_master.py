# # Copyright (C) KonaAI - All Rights Reserved
"""This module generates master data for the transaction and vendor data"""
import secrets
from datetime import datetime

import pycountry
from data_generators.utils import config
from faker import Faker

fake = Faker()
seed_value = int(config.get("DATA", "RANDOM_STATE"))
fake.seed_instance(seed_value)

start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-11-22", "%Y-%m-%d")


def generate_transaction_master(records=10):
    """
    Generate master data for transaction records.
    This function returns a dictionary containing various master data fields relevant to transaction records,
    such as transaction status, payment types, document types, business units, and more. The generated data
    can be used for testing, data generation, or as reference master data in financial or procurement systems.
    Args:
    ----
        records (int, optional): Number of records to generate for fields that require random values (e.g., general ledger).
            Defaults to 10.
            
    Returns:
        dict: A dictionary with keys representing transaction master data fields and values as lists of possible values
            or randomly generated data for those fields.
    """

    return {
        "date": fake.date_between_dates(date_start=start_date, date_end=end_date),
        "credit_debit": ["H-Credit", "S-Debit"],
        "status": [
            "Draft",
            "Pending Approval",
            "Approved",
            "Rejected",
            "Pending Payment",
            "Paid",
            "Partially Paid",
            "On Hold",
            "Pending Reconciliation",
            "Closed",
            "Completed",
            "Declined",
            "Authorized",
            "Awaiting Confirmation",
            "In Progress",
            "Settled",
            "Verified",
            "Posted",
            "Rejected",
            "Cancelled",
            "Disbursed",
            "Cleared",
            "Refunded",
            "Held for Review",
            "Partially Paid",
            "Sent for Approval",
            "Awaiting Payment",
        ],
        "po_type": [
            "SPO - Standard Purchase Order (PO)",
            "BPO - Blanket Purchase Order",
            "CPO - Contract Purchase Order",
            "SPO - Scheduled Purchase Order",
            "PPO - Planned Purchase Order",
            "FPO - Framework Purchase Order",
            "EPO - Emergency Purchase Order",
            "TPO - Turnkey Purchase Order",
            "DPO - Drop-Ship Purchase Order",
            "CPO - Consignment Purchase Order",
            "DPO - Direct Purchase Order",
            "IPO - Internal Purchase Order",
            "SPO - Spot Buy Purchase Order",
            "RPO - Recurring Purchase Order",
            "SPO - Service Purchase Order",
        ],
        "posting_keys": [
            "31-Invoice",
            "21-Credit memo",
            "22-Reverse invoice",
            "32-Reverse credit memo",
        ],
        "invoice_type": [
            "RE-Invoice - Gross",
            "KR-Vendor Invoice",
            "ZI-IC EDI Invoice",
            "KG-Vendor Credit Memo",
            "KZ-Vendor Payment",
            "ZD-PCard Interface",
            "ZS-Serengeti Interface",
            "CS-CO Posting Secondary",
            "KA-Vendor Document",
            "IH-IHC Bank Statements",
            "ZZ-Cutover Postings",
            "AB-Accounting Document",
        ],
        "payment_type": [
            "Wire Transfer",
            "Credit Card",
            "Check",
            "ACH (Automated Clearing House) Payment",
            "EFT (Electronic Funds Transfer)",
            "Mobile Payments",
            "PayPal",
            "Direct Debit",
            "Cash",
            "Letter of Credit",
            "Bill of Exchange",
            "Cryptocurrency",
            "Money Order",
            "Promissory Note",
            "Standing Order",
            "Barter",
            "Escrow",
            "Prepaid Card",
            "Cryptocurrency Wallet Transfer",
            "Money Transfer Services",
        ],
        "document_type": [
            "KC",
            "RE",
            "KR",
            "RC",
            "KA",
            "ZL",
        ],
        "general_ledger": list(
            zip(
                [fake.bothify(text="????").upper() for _ in range(records)],
                [fake.random_number(digits=6) for _ in range(records)],
            )
        ),
        "roles": [
            "Accounts Payable Clerk",
            "Data Entry Specialist",
            "Invoice Processor",
            "Automated System",
            "Vendor Portal",
            "EDI System",
            "Procurement Officer",
            "Receiving Department",
            "Integrated ERP System",
            "Contract Administrator",
        ],
        "inco_terms": [
            "EXW - Ex Works",
            "FOB - Free On Board",
            "CIF - Cost, Insurance, and Freight",
            "DDP - Delivered Duty Paid",
            "DAP - Delivered At Place",
            "CIP - Carriage and Insurance Paid To",
            "DAT - Delivered At Terminal",
            "FCA - Free Carrier",
            "CFR - Cost and Freight",
            "DDU - Delivered Duty Unpaid",
            "FAS - Free Alongside Ship",
            "DEQ - Delivered Ex Quay",
            "CPT - Carriage Paid To",
            "DDU - Delivered Duty Unpaid",
            "DDP - Delivered Duty Paid",
        ],
        "business_unit": [
            "Finance Department",
            "Marketing Division",
            "Sales Team",
            "Operations Unit",
            "Human Resources",
            "IT Department",
            "Research and Development",
            "Customer Service",
            "Logistics Division",
            "Product Management",
            "Legal Department",
            "Supply Chain",
            "Quality Assurance",
            "Administration",
            "Strategic Planning",
        ],
        "accounting_assignment_category": [
            "Expense",
            "Capital Expenditure",
            "General Overhead",
            "Cost of Goods Sold (COGS)",
            "Research and Development",
            "Marketing Expenses",
            "Administrative Costs",
            "Utilities",
            "Travel and Entertainment",
            "IT Expenses",
            "Legal and Compliance",
            "Training Costs",
            "Maintenance",
            "Insurance",
            "Taxes",
        ],
        "difference_category": [
            "Price Discrepancy",
            "Quantity Discrepancy",
            "Delivery Date Discrepancy",
            "Quality Discrepancy",
            "Supplier Information Discrepancy",
            "Currency Discrepancy",
            "Unit of Measure Discrepancy",
            "Payment Terms Discrepancy",
            "Tax Discrepancy",
            "Customs Duty Discrepancy",
            "Specification Discrepancy",
            "Discount Discrepancy",
            "Shipping Method Discrepancy",
            "Billing Address Discrepancy",
            "Vendor Evaluation Discrepancy",
        ],
        "payment_method": [
            "Wire Transfer",
            "Credit Card",
            "Check",
            "ACH (Automated Clearing House) Payment",
            "EFT (Electronic Funds Transfer)",
            "Mobile Payments",
            "PayPal",
            "Direct Debit",
            "Cash",
            "Letter of Credit",
            "Bill of Exchange",
            "Cryptocurrency",
            "Money Order",
            "Promissory Note",
            "Standing Order",
            "Barter",
            "Escrow",
            "Prepaid Card",
            "Cryptocurrency Wallet Transfer",
            "Money Transfer Services",
        ],
        "cost_center": [
            "Marketing",
            "Research and Development",
            "Operations",
            "Sales",
            "Administration",
        ],
        "LineItemDescription": [
            "Consulting Services",
            "Software License",
            "Hardware Purchase",
            "Office Supplies",
            "Travel Expenses",
            "Training Costs",
            "Legal Fees",
            "Marketing Expenses",
            "Utilities",
            "Rent",
            "Insurance",
            "Maintenance",
            "Taxes",
            "Shipping Costs",
            "Professional Services",
            "Freight Charges",
            "Customs Duty",
            "IT Services",
            "Telecommunications",
            "Advertising",
            "Printing and Stationery",
            "Subscriptions",
            "Recruitment Costs",
            "Employee Benefits",
            "Utilities",
            "Equipment Rental",
        ],
        "Approval Status": [
            "Approved",
            "Pending Approval",
            "Rejected",
            "Pending Payment",
            "Not Submitted",
            "Sent back to Employee",
            "Pending Review",
        ],
        "Transaction Line Type": [
            "Postage",
            "Meals",
            "Business Meals",
            "Valet/Tips",
            "Employee Appreciation",
            "Room Tax",
            "Parking & Tolls",
            "Hotel",
            "Office Supplies",
            "Other Travel Expense",
            "Postage",
            "Bank Fees",
            "Recruiting (HR Only)",
            "Taxi",
            "Rental / Fleet Car: Gas",
            "Customer Entertainment - Deductible",
            "Company Car: Gas/Car Mntnc/Car Wash/Registration",
            "Airfare",
        ],
        "PaymentTerms": [
            "Net 30",
            "Net 60",
            "Net 90",
            "Due on Receipt",
            "End of Month",
            "Weekly",
            "Bi-Weekly",
            "Monthly",
            "Quarterly",
            "Annually",
        ],
    }


def generate_vendor_master(records=10):
    """Generate master data for vendors."""

    # Get a list of all available countries with ISO 4217 currency codes
    all_countries = list(pycountry.countries)
    country_currency = {country.name: country.alpha_3 for country in all_countries}

    company_codes = [fake.bothify(text="??").upper() for _ in range(records)]
    countries = [
        list(country_currency.keys())[secrets.randbelow(len(country_currency) - 1)]
        for _ in range(records)
    ]
    vendor_name = [fake.name() for _ in range(records)]
    vendor_number = [fake.random_number(digits=5) for _ in range(records)]
    currencies = [country_currency[country] for country in countries]
    cities = [fake.city() for _ in range(records)]
    states = [fake.state() for _ in range(records)]
    addresses = [fake.address() for _ in range(records)]
    vendor_category = [
        "Raw Material Supplier",
        "Manufacturing Partner",
        "Logistics Provider",
        "Service Provider",
        "Technology Vendor",
        "Consulting Firm",
        "Retail Supplier",
    ]

    return list(
        zip(
            company_codes,
            countries,
            vendor_name,
            vendor_number,
            currencies,
            cities,
            states,
            addresses,
            vendor_category,
        )
    )
