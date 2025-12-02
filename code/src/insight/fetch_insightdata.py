# # Copyright (C) KonaAI - All Rights Reserved
"""
This module provides helper functions to collect analytical data
for a given project instance. It integrates data from local JSON
files, project metadata databases, and AutoML experiment trackers.

"""
import configparser
import os

from src.automl.fetch_data import ModelData
from src.automl.model_tracker import ModelTracker
from src.automl.questionnaire import TemplateQuestionnaire
from src.utils.global_config import GlobalSettings
from src.utils.metadata import Metadata
from src.utils.submodule import Submodule


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")

config = configparser.ConfigParser()
config.read(CONFIG_PATH, encoding="utf-8")


def get_submodules(instance_id):
    """
    internal helper to fetch all submodules objects using metadata.
    """
    metadata = Metadata(instance_id)
    modules = metadata.modules
    submodules = []
    for module in modules:
        submodule_names = metadata.get_submodule_names(module)
        for sub_name in submodule_names:
            submodules.append(
                Submodule(instance_id=instance_id, module=module, submodule=sub_name)
            )

    return submodules


def get_amount_count_data(instance_id):
    """
    Aggregate transaction counts and amounts for each submodule.

    For every submodule:
        - Queries the project database for transaction records.
        - Calculates:
            * Total transaction count
            * Total transaction amount
            * Concern transaction count (ML_Prediction == 1)
            * Concern transaction amount

    Args:
        instance_id (str): Unique identifier of the active project instance.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - Module (str)
            - Sub Module (str)
            - RiskTransactionAmountField (str): Name of the amount column
            - SPT_Base_Table (str): Table name
            - total_count (int)
            - Total_amount (float)
            - total_concern_count (int)
            - total_concern_amount (float)
    """
    results = []
    submodules = get_submodules(instance_id)

    for sub in submodules:
        amount_field = sub.get_amount_column()
        table_name = sub.get_data_table_name()
        if not amount_field or not table_name:
            continue

        instance = GlobalSettings().instance_by_id(sub.instance_id)

        # Load query from config.ini
        query_template = config["INSIGHT"]["AMOUNT_COUNT_QUERY"]
        query = query_template.format(amount_field=amount_field, table_name=table_name)

        dff = instance.settings.projectdb.download_table_or_query(query=query)
        if dff is None:
            continue
        df = dff.compute()
        if df.empty:
            continue

        total_count = len(df)
        total_amount = df[amount_field].sum()
        concern_df = df[df["ML_Prediction"] == 1]
        total_concern_count = len(concern_df)
        total_concern_amount = concern_df[amount_field].sum()

        results.append(
            {
                "Module": sub.module,
                "Sub Module": sub.submodule,
                "RiskTransactionAmountField": amount_field,
                "SPT_Base_Table": table_name,
                "total_count": total_count,
                "Total_amount": total_amount,
                "total_concern_count": total_concern_count,
                "total_concern_amount": total_concern_amount,
            }
        )
    return results


def get_top_risk_patterns(instance_id):
    """
    Retrieve top risk test patterns across modules and submodules.

    Executes a SQL query defined in `config.ini` to extract pattern-level
    statistics from the project database.

    Args:
        instance_id (str): Unique identifier of the active project instance.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - Module (str)
            - Sub Module (str)
            - PatternID (str)
            - PatternDescriptionLong (str)
            - KRI (str)
            - TestedCount (int)
            - FailedCount (int)
            - FailedCount_% (float)
    """
    results = []
    instance = GlobalSettings().instance_by_id(instance_id)

    # Load query from config.ini
    query_template = config["INSIGHT"]["RISK_PATTERN_REPORT"]
    query = query_template

    dff = instance.settings.projectdb.download_table_or_query(query=query)
    if dff is None:
        return results
    df = dff.compute()
    if df.empty:
        return results

    for _, row in df.iterrows():
        results.append(
            {
                "Module": row.get("Module"),
                "Sub Module": row.get("SubModule"),
                "PatternID": row.get("PatternID"),
                "PatternDescriptionLong": row.get("PatternDescriptionLong"),
                "KRI": row.get("KRI"),
                "TestedCount": row.get("TestedCount"),
                "FailedCount": row.get("FailedCount"),
                "FailedCount_%": row.get("FailedCount_Percent"),
            }
        )
    return results


def get_concern_report(instance_id):
    """
    Fetch concern records with concern questionnaire responses only.

    For every submodule:
        - Runs concern report query.
        - Filters only concern questions using ModelData.
        - Maps questionnaire IDs to text and includes response text.

    Returns:
        list[dict]: Concern Q&A with responses for all submodules.
    """
    results = []
    submodules = get_submodules(instance_id)

    # Load questionnaire text
    questionnaire = TemplateQuestionnaire(instance_id)
    questions_df = questionnaire.load_all_questions()
    question_lookup = {}
    if not questions_df.empty:
        question_lookup = dict(
            zip(questions_df["QuestionnaireID"], questions_df["QuestionnaireText"])
        )

    for sub in submodules:
        amount_field = sub.get_amount_column()
        table_name = sub.get_data_table_name()
        if not amount_field or not table_name:
            continue

        instance = GlobalSettings().instance_by_id(sub.instance_id)

        # Get concern-only keys using fetch_data
        model_data = ModelData(sub)
        concern_keys, _ = model_data.get_filtered_questionnaire_data()
        concern_ids = concern_keys.values.tolist()

        if not concern_ids:
            continue

        # Pull data from SQL query
        query_template = config["INSIGHT"]["CONCERN_REPORT_QUERY"]
        query = query_template.format(
            Module=sub.module,
            SubModule=sub.submodule,
            AmountColumn=amount_field,
            SPT_BaseTable=table_name,
        )

        dff = instance.settings.projectdb.download_table_or_query(query=query)
        if dff is None:
            continue
        df = dff.compute()
        if df.empty:
            continue

        df.columns = [col.strip().upper() for col in df.columns]

        # Keep only concern rows
        df = df[df["SPT_ROWID"].isin(concern_ids)]

        for _, row in df.iterrows():
            qtaid = row.get("QTAID")
            question_text = question_lookup.get(qtaid)

            results.append(
                {
                    "Module": sub.module,
                    "Sub Module": sub.submodule,
                    "UDM Table": row.get("UDMTABLE"),
                    "SPT_RowID": row["SPT_ROWID"],
                    "Total_Concern_Amount": row.get("TOTAL_CONCERN_AMOUNT"),
                    "Tests_Failed_Count": row.get("TESTS_FAILED_COUNT"),
                    "QTAID": qtaid,
                    "QuestionnaireText": question_text,
                    "ResponseText": row.get("RESPONSETEXT"),
                    "ClosedAs": row.get("CLOSEDAS"),
                }
            )
    return results


def get_automl_report_data(instance_id):
    """
    Collect AutoML experiment results and performance metrics.

    For every submodule:
        - Loads AutoML experiments using `ModelTracker`.
        - Iterates through available ML models.
        - Extracts evaluation metrics including F1, precision,
          recall, accuracy, and balanced accuracy.

    Args:
        instance_id (str): Unique identifier of the active project instance.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - Module (str)
            - Sub Module (str)
            - Experiment (str)
            - Model (str)
            - F1 (float)
            - Precision (float)
            - Recall (float)
            - Accuracy (float)
            - Balanced_Accuracy (float)
    """
    results = []
    submodules = get_submodules(instance_id)

    for sub in submodules:
        tracker = ModelTracker(sub)
        experiments = tracker.ml_experiments
        models = tracker.get_ml_models(experiments)

        for model in models:
            metrics = model.metrics.model_dump()
            results.append(
                {
                    "Module": sub.module,
                    "Sub Module": sub.submodule,
                    "Experiment": experiments[0].name if experiments else "",
                    "Model": model.name,
                    "F1": metrics.get("f1"),
                    "Precision": metrics.get("precision"),
                    "Recall": metrics.get("recall"),
                    "Accuracy": metrics.get("accuracy"),
                    "Balanced_Accuracy": metrics.get("balanced_accuracy"),
                }
            )
    return results
