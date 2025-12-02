# # Copyright (C) KonaAI - All Rights Reserved
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from src.insight import fetch_insightdata as fid


class FakeInstance:
    def __init__(self):
        self.settings = MagicMock()
        self.settings.projectdb = MagicMock()


@pytest.fixture
def fake_instance():
    return FakeInstance()


@patch("src.insight.fetch_insightdata.Metadata")
@patch("src.insight.fetch_insightdata.Submodule")
def test_get_submodules(mock_submodule, mock_metadata):
    """Should construct Submodule objects from Metadata"""
    mock_metadata.return_value.modules = ["Module1"]
    mock_metadata.return_value.get_submodule_names.return_value = ["Sub1", "Sub2"]

    subs = fid.get_submodules("inst1")

    assert len(subs) == 2
    mock_submodule.assert_any_call(
        instance_id="inst1", module="Module1", submodule="Sub1"
    )
    mock_submodule.assert_any_call(
        instance_id="inst1", module="Module1", submodule="Sub2"
    )


@patch("src.insight.fetch_insightdata.GlobalSettings")
def test_get_amount_count_data(mock_settings, fake_instance):
    """Should compute total/concern counts and amounts"""
    df = pd.DataFrame(
        {
            "ML_Prediction": [0, 1, 1],
            "amount": [100, 200, 300],
        }
    )
    dff = MagicMock()
    dff.compute.return_value = df
    fake_instance.settings.projectdb.download_table_or_query.return_value = dff
    mock_settings.return_value.instance_by_id.return_value = fake_instance

    with patch("src.insight.fetch_insightdata.get_submodules") as mock_subs:
        mock_sub = MagicMock()
        mock_sub.get_amount_column.return_value = "amount"
        mock_sub.get_data_table_name.return_value = "table1"
        mock_sub.module, mock_sub.submodule, mock_sub.instance_id = "M1", "S1", "inst1"
        mock_subs.return_value = [mock_sub]

        result = fid.get_amount_count_data("inst1")

    assert result[0]["total_count"] == 3
    assert result[0]["total_concern_count"] == 2
    assert result[0]["Total_amount"] == 600
    assert result[0]["total_concern_amount"] == 500


@patch("src.insight.fetch_insightdata.GlobalSettings")
def test_get_top_risk_patterns(mock_settings, fake_instance):
    """Should map DB results into risk pattern dicts"""
    df = pd.DataFrame(
        [
            {
                "Module": "P2P",
                "SubModule": "Payments",
                "PatternID": "P1",
                "PatternDescriptionLong": "Desc",
                "KRI": "High",
                "TestedCount": 10,
                "FailedCount": 2,
                "FailedCount_Percent": 20.0,
            }
        ]
    )
    dff = MagicMock()
    dff.compute.return_value = df
    fake_instance.settings.projectdb.download_table_or_query.return_value = dff
    mock_settings.return_value.instance_by_id.return_value = fake_instance

    result = fid.get_top_risk_patterns("inst1")

    assert len(result) == 1
    assert result[0]["PatternID"] == "P1"
    assert result[0]["FailedCount_%"] == 20.0


@patch("src.insight.fetch_insightdata.GlobalSettings")
@patch("src.insight.fetch_insightdata.TemplateQuestionnaire")
@patch("src.insight.fetch_insightdata.ModelData")
def test_get_concern_report_valid(
    mock_modeldata, mock_questionnaire, mock_settings, fake_instance
):
    """Should return concern report with mapped questionnaire text"""
    # Questionnaire text lookup
    mock_questionnaire.return_value.load_all_questions.return_value = pd.DataFrame(
        {"QuestionnaireID": [27], "QuestionnaireText": ["5a. Alert Determination"]}
    )

    # Concern keys from ModelData
    mock_modeldata.return_value.get_filtered_questionnaire_data.return_value = (
        pd.Series(["123"]),
        None,
    )

    # DB result with matching SPT_ROWID
    df = pd.DataFrame(
        [
            {
                "UDMTable": "Analytics.Payments",
                "SPT_RowID": "123",
                "Total_Concern_Amount": 5000,
                "Tests_Failed_Count": 3,
                "QTAID": 27,
                "ResponseText": "Concern",
                "ClosedAs": "Concern",
            }
        ]
    )
    dff = MagicMock()
    dff.compute.return_value = df
    fake_instance.settings.projectdb.download_table_or_query.return_value = dff
    mock_settings.return_value.instance_by_id.return_value = fake_instance

    # Patch submodules
    with patch("src.insight.fetch_insightdata.get_submodules") as mock_subs:
        mock_sub = MagicMock()
        mock_sub.get_amount_column.return_value = "amount"
        mock_sub.get_data_table_name.return_value = "Analytics.Payments"
        mock_sub.module, mock_sub.submodule, mock_sub.instance_id = (
            "P2P",
            "Payments",
            "inst1",
        )
        mock_subs.return_value = [mock_sub]

        result = fid.get_concern_report("inst1")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["Total_Concern_Amount"] == 5000
    assert result[0]["QuestionnaireText"] == "5a. Alert Determination"
    assert result[0]["SPT_RowID"] == "123"


@patch("src.insight.fetch_insightdata.GlobalSettings")
@patch("src.insight.fetch_insightdata.ModelTracker")
def test_get_automl_report_data(mock_tracker, mock_settings, fake_instance):
    """Should extract model metrics from AutoML experiments"""
    mock_settings.return_value.instance_by_id.return_value = fake_instance

    fake_experiment = MagicMock()
    fake_experiment.name = "Exp1"

    fake_model = MagicMock()
    fake_model.name = "XGBoost"
    fake_model.metrics.model_dump.return_value = {
        "f1": 0.8,
        "precision": 0.75,
        "recall": 0.7,
        "accuracy": 0.9,
        "balanced_accuracy": 0.85,
    }

    mock_tracker.return_value.ml_experiments = [fake_experiment]
    mock_tracker.return_value.get_ml_models.return_value = [fake_model]

    with patch("src.insight.fetch_insightdata.get_submodules") as mock_subs:
        mock_sub = MagicMock()
        mock_sub.module, mock_sub.submodule, mock_sub.instance_id = (
            "P2P",
            "Payments",
            "inst1",
        )
        mock_subs.return_value = [mock_sub]

        result = fid.get_automl_report_data("inst1")

    assert isinstance(result, list)
    assert result[0]["Model"] == "XGBoost"
    assert result[0]["F1"] == 0.8
    assert result[0]["Accuracy"] == 0.9
