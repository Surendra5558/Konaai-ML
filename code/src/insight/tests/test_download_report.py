# # Copyright (C) KonaAI - All Rights Reserved
import io
from unittest.mock import patch

import pandas as pd
import pytest
from src.insight import download_report


class FakeInstance:
    def __init__(self):
        self.client_name = "Test Client"
        self.project_name = "Test Project"
        self.instance_id = "inst_123"


@pytest.fixture
def fake_instance():
    return FakeInstance()


@patch("src.insight.download_report.GlobalSettings")
def test_get_info_df_returns_metadata(mock_settings, fake_instance):
    """Ensure get_info_df returns correct metadata in DataFrame."""
    mock_settings.return_value.instance_by_id.return_value = fake_instance

    df = download_report.get_info_df("inst_123")

    assert not df.empty
    assert "Client Name" in df["Field"].values
    assert "Project Name" in df["Field"].values
    assert "Instance ID" in df["Field"].values


@patch(
    "src.insight.download_report.get_amount_count_data", return_value=[{"Module": "M1"}]
)
@patch("src.insight.download_report.get_top_risk_patterns", return_value=[])
@patch("src.insight.download_report.get_concern_report", return_value=[])
@patch("src.insight.download_report.get_automl_report_data", return_value=[])
@patch("src.insight.download_report.GlobalSettings")
def test_excel_report_creates_sheets(
    mock_settings, mock_auto, mock_concern, mock_risk, mock_amt, fake_instance
):
    """Ensure excel_report generates valid Excel file with expected sheets."""
    mock_settings.return_value.instance_by_id.return_value = fake_instance

    output = download_report.excel_report("inst_123")
    assert isinstance(output, io.BytesIO)

    xls = pd.ExcelFile(output)
    assert "Info" in xls.sheet_names
    assert "Amount and Count Report" in xls.sheet_names
    # Optional: ensure metadata columns are present
    df_amt = pd.read_excel(output, sheet_name="Amount and Count Report")
    assert "Client Name" in df_amt.columns
    assert "Project Name" in df_amt.columns
    assert "Instance ID" in df_amt.columns
