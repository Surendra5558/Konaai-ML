# # Copyright (C) KonaAI - All Rights Reserved
import io
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.insight import email_report


class FakeEmailList:
    def __init__(self, emails=None):
        self.emails = emails or []


class FakeInstance:
    """Simulates an instance with notification settings."""

    def __init__(self, recipients=None, copies=None):
        self.settings = MagicMock()
        self.settings.notification = MagicMock()
        self.settings.notification.RecipientEmails = (
            [] if recipients is None else recipients
        )
        self.settings.notification.CopyEmails = [] if copies is None else copies


@pytest.mark.asyncio
@patch("src.insight.email_report.GlobalSettings")
async def test_no_recipients_returns_error(mock_settings):
    """When no recipients exist, should return error message."""
    fake_instance = FakeInstance(recipients=[], copies=[])

    mock_settings.return_value.active_instance_id = "inst_123"
    mock_settings.instance_by_id.return_value = fake_instance

    email_list = FakeEmailList(emails=[])
    success, msg = await email_report.send_email_report(email_list)

    assert success is False
    assert "No recipient email" in msg


@pytest.mark.asyncio
@patch("src.insight.email_report.GlobalSettings")
@patch("src.insight.email_report.EmailNotification")
@patch("src.insight.email_report.excel_report")
async def test_successful_send_returns_true(mock_excel, mock_notifier, mock_settings):
    """Happy path: recipients exist, notifier works, report sends successfully."""
    fake_instance = FakeInstance(
        recipients=["inst@test.com"],
        copies=["copy@test.com"],
    )

    mock_settings.return_value.active_instance_id = "inst_123"
    mock_settings.instance_by_id.return_value = fake_instance

    mock_excel.return_value = io.BytesIO(b"fake-excel")

    notifier = mock_notifier.return_value
    notifier.load_config.return_value = True
    notifier.is_connected.return_value = True
    notifier.send.return_value = True

    email_list = FakeEmailList(emails=["user@test.com"])
    success, recipients = await email_report.send_email_report(email_list)

    assert success is True
    assert "inst@test.com" in recipients
    assert "copy@test.com" in recipients
    assert "user@test.com" in recipients


@pytest.mark.asyncio
@patch("src.insight.email_report.GlobalSettings")
@patch("src.insight.email_report.EmailNotification")
@patch("src.insight.email_report.excel_report")
async def test_send_failure_returns_false(mock_excel, mock_notifier, mock_settings):
    """If notifier.send() fails, should return failure message."""
    fake_instance = FakeInstance(
        recipients=["inst@test.com"],
        copies=["copy@test.com"],
    )

    mock_settings.return_value.active_instance_id = "inst_123"
    mock_settings.instance_by_id.return_value = fake_instance

    mock_excel.return_value = io.BytesIO(b"fake-excel")

    notifier = mock_notifier.return_value
    notifier.load_config.return_value = True
    notifier.is_connected.return_value = True
    notifier.send.return_value = False

    email_list = FakeEmailList(emails=["user@test.com"])
    success, msg = await email_report.send_email_report(email_list)

    assert success is False
    assert msg == "send failure"
