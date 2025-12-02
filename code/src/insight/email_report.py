# # Copyright (C) KonaAI - All Rights Reserved
"""
This module handles the generation and delivery of insight reports via email.
It integrates with the notification system to fetch instance-level recipient
emails and allows users to specify additional recipients.
"""
import datetime
import os
import shutil
import tempfile

from nicegui import run
from src.insight.download_report import excel_report
from src.utils.global_config import GlobalSettings
from src.utils.notification import EmailNotification


async def send_email_report(email_list):
    """
    Asynchronously builds an Excel insight report and emails it to a list of recipients.
        email_list: An object with an 'emails' attribute containing a list of user-specified email addresses.
        tuple:
            - (True, recipients): On success, where recipients is the list of email addresses the report was sent to.
            - (False, error_message): On failure, with a string describing the error.
        - Collects recipient emails from instance notification settings and the provided email_list.
        - Ensures at least one recipient is configured.
        - Loads email configuration and checks SMTP server connectivity.
        - Attaches the report to the email and sends it to the recipients.
        - Handles and reports errors encountered during the process.
    """
    instance_id = GlobalSettings().active_instance_id
    instance_obj = GlobalSettings.instance_by_id(instance_id)

    notification = instance_obj.settings.notification
    instance_emails = []
    if notification:
        if getattr(notification, "RecipientEmails", None):
            instance_emails.extend(notification.RecipientEmails)
        if getattr(notification, "CopyEmails", None):
            instance_emails.extend(notification.CopyEmails)

    user_emails = email_list.emails if email_list.emails else []
    recipients = (
        list(set(instance_emails + user_emails)) if user_emails else instance_emails
    )
    if not recipients:
        return False, "No recipient email is configured"

    def send_insight_report(instance_id, recipients):
        """
        Send an insight report as an email with a custom list of recipient emails.
        Args:
            instance_id (str or int): The identifier for the report instance, used to load configuration and generate the report.
            recipients (list of str): List of email addresses to send the report to.
        Returns:
            tuple: (bool, str or None)
                - True, None: if the report was sent successfully.
                - False, str: if there was an error, with the error message.
        Raises:
            Exception: If any unexpected error occurs during the process.
        Process:
            - Loads email configuration for the given instance.
            - Checks SMTP server connection.
            - Generates an Excel report and saves it to a temporary file.
            - Attaches the report to the email.
            - Sends the email to the specified recipients.
        """
        try:
            notifier = EmailNotification(instance_id)
            if not notifier.load_config():
                return False, "Email configuration load failed"

            if not notifier.is_connected():
                return False, "SMTP server connection failed"

            notifier.recipient_emails = recipients
            excel_file = excel_report(instance_id)
            excel_file.seek(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(excel_file.getvalue())
                tmp_path = tmp.name
                today_str = datetime.datetime.now().strftime("%Y-%m-%d")

            final_path = os.path.join(
                os.path.dirname(tmp_path), f"insight_report_{today_str}.xlsx"
            )
            shutil.move(tmp_path, final_path)

            notifier.attach(final_path)
            notifier.add_content(
                "Insight Report", "Please find the attached Insight Report."
            )

            success = notifier.send(subject="Insight Report")
            return (True, None) if success else (False, "send failure")
        except Exception as e:
            return False, f"SMTP error: {str(e)}"

    success, message = await run.io_bound(send_insight_report, instance_id, recipients)
    if success:
        return True, recipients
    return False, message
