# # Copyright (C) KonaAI - All Rights Reserved
"""
This module defines the Insight Management user interface using NiceGUI.
It allows users to generate, view, download, and email analytical reports
based on project instance data.
"""
import asyncio
import datetime
import tempfile

from nicegui import run
from nicegui import ui
from src.admin import theme
from src.admin.components.email_list import EmailList
from src.admin.components.spinners import create_loader
from src.admin.components.spinners import create_overlay_spinner
from src.insight.download_report import excel_report
from src.insight.email_report import send_email_report
from src.insight.fetch_insightdata import get_amount_count_data
from src.insight.fetch_insightdata import get_automl_report_data
from src.insight.fetch_insightdata import get_concern_report
from src.insight.fetch_insightdata import get_top_risk_patterns
from src.utils.global_config import GlobalSettings


def insight_management():
    """
    Build the Insight Management UI frame.

    Features:
        - Displays active instance metadata (ID, client name, project name)
        - "Generate Report" button to fetch insight datasets
        - Dynamic tables rendered for four datasets
        - "Download Report" button to generate Excel report from backend
        - "Email Report" button to send Excel report via email
    """

    with theme.frame("Insight Management"):
        ui.markdown("# Insight Management")

        # Show active instance info
        if GlobalSettings().active_instance_id:
            if instance_obj := GlobalSettings.instance_by_id(
                GlobalSettings().active_instance_id
            ):
                with ui.grid(columns=3).classes("w-full gap-1 items-start"):
                    for label, value in [
                        ("Active Instance ID", instance_obj.instance_id),
                        ("Client Name", instance_obj.client_name),
                        ("Project Name", instance_obj.project_name),
                    ]:
                        with ui.column().classes("gap-1"):
                            ui.label(label).classes("text-sm text-gray-500")
                            ui.label(value).classes("text-base font-semibold")
            else:
                ui.label("No active instance found").classes("text-red-500")
        else:
            ui.notify("Please activate an instance to continue", type="negative")

        # --- Generate Report Section ---
        with ui.row().classes("w-full items-start mt-8 gap-16 justify-start"):
            generate_button = ui.button("Generate  Report", color="primary").classes(
                "w-72 h-16 text-2xl font-bold self-center"
            )

        tables_container = ui.column().classes("w-full").style("display: none;")

        def show_reports():
            # Show overlay spinner immediately
            spinner = create_overlay_spinner("Generating report, please wait...")

            async def load_tables():
                try:
                    tables_container.clear()
                    tables_container.style("display: block;")
                    instance_id = GlobalSettings().active_instance_id

                    with tables_container:
                        # --- Amount and Count Table ---

                        amount_count_columns = [
                            {"name": "Module", "label": "Module", "field": "Module"},
                            {
                                "name": "Sub Module",
                                "label": "Sub Module",
                                "field": "Sub Module",
                            },
                            {
                                "name": "RiskTransactionAmountField",
                                "label": "RiskTransactionAmountField",
                                "field": "RiskTransactionAmountField",
                            },
                            {
                                "name": "SPT_Base_Table",
                                "label": "SPT_Base_Table",
                                "field": "SPT_Base_Table",
                            },
                            {
                                "name": "total_count",
                                "label": "Total Count",
                                "field": "total_count",
                            },
                            {
                                "name": "Total_amount",
                                "label": "Total Amount",
                                "field": "Total_amount",
                            },
                            {
                                "name": "total_concern_count",
                                "label": "Total Concern Count",
                                "field": "total_concern_count",
                            },
                            {
                                "name": "total_concern_amount",
                                "label": "Total Concern Amount",
                                "field": "total_concern_amount",
                            },
                        ]
                        amount_count_data = await run.io_bound(
                            get_amount_count_data, instance_id
                        )
                        with ui.expansion(
                            "Amount and Count Report", icon="📊", value=False
                        ).classes("w-full shadow-lg mt-6"):
                            ui.table(
                                columns=amount_count_columns,
                                rows=amount_count_data,
                                row_key="Module",
                            ).classes("w-full border")

                        # --- Top Risk Patterns Table ---

                        top_risk_columns = [
                            {"name": "Module", "label": "Module", "field": "Module"},
                            {
                                "name": "Sub Module",
                                "label": "Sub Module",
                                "field": "Sub Module",
                            },
                            {
                                "name": "PatternID",
                                "label": "PatternID",
                                "field": "PatternID",
                            },
                            {
                                "name": "PatternDescriptionLong",
                                "label": "PatternDescriptionLong",
                                "field": "PatternDescriptionLong",
                            },
                            {"name": "KRI", "label": "KRI", "field": "KRI"},
                            {
                                "name": "TestedCount",
                                "label": "Tested Count",
                                "field": "TestedCount",
                            },
                            {
                                "name": "FailedCount",
                                "label": "Failed Count",
                                "field": "FailedCount",
                            },
                            {
                                "name": "FailedCount_%",
                                "label": "Failed Count (%)",
                                "field": "FailedCount_%",
                            },
                        ]

                        top_risk_data = await run.io_bound(
                            get_top_risk_patterns, instance_id
                        )
                        with ui.expansion(
                            "Top Risk Pattern IDs", icon="⚠️", value=False
                        ).classes("w-full shadow-lg mt-6"):
                            ui.table(
                                columns=top_risk_columns,
                                rows=top_risk_data,
                                row_key="PatternID",
                            ).classes("w-full border")

                        # --- Concern Report Table ---

                        concern_report_columns = [
                            {"name": "Module", "label": "Module", "field": "Module"},
                            {
                                "name": "Sub Module",
                                "label": "Sub Module",
                                "field": "Sub Module",
                            },
                            {
                                "name": "UDM_Table",
                                "label": "UDMTable",
                                "field": "UDM Table",
                            },
                            {
                                "name": "SPT_RowID",
                                "label": "SPT_RowID",
                                "field": "SPT_RowID",
                            },
                            {
                                "name": "Total_Concern_Amount",
                                "label": "Total Concern Amount",
                                "field": "Total_Concern_Amount",
                            },
                            {
                                "name": "Tests_Failed_Count",
                                "label": "Tests Failed Count",
                                "field": "Tests_Failed_Count",
                            },
                            {"name": "QTAID", "label": "QTAID", "field": "QTAID"},
                            {
                                "name": "QuestionnaireText",
                                "label": "Questionnaire Text",
                                "field": "QuestionnaireText",
                            },
                            {
                                "name": "ResponseText",
                                "label": "Response Text",
                                "field": "ResponseText",
                            },
                            {
                                "name": "ClosedAs",
                                "label": "Closed As",
                                "field": "ClosedAs",
                            },
                        ]
                        concern_report_data = await run.io_bound(
                            get_concern_report, instance_id
                        )
                        with ui.expansion(
                            "Concern Records", icon="🚨", value=False
                        ).classes("w-full shadow-lg mt-6"):
                            ui.table(
                                columns=concern_report_columns,
                                rows=concern_report_data,
                                row_key="SPT_RowID",
                            ).classes("w-full border")

                        # --- AutoML Report Table ---

                        automl_report_columns = [
                            {"name": "Module", "label": "Module", "field": "Module"},
                            {
                                "name": "Sub Module",
                                "label": "Sub Module",
                                "field": "Sub Module",
                            },
                            {
                                "name": "Experiment",
                                "label": "Experiment",
                                "field": "Experiment",
                            },
                            {"name": "Model", "label": "Model", "field": "Model"},
                            {"name": "F1", "label": "F1", "field": "F1"},
                            {
                                "name": "Precision",
                                "label": "Precision",
                                "field": "Precision",
                            },
                            {"name": "Recall", "label": "Recall", "field": "Recall"},
                            {
                                "name": "Accuracy",
                                "label": "Accuracy",
                                "field": "Accuracy",
                            },
                            {
                                "name": "Balanced_Accuracy",
                                "label": "Balanced Accuracy",
                                "field": "Balanced_Accuracy",
                            },
                        ]
                        automl_report_data = await run.io_bound(
                            get_automl_report_data, instance_id
                        )
                        with ui.expansion(
                            "AutoML Report", icon="🤖", value=False
                        ).classes("w-full shadow-lg mt-6"):
                            ui.table(
                                columns=automl_report_columns,
                                rows=automl_report_data,
                                row_key="Experiment",
                            ).classes("w-full border")

                        async def download_report(download_btn):
                            download_btn.disable()
                            loader = create_loader("Downloading Report...")
                            try:
                                instance_id = GlobalSettings().active_instance_id
                                excel_file = await run.io_bound(
                                    excel_report, instance_id
                                )

                                def write_temp_file():
                                    with tempfile.NamedTemporaryFile(
                                        delete=False, suffix=".xlsx"
                                    ) as tmp:
                                        tmp.write(excel_file.getvalue())
                                        tmp_path = tmp.name
                                    return tmp_path

                                tmp_path = await run.io_bound(write_temp_file)
                                today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                                filename = f"Insight_Report_{today_str}.xlsx"
                                ui.download(tmp_path, filename=filename)
                                ui.notify(
                                    "Report downloaded successfully", type="positive"
                                )

                            finally:
                                loader.delete()  # hide spinner
                                download_btn.enable()

                        def email_report_dialog():
                            instance_id = GlobalSettings().active_instance_id
                            instance_obj = GlobalSettings.instance_by_id(instance_id)

                            default_emails = []
                            if instance_obj and hasattr(instance_obj, "settings"):
                                notification = getattr(
                                    instance_obj.settings, "notification", None
                                )
                                if notification:
                                    if getattr(notification, "RecipientEmails", None):
                                        default_emails.extend(
                                            notification.RecipientEmails
                                        )
                                    if getattr(notification, "CopyEmails", None):
                                        default_emails.extend(notification.CopyEmails)

                            dialog = ui.dialog()

                            with dialog, ui.card().classes("w-[500px] p-6 gap-4"):
                                ui.label("Configured Emails").classes(
                                    "text-lg font-semibold"
                                )

                                if default_emails:
                                    for mail in default_emails:
                                        ui.label(mail).classes("text-blue-700 ml-2")
                                else:
                                    ui.label("No email configured").classes(
                                        "text-red-500 italic ml-2"
                                    )

                                email_list = EmailList(
                                    label="Recipient Emails",
                                    emails=[],
                                    max_count=15,
                                ).classes("w-full")

                                with ui.row().classes(
                                    "gap-4 mt-6 justify-between w-full"
                                ):
                                    send_btn = ui.button(
                                        "Send", color="primary"
                                    ).classes("w-28")
                                    ui.button("Close", on_click=dialog.close).classes(
                                        "w-28"
                                    )

                                    async def confirm_send():
                                        send_btn.disable()
                                        loader = create_loader("Sending Report...")
                                        try:
                                            success, data = await send_email_report(
                                                email_list
                                            )
                                            if not success:
                                                ui.notify(
                                                    f"Failed to send email: {data}",
                                                    type="negative",
                                                )
                                            else:
                                                recipients = data
                                                ui.notify(
                                                    f"Email sent successfully to: {', '.join(recipients)}",
                                                    type="positive",
                                                )
                                        finally:
                                            loader.delete()
                                            send_btn.enable()
                                            dialog.close()

                                    send_btn.on_click(confirm_send)

                            dialog.open()

                        with ui.row().classes(
                            "w-full justify-center gap-16 mt-12 mb-20"
                        ):
                            download_btn = ui.button(
                                "Download Report",
                                color="primary",
                                on_click=lambda: download_report(download_btn),
                            ).classes("w-56 h-12 text-lg")

                            ui.button(
                                "Email Report",
                                color="primary",
                                on_click=email_report_dialog,
                            ).classes("w-56 h-12 text-lg")

                finally:
                    spinner.delete()  # hide spinner

            asyncio.create_task(load_tables())

        # Connect button to function
        generate_button.on_click(show_reports)
