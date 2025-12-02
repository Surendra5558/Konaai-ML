# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains all pages for the application."""
from nicegui import ui
from src.admin.pages.backup_restore import backup_restore
from src.admin.pages.global_settings import global_settings
from src.admin.pages.health_check import health_check
from src.admin.pages.home import home
from src.admin.pages.instance_form import instance
from src.admin.pages.login import login
from src.admin.pages.logs import logs
from src.admin.pages.task_mgmt import task_mgmt
from src.automl.UI.main import automl_mgmt
from src.insight.UI.main import insight_management
from src.sql_agent.UI.main import ConversationUI


def create_ui_routes() -> None:
    """
    Registers various application pages with their corresponding route handlers.

    This function maps specific URL paths to their respective handler functions
    using the `ui.page` decorator. Each handler function is responsible for
    rendering the content of the associated page.
    Returns:
        None
    """
    ui.page("/")(home)
    ui.page("/login")(login)
    ui.page("/global_settings")(global_settings)
    ui.page("/instance")(instance)
    ui.page("/health_check")(health_check)
    ui.page("/backup_restore")(backup_restore)
    ui.page("/logs")(logs)
    ui.page("/task_mgmt")(task_mgmt)
    ui.page("/automl_mgmt")(automl_mgmt)
    ui.page("/insight_management")(insight_management)
    ui.page("/sql_agent")(ConversationUI().render)


if __name__ == "__main__":
    create_ui_routes()
