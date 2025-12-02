# # Copyright (C) KonaAI - All Rights Reserved
"""This file contains the instance management page components"""
import asyncio
from typing import Optional

import humanize
import pandas as pd
from nicegui import ui
from src.admin import theme
from src.admin.components.clipboard import copy_to_clipboard
from src.admin.components.spinners import create_loader
from src.admin.components.spinners import create_overlay_spinner
from src.admin.pages.database_form import DatabaseForm
from src.admin.pages.instance_settings_form import InstanceSettingsForm
from src.admin.utils import config
from src.utils.database_config import SQLDatabaseManager
from src.utils.global_config import GlobalSettings
from src.utils.status import Status


def instance():
    """
    Displays and manages instances within the admin interface.
    This function renders the "Instance Management" UI, showing details of the currently active instance,
    providing notifications if no instance is active, and presenting expandable sections for viewing
    existing instances and creating new ones.
    UI Components:
    -------------
    - Active instance details (ID, client name, project name)
    - Notification if no active instance is set
    - Expandable panels for viewing and creating instances
    """
    with theme.frame("Instance Management"):
        ui.markdown("# Instance Management")

        # Show active instance
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

        # view instances with expanders
        with (
            ui.expansion("View Instances")
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            view_instances()

        with (
            ui.expansion("Create New Instance")
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            create_instance()


@ui.refreshable_method
def view_instances():
    """
    Displays a table of available instances and provides interactive controls for selecting a client, project, and viewing instance details.

    The function performs the following:
    - Shows the total number of instances or a message if none are found.
    - Allows selection of a client, then dynamically updates available projects for that client.
    - Allows selection of a project, then displays details of the corresponding instance, including IDs, creation date, and related settings.
    - Provides action buttons to delete or activate the selected instance.

    UI elements are dynamically updated based on user selections.
    """

    instances = GlobalSettings().instances
    if not instances:
        ui.label("No instances found")
        return

    ui.label(f"Total instances: {len(instances)}")

    # Initialize state dictionary
    state = {"client": None, "project": None, "instance_id": None}

    # Client selection
    all_clients = {inst.ClientUID: inst.client_name for inst in instances}
    all_projects = {}  # Initialize all_projects as an empty dictionary
    (
        ui.select(
            options=all_clients,
            label="Client Name",
            on_change=lambda: _update_project(),  # pylint: disable=unnecessary-lambda
        )
        .bind_value_to(state, "client")
        .classes("w-full")
    )

    def _update_project():
        if state.get("client"):
            all_projects = {
                inst.ProjectUID: inst.project_name
                for inst in instances
                if inst.ClientUID == state.get("client")
            }
            project_select.options = all_projects
            project_select.update()

    # Project selection
    project_select = (
        ui.select(
            options=all_projects,
            label="Project Name",
            on_change=lambda: _update_instance_id(),  # pylint: disable=unnecessary-lambda
        )
        .bind_value_to(state, "project")
        .classes("w-full")
    )

    instance_details = ui.element("div").classes("w-full")

    def _update_instance_id():
        instance_obj = GlobalSettings.instance_by_client_project(
            client_uid=state.get("client"), project_uid=state.get("project")
        )
        state["instance_id"] = instance_obj.instance_id
        instance_details.clear()
        if instance_obj:
            with instance_details:
                ui.space()
                with ui.grid(columns=2).classes("w-full gap-1 items-start"):
                    # Show instance details
                    ui.label("Instance ID").classes("text-sm text-gray-500")
                    ui.label(instance_obj.instance_id).classes(
                        "text-base font-semibold"
                    )

                    # client and project details
                    ui.label("Client UID").classes("text-sm text-gray-500")
                    copy_to_clipboard(instance_obj.ClientUID)

                    ui.label("Project UID").classes("text-sm text-gray-500")
                    copy_to_clipboard(instance_obj.ProjectUID)

                    # Created date
                    ui.label("Created At").classes("text-sm text-gray-500")
                    ui.label(
                        f"{instance_obj.created_date.strftime('%d-%B-%Y %H:%M:%S')} ({humanize.naturaltime(instance_obj.created_date)})"
                    ).classes("text-base font-semibold")

                # Settings
                InstanceSettingsForm(instance=instance_obj).render()

    # Action buttons
    with ui.row().classes("w-full justify-end gap-2"):
        ui.button(
            "Delete", on_click=lambda: delete_instance(state["instance_id"])
        ).props("icon=delete").bind_enabled_from(
            state,
            "instance_id",
        )

        ui.button(
            "Activate", on_click=lambda: activate_instance(state["instance_id"])
        ).props("icon=check_circle").bind_enabled_from(
            state,
            "instance_id",
        )


def delete_instance(instance_id):
    """
    Delete the instance with the given instance_id.

    If no instance_id is given, do nothing and show a warning notification.
    """
    if not instance_id:
        ui.notify("No instance selected", type="warning")
        return

    if GlobalSettings().delete_instance(instance_id):
        # Delete implementation here
        ui.notify(f"Deleted instance {instance_id}", type="positive")
        view_instances.refresh()
    else:
        ui.notify(f"Failed to delete instance {instance_id}", type="negative")
        Status.FAILED(
            "Failed to delete instance", error="Instance not found or invalid"
        )
        return


def activate_instance(instance_id):
    """
    Activate the instance with the given instance_id.

    If no instance_id is given, show a warning notification and do nothing.
    """
    if not instance_id:
        ui.notify("No instance selected", type="warning")
        return

    if GlobalSettings().set_active_instance(instance_id):
        # Activation implementation here
        ui.notify(f"Activated instance {instance_id}", type="positive")
        # refresh the page
        ui.navigate.reload()
    else:
        ui.notify(f"Failed to activate instance {instance_id}", type="negative")
        Status.FAILED(
            "Failed to activate instance", error="Instance not found or invalid"
        )
        return


def download_client_master(db: SQLDatabaseManager) -> Optional[pd.DataFrame]:
    """
    Download client master data from master database using the given query.

    Returns:
        pd.DataFrame: Client master dataframe if download is successful, else None.
    """
    try:
        if not db.is_db_connected:
            raise ConnectionError("Database connection failed.")

        query = config.get("SPT_MASTER", "instance_query")
        ddf = db.download_table_or_query(query=query)
        if ddf is None or len(ddf) == 0:
            raise ConnectionError(
                "Error downloading client master. Check DB connection."
            )

        return ddf.compute()
    except BaseException as _e:
        Status.FAILED("Client master download failed:", error=str(_e), traceback=False)
        ui.notify("Something went wrong. Contact support.", type="negative")
        return None


@ui.refreshable_method
def create_instance(masterdb: SQLDatabaseManager = None):
    """
    Creates a UI form component for setting up a master database instance and loading client master data.
    Args:
        masterdb (SQLDatabaseManager, optional): An instance of the SQLDatabaseManager. If not provided, a new instance is created.
    UI Components:
    -------------
        - Master Database label and form
        - Status label for error messages
        - "Load Client Master" button to trigger client master loading
        - Container for dynamic content
    Exceptions:

        Displays an error label in the UI if any exception occurs during setup.
    """
    try:
        masterdb: SQLDatabaseManager = SQLDatabaseManager()

        with ui.column().classes("w-full gap-4 mb-5"):
            # Setup Master DB
            ui.label("Master Database").classes("text-lg font-semibold")
            DatabaseForm(model=masterdb).render()

            status_label = ui.label().style("color: red")

            async def load_client_master():
                container.clear()
                with container:
                    spinner = create_overlay_spinner("Loading client master...")
                await asyncio.sleep(0.2)
                spinner.delete()
                _draw_client_master(masterdb, container, status_label)

            ui.button("Load Client Master", on_click=load_client_master).props(
                "icon=download"
            ).classes("w-full")

            container = ui.element("div").classes("w-full")
    except Exception as _e:
        ui.label(str(_e)).style("color:red")


@ui.refreshable
def _draw_client_master(db: SQLDatabaseManager, container, status_label):
    """
    Renders the client and project selection UI for creating an instance.
    This function clears the provided container and status label, checks the database connection,
    downloads the client master data, and displays selection widgets for clients and projects.
    It also provides a button to create a new instance based on the selected client and project.
    Args:
        db (SQLDatabaseManager): The database manager instance to interact with the database.
        container: The UI container element where the selection widgets and buttons are rendered.
        status_label: The UI label element used to display status messages.
    Raises:
        OSError: If the client master data cannot be downloaded or is not a valid DataFrame.
    """

    container.clear()
    status_label.set_text("")

    if not db.is_db_connected:
        container.clear()
        status_label.set_text("Database not connected. Please check the configuration.")
        status_label.style("color:red")
        ui.notify(
            "Database not connected. Please check the configuration.", type="negative"
        )
        return

    dff: pd.DataFrame = download_client_master(db)
    if dff is None or not isinstance(dff, pd.DataFrame):
        raise OSError("Error downloading client master. Check DB connection.")

    # Select Project and Client
    with container:
        ui.label("Select Client and Project").classes("text-lg font-semibold")

        state = {"selected_client": None, "selected_project": None}

        # Create client select first
        client_options = dff.set_index("clientuid")["ClientIdentifier"].to_dict()
        ui.select(
            label="Client Name",
            options=client_options,
            on_change=lambda: _update_projects(),  # pylint: disable=unnecessary-lambda
        ).classes("w-full").bind_value_to(state, "selected_client")

        # update filtered projects
        def _update_projects():
            if state.get("selected_client"):
                all_projects = dff[  # pylint: disable=unsubscriptable-object
                    dff["clientuid"]  # pylint: disable=unsubscriptable-object
                    == state.get("selected_client")
                ]
                project_select.options = all_projects.set_index("projectuid")[
                    "ProjectName"
                ].to_dict()
                project_select.update()

        # Create project select second but keep reference
        project_select = (
            ui.select(
                label="Project Name",
                options=[],
            )
            .classes("w-full")
            .bind_value_to(state, "selected_project")
        )
        with container:
            ui.button(
                "Create Instance",
                on_click=lambda: asyncio.create_task(
                    _create_instance(
                        dff, state, selected_instance, db, loader_container
                    )
                ),
            ).props("icon=add").classes("mt-3")
            loader_container = ui.element("div")
            # create a single message label to show instance id
            selected_instance = ui.label().style("color: green").classes("mt-3")


# Keep the _create_instance function same as before
async def _create_instance(
    dff: pd.DataFrame,
    state: dict,
    selected_instance: ui.label,
    masterdb: SQLDatabaseManager,
    loader_container,
):
    """
    Asynchronously creates a new instance for the selected client and project.
    This function performs the following steps:
    1. Clears the loader container and resets the selected instance label.
    2. Checks if an instance already exists for the given client and project.
    3. Validates the existence of the selected project in the provided DataFrame.
    4. Creates a new instance using the client and project information.
    5. Sets the active instance and saves its settings to the master database.
    6. Updates the UI to reflect the result and refreshes the instance view.
    Args:
        dff (pd.DataFrame): DataFrame containing client and project information.
        state (dict): Dictionary holding the current UI state, including selected client and project.
        selected_instance (ui.label): UI label to display instance creation status.
        masterdb (SQLDatabaseManager): Database manager for saving instance settings.
        loader_container: UI container for displaying loading indicators.
    Raises:
        Exception: If any error occurs during instance creation, updates the UI and logs the error.
    """

    loader_container.clear()
    selected_instance.set_text("")
    with loader_container:
        create_loader()
    await asyncio.sleep(0.2)

    try:
        selected_instance.set_text("")

        client_id = state["selected_client"]
        project_id = state["selected_project"]

        # Check if instance already exists
        if instance_id := GlobalSettings().instance_by_client_project(
            client_id, project_id
        ):
            selected_instance.set_text(f"Instance already exists: {instance_id}")
            selected_instance.style("color:red")
            return

        # Use correct column names (lowercase)
        project_record = dff[
            (dff["clientuid"] == client_id) & (dff["projectuid"] == project_id)
        ]
        if project_record.empty:
            selected_instance.set_text("No project found for the selected client.")
            selected_instance.style("color:red")
            return

        client_name = project_record["ClientIdentifier"].values[0]
        project_name = project_record["ProjectName"].values[0]

        instance_obj = GlobalSettings().create_instance(
            client_name=client_name,
            client_uid=client_id,
            project_name=project_name,
            project_uid=project_id,
        )
        if not instance_obj:
            selected_instance.set_text("Failed to create instance.")
            selected_instance.style("color:red")
            return

        # Set the active instance
        instance_obj.settings.masterdb = masterdb
        if instance_obj.save_settings():
            selected_instance.set_text(
                f"Instance created successfully: {instance_obj.instance_id}"
            )
            selected_instance.style("color:green")
            view_instances.refresh()
    except Exception as _e:
        Status.FAILED("Error creating instance", error=str(_e))
        selected_instance.set_text("Something went wrong. Contact support.")
    finally:
        loader_container.clear()
