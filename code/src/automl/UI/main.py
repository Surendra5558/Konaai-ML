# # Copyright (C) KonaAI - All Rights Reserved
"""Main function to run the AutoML Management app"""
from nicegui import run
from nicegui import ui
from src.admin import theme
from src.admin.components.spinners import create_overlay_spinner
from src.automl.UI.explainations import PredictionExplainerUI
from src.automl.UI.model_tracking import ModelTrackerUI
from src.automl.UI.training_questions import show_question_selection
from src.automl.UI.training_settings import TrainingSettingsUI
from src.utils.global_config import GlobalSettings
from src.utils.metadata import Metadata
from src.utils.submodule import Submodule


def automl_mgmt():
    """
    Main function to run the AutoML management UI.
    This function renders the main interface for managing AutoML instances and modules.
    It displays the currently active instance details, provides dropdowns for selecting modules and submodules,
    and allows users to view and configure sections related to the selected module and submodule.
    If no active instance is found, it notifies the user to activate one before proceeding.

    UI Elements:

        - Active instance details (ID, client name, project name)
        - Module selection dropdown
        - Submodule selection dropdown (populated based on selected module)
        - 'Select' button to load configuration sections for the chosen module/submodule
        - Area to display configuration sections
    """

    with theme.frame("Automl Management"):
        ui.markdown("# AutoML Management").classes()

        # Show active instance
        if GlobalSettings().active_instance_id:
            instance_obj = GlobalSettings.instance_by_id(
                GlobalSettings().active_instance_id
            )
            if instance_obj:
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
            return

        # Get all modules
        md: Metadata = Metadata(GlobalSettings().active_instance_id)
        all_modules = md.modules

        def update_submodules():
            """Update submodules based on selected module and load config"""
            selected_submodule.options = md.get_submodule_names(selected_module.value)
            selected_submodule.update()
            sections_area.clear()

        # Module selection dropdown
        selected_module = (
            ui.select(
                label="Select Module",
                options=all_modules,
                on_change=lambda: update_submodules(),  # pylint: disable=unnecessary-lambda
            )
            .classes("w-full")
            .props("outlined")
        )

        # Submodule selection dropdown
        selected_submodule = (
            ui.select(label="Select Submodule", options=[])
            .classes("w-full")
            .props("outlined")
        ).on_value_change(
            lambda: sections_area.clear()  # pylint: disable=unnecessary-lambda
        )

        select_btn = ui.button(text="Select")
        sections_area = ui.column().classes("w-full gap-2")
        select_btn.on_click(
            lambda: (
                show_automl_sections(
                    selected_module.value, selected_submodule.value, sections_area
                )
            )
        )


@ui.refreshable_method
async def show_automl_sections(module: str, submodule: str, display_area: ui.column):
    """
    Asynchronously displays the AutoML sections for a given module and submodule in the provided display area.
    This function clears the display area, shows a loading spinner, initializes the submodule, and then renders
    the question selection, training settings UI, model tracker UI, and prediction explainer UI in sequence.
    Finally, it removes the loading spinner.
    Args:
        module (str): The name of the module to display.
        submodule (str): The name of the submodule to display.
        display_area (ui.column): The UI column where the sections will be rendered.
    Returns:
        None
    """
    display_area.clear()
    spinner = create_overlay_spinner("Loading AutoML Sections...")

    sub = await run.io_bound(
        lambda: Submodule(
            module=module,
            submodule=submodule,
            instance_id=GlobalSettings().active_instance_id,
        )
    )

    with display_area:
        await show_question_selection(sub)
        TrainingSettingsUI(submodule=sub).render()
        await ModelTrackerUI(submodule=sub).render()
        await PredictionExplainerUI(submodule=sub).render()

    spinner.delete()


if __name__ in ("__main__", "__mp_main__"):
    automl_mgmt()
    ui.run(title="AutoML Management", reload=True, port=8080)
