# # Copyright (C) KonaAI - All Rights Reserved
"""Class to handle the UI for the chat conversation."""
from typing import List
from typing import Optional
from typing import TypedDict

from nicegui import run
from nicegui import ui
from src.admin.components.spinners import create_loader
from src.sql_agent.context_evaluation import ContextEvaluation
from src.sql_agent.data_dictionary import SQLDataDictionary
from src.sql_agent.prompt_catalogue import PromptCatalogue
from src.sql_agent.UI.context_evaluation_ui import ContextEvaluationUI
from src.sql_agent.UI.data_dictionary_ui import DataDictionaryUI
from src.sql_agent.UI.prompt_catalog_ui import PromptCatalogueUI
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.metadata import Metadata
from src.utils.status import Status
from src.utils.submodule import Submodule


async def display_configuration():
    """Shows the configuration options for the chatbot."""
    # load active instance
    if not GlobalSettings().active_instance_id:
        ui.notify("No active instance selected.", type="negative")
        return

    instance = GlobalSettings.instance_by_id(GlobalSettings().active_instance_id)
    if not instance:
        ui.notify("Active instance ID is invalid.", type="negative")
        return

    # show context evaluation widget (module descriptions)
    ce = (
        ui.expansion(
            "Module Descriptions (Context Evaluation)",
            icon="auto_awesome",
            group="configuration",
        )
        .classes("w-full border-2 rounded-md")
        .props("outlined")
    )
    ce.on_value_change(
        lambda e: ContextEvaluationUI(instance).render(ce) if e.value else None
    )

    # show data dictionary widget
    dad = ui.expansion(
        "Data Dictionary",
        icon="table_view",
        group="configuration",
    ).classes("w-full border-2 rounded-md")
    dad.on_value_change(
        lambda e: DataDictionaryUI(instance).render(dad) if e.value else None
    )

    # show prompt data widget
    pc = (
        ui.expansion(
            "Prompt Catalogue",
            icon="description",
            group="configuration",
        )
        .classes("w-full border-2 rounded-md")
        .props("outlined")
    )
    pc.on_value_change(
        lambda e: PromptCatalogueUI(instance).render(pc) if e.value else None
    )

    # validate configuration
    vc = (
        ui.expansion(
            "Validate Configuration", icon="check_circle", group="configuration"
        )
        .classes("w-full border-2 rounded-md")
        .props("outlined")
    )
    vc.on_value_change(lambda e: display_validation_results(vc) if e.value else None)


def _return_boolean_icon(value: bool) -> str:
    """Returns a check or cross icon based on the boolean value."""
    return "✅" if value else "❌"


class ValidationResult(TypedDict):
    """TypedDict to hold validation results for a module and submodule."""

    module: str
    submodule: str
    prompt_catalogue: str = _return_boolean_icon(False)
    data_dictionary: str = _return_boolean_icon(False)


async def display_validation_results(element: ui.element):
    """Displays the validation results of the configuration."""
    element.clear()
    with element:
        instance: Optional[Instance] = GlobalSettings.instance_by_id(
            GlobalSettings().active_instance_id
        )

        if not instance:
            ui.label("No active instance found.").classes("text-red-500")
            return

        loader = create_loader("Validating configuration...")
        validation_result: Optional[List[ValidationResult]] = await run.io_bound(
            _validate_config, instance
        )
        is_context_prompt = await run.io_bound(
            lambda: ContextEvaluation(instance).exists
        )
        loader.delete()

        # display context evaluation prompt status
        if is_context_prompt:
            Status.INFO("Context evaluation prompt exists.", instance)
        else:
            Status.WARNING("Context evaluation prompt does not exist.", instance)
        with ui.row().classes("items-center gap-4"):
            ui.label("Context Evaluation Prompt").classes("font-bold")
            ui.label(_return_boolean_icon(is_context_prompt))

        # display validation results table
        if not validation_result:
            ui.label("No validation results to display.").classes("text-red-500")
            return

        with (
            ui.column()
            .classes("w-full gap-2")
            .style("max-height: 400px; overflow-y: auto;")
        ):
            ui.table(rows=validation_result).props("flat bordered hoverable").classes(
                "w-full"
            )


def _validate_config(instance: Optional[Instance]) -> Optional[List[ValidationResult]]:
    """Validates the configuration of the chatbot for the active instance."""
    md = Metadata(instance.instance_id)
    module_names: List[str] = md.modules
    if not module_names:
        ui.label("No modules found for the active instance.").classes("text-red-500")
        return None

    result: List[ValidationResult] = []
    # validate each module and submodule
    for module_name in module_names:
        submodule_names: List[str] = md.get_submodule_names(module_name)
        for submodule_name in submodule_names:
            sub: Submodule = Submodule(
                module=module_name,
                submodule=submodule_name,
                instance_id=instance.instance_id,
            )
            table_name: str = sub.get_data_table_name()
            if not table_name:
                # at times we have test modules/submodules without data tables
                result.append(
                    ValidationResult(
                        module=module_name,
                        submodule=submodule_name,
                        prompt_catalogue=_return_boolean_icon(False),
                        data_dictionary=_return_boolean_icon(False),
                    )
                )
                continue

            schema, table = table_name.split(".", 1)

            # Validate data dictionary
            valid_data_dictionary: bool = SQLDataDictionary(
                table_schema=schema, table_name=table, db=instance.settings.projectdb
            ).exists
            if valid_data_dictionary:
                Status.INFO("Data dictionary is populated", sub)
            else:
                Status.WARNING("Data dictionary is not populated", sub)

            # Validate prompt catalogue
            valid_prompt_catalogue: bool = PromptCatalogue(
                module=module_name,
                submodule=submodule_name,
                instance_id=instance.instance_id,
            ).exists
            if valid_prompt_catalogue:
                Status.INFO("Prompt catalogue is present", sub)
            else:
                Status.WARNING("Prompt catalogue is not present", sub)

            # record results
            result.append(
                ValidationResult(
                    module=module_name,
                    submodule=submodule_name,
                    prompt_catalogue=_return_boolean_icon(valid_prompt_catalogue),
                    data_dictionary=_return_boolean_icon(valid_data_dictionary),
                )
            )

    return result
