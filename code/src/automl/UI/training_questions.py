# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the training questions UI"""
import ast
import asyncio
import json
from typing import List
from typing import Optional

import pandas as pd
from nicegui import run
from nicegui import ui
from src.admin.components.spinners import create_loader
from src.automl.fetch_data import ModelData
from src.automl.questionnaire import TemplateQuestion
from src.automl.questionnaire import TemplateQuestionnaire
from src.automl.utils import config
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule


@ui.refreshable_method
async def show_question_selection(sub: Submodule, opened: Optional[bool] = False):
    """
    Displays a UI component for selecting and saving training questionnaire questions.
    This function creates an expandable UI section titled "Training Questionnaire"
    where users can select and configure questions for "Concern Question" and
    "No Concern Question". It also provides a "Save" button to persist the
    configuration.
    Args:
        sub (Submodule): An instance of the Submodule class containing the
                         concern and no-concern questionnaires.
    UI Elements:
        - Expansion panel with title "Training Questionnaire".
        - Label displaying "Alert Status: Closed".
        - Dropdowns for selecting "Concern Question" and "No Concern Question".
        - Save button to save the selected configuration.
    Behavior:
        - The selected questions are saved to the `sub` object.
        - The configuration is persisted using the `save_configuration` method of `sub`.
        - Notifications are displayed to indicate success or failure of the save operation.
    """
    # show questions
    with (
        ui.expansion(text="Training Questionnaire", icon="assignment", value=opened)
        .classes("w-full border-2 rounded-md")
        .props("outlined")
    ):
        # Alert state selector
        ui.label("Alert Status: Closed")
        sub.alert_status = "Closed"

        # Select Questionnaire Template Name

        await select_template(sub, "Questionnaire Template Name")

        if not sub.template_id:
            ui.label("No questionnaire template selected").classes("font-extrabold")
            return

        # concern question selection
        concern_question = (
            sub.concern_questionnaire[0]
            if sub.concern_questionnaire
            else TemplateQuestion()
        )
        cq = await select_question(
            concern_question, sub.template_id, "Concern Question"
        )

        # no concern question selection
        no_concern_question = (
            sub.no_concern_questionnaire[0]
            if sub.no_concern_questionnaire
            else TemplateQuestion()
        )
        ncq = await select_question(
            no_concern_question, sub.template_id, "No Concern Question"
        )

        ui.button(
            "Save",
            icon="save",
            on_click=lambda: save(),  # pylint: disable=unnecessary-lambda
        )

        def save():
            # Save configuration
            if not cq or not cq.question or not cq.options or not cq.options[0]:
                ui.notify("Please select a valid Concern Question", type="negative")
                return

            if not ncq or not ncq.question or not ncq.options or not ncq.options[0]:
                ui.notify("Please select a valid No Concern Question", type="negative")
                return

            # Check if both questions are the same
            if cq.question == ncq.question and cq.options[0] == ncq.options[0]:
                ui.notify(
                    "Concern and No Concern questions cannot be the same",
                    type="negative",
                )
                return
            sub.concern_questionnaire = [cq]
            sub.no_concern_questionnaire = [ncq]
            # Save the configuration
            saved = sub.save_configuration()
            if saved:
                ui.notify("Configuration saved", type="positive")
            else:
                ui.notify("Failed to save configuration", type="negative")

        # -------- Minimum data requirement check UI --------
        ui.separator().classes("my-4")
        with ui.card().classes("w-full p-4"):
            # Header: icon + title
            with ui.row().classes("items-center justify-between"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("assessment").classes("text-2xl")  # ← the added icon
                    ui.label("Minimum data requirement Validation").classes(
                        "text-lg font-semibold mb-0"
                    )
                # place the button in the header for a tidy layout
            with ui.row().classes("items-center gap-4"):
                ui.button(
                    "Run Check",
                    icon="play_circle_filled",
                    on_click=lambda: run_check(),  # pylint: disable=unnecessary-lambda
                )

            # compact grid for metrics
            with ui.row().classes("items-center gap-4"):
                concern_count = ui.label("Concern Records: -").classes("font-medium")
                no_concern_count = ui.label("No Concern Records: -").classes(
                    "font-medium"
                )
                total_count = ui.label("Total Records: -").classes("font-medium")

            with ui.row().classes("items-center gap-4 mt-2"):
                min_for_class_label = ui.label(
                    "Minimum records required per class: -"
                ).classes("text-sm")
                total_min_label = ui.label("Minimum total records required: -").classes(
                    "text-sm"
                )

            # pass/fail indicator and error message
            pass_fail_label = ui.markdown("**Status:** -").classes("mt-3")

            def run_check():
                """Function to run the minimum data requirement check"""
                try:
                    # Run validation logic
                    data = ModelData(submodule_obj=sub)
                    validation_result: ModelData.TrainingDataValidationResult = (
                        data.min_data_validator()
                    )
                    if validation_result is None:
                        return

                    concern_count.set_text(
                        f"Concern Records: {validation_result.total_concern_records}"
                    )
                    no_concern_count.set_text(
                        f"No Concern Records: {validation_result.total_no_concern_records}"
                    )
                    total_count.set_text(
                        f"Total Records: {validation_result.total_training_data}"
                    )
                    min_for_class_label.set_text(
                        f"Minimum records required per class: {validation_result.min_data_per_class}"
                    )
                    total_min_label.set_text(
                        f"Minimum total records required: {validation_result.min_training_data}"
                    )
                    pass_fail = (
                        "Sufficient Training Data"
                        if validation_result.has_min_training_data
                        else "Insufficient Training Data"
                    )
                    pass_fail_label.set_content(f"**Status:** {pass_fail}")
                except Exception as e:
                    # Handle unexpected errors gracefully
                    pass_fail_label.set_content(f"**Status:** Check Failed - {str(e)}")
                    pass_fail_label.classes(remove="text-green-600", add="text-red-600")


async def select_template(sub: Submodule, header: str):
    """
    Displays a UI selection widget for choosing a questionnaire template and assigns the selected template ID to the given submodule.
    Args:
    ----
        sub (Submodule): The submodule object whose template ID will be set based on the user's selection.
        header (str): The header text to display above the selection widget.
    Side Effects:
        - Updates the UI with a label and a select dropdown containing available questionnaire template names.
        - Notifies the user if no questionnaire templates are available.
        - Assigns the selected template ID to the submodule when the selection changes.
    """
    ui.label(header).classes("font-extrabold")

    all_questionnaire_template_name = await run.io_bound(
        lambda: TemplateQuestionnaire(
            instance_id=GlobalSettings().active_instance_id
        ).load_questionnaire_template_name()
    )

    if (
        all_questionnaire_template_name is None
        or len(all_questionnaire_template_name) == 0
    ):
        ui.notify("No questionnaire template name available", type="negative")
        return

    # Get column names from config
    name_col = config.get("QUESTIONNAIRE", "QUESTIONNAIRE_TEMPLATE_NAME_COLUMN")
    id_col = config.get("QUESTIONNAIRE", "QUESTIONNAIRE_TEMPLATE_ID_COLUMN")

    # Create options dict {name: id}
    options_dict = (
        all_questionnaire_template_name[[name_col, id_col]]
        .dropna()
        .set_index(name_col)[id_col]
        .to_dict()
    )

    default_name = next(
        (name for name, id in options_dict.items() if id == sub.template_id), None
    )
    # selection template name
    selected_qes_template = (
        ui.select(
            label="Questionnaire Template Name",
            options=list(options_dict.keys()),  # Show names only
            value=default_name,
        )
        .classes("w-full")
        .props("outlined")
    )
    selected_qes_template.on_value_change(
        lambda: _assign_template_id(sub, options_dict.get(selected_qes_template.value))
    )


def _assign_template_id(sub: Submodule, value: int):
    """
    Assigns a template ID to the given submodule and refreshes the question selection UI.

    Args:
        sub (Submodule): The submodule instance to which the template ID will be assigned.
        value (int): The template ID to assign.

    Side Effects:
        Updates the `template_id` attribute of the provided submodule.
        Refreshes the question selection UI to reflect the selected template ID.
    """
    sub.template_id = value
    show_question_selection.refresh(
        sub, True
    )  # Refresh the UI to show the selected template id


async def select_question(
    q: TemplateQuestion, template_id: int, header: str
) -> Optional[TemplateQuestion]:
    """
    Displays a UI for selecting a question and its corresponding answer options.
    Args:
    ----
        q (TemplateQuestion): An instance of TemplateQuestion containing the current question
                              and its associated options.
        template_id (int): The ID of the questionnaire template to filter questions by.
        header (str): The header text to display above the question selection UI.
    Returns:

        TemplateQuestion: The updated TemplateQuestion instance with the selected question
                          and answer options. Returns None if no questions are available.
    Behavior:

        - Displays a header label with the specified text.
        - Loads all available questions using the TemplateQuestionnaire class.
        - If no questions are available, notifies the user and returns None.
        - Provides a dropdown for selecting a question from the loaded list.
        - Dynamically updates the answer options dropdown based on the selected question.
        - Binds the selected question and answer options to the provided TemplateQuestion instance.
    Notes:

        - The `update_answer_options` function dynamically updates the answer options dropdown
          based on the selected question.
        - The `update_answer` function updates the `options` attribute of the TemplateQuestion
          instance with the selected answer option.
    """
    ui.label(header).classes("font-extrabold")
    spinner = create_loader()

    try:
        all_questions = await run.io_bound(
            lambda: TemplateQuestionnaire(
                instance_id=GlobalSettings().active_instance_id
            ).load_all_questions(template_id=template_id)
        )
        await asyncio.sleep(0.5)
    except Exception:
        ui.notify("Failed to load questions", type="negative")
        spinner.delete()
        return None
    finally:
        spinner.delete()

    if all_questions is None or len(all_questions) == 0:
        ui.notify("No questions available", type="negative")
        return None

    question_column = config.get("QUESTIONNAIRE", "QUESTION_COLUMN")
    q_value = (
        q.question if q.question in all_questions[question_column].values else None
    )

    selected_question = (
        ui.select(
            label="Select Question",
            options=all_questions[question_column].tolist(),
            on_change=lambda: update_answer_options(),  # pylint: disable=unnecessary-lambda
            value=q_value,
        )
        .classes("w-full")
        .props("outlined")
    )
    selected_question.bind_value_to(q, "question")

    # Initialize with empty options
    answer_options = (
        get_answer_options(selected_question.value, all_questions)
        if selected_question.value
        else []
    )

    # Set default value only if it exists in current options
    a_value = (
        q.options[0]
        if q.options and q.options[0] in answer_options
        else (answer_options[0] if answer_options else None)
    )

    # Create answer options dropdown
    selected_ans_opt = (
        ui.select(
            label="Select Answer Option",
            options=answer_options,
            value=a_value,
            on_change=lambda: update_answer(),  # pylint: disable=unnecessary-lambda
        )
        .classes("w-full")
        .props("outlined")
    )

    def update_answer():
        # Ensure we only set string values
        if selected_ans_opt.value is not None:
            q.options = [selected_ans_opt.value]
        else:
            q.options = [""]  # Default empty string if no option selected

    def update_answer_options():
        nonlocal answer_options
        answer_options = (
            get_answer_options(selected_question.value, all_questions)
            if selected_question.value
            else []
        )

        # Update dropdown options
        selected_ans_opt.options = answer_options
        selected_ans_opt.value = answer_options[0] if answer_options else None
        selected_ans_opt.update()

        # Update question object
        if answer_options:
            q.options = [answer_options[0]]
        else:
            q.options = [""]  # Default empty string if no options available

    # Initialize question options
    if not q.options or not answer_options:
        q.options = [answer_options[0]] if answer_options else [""]

    return q


def get_answer_options(question: str, all_questions: pd.DataFrame) -> List[str]:
    """
    Retrieve answer options for a given question from a DataFrame.
    Args:
    ----
        question (str): The question for which answer options are to be retrieved.
        all_questions (pd.DataFrame): A DataFrame containing all questions and their corresponding answer options.
    Returns:

        List[str]: A list of answer options for the given question. If no options are found, an empty list is returned.
    """
    try:
        question_col = config.get("QUESTIONNAIRE", "QUESTION_COLUMN")
        options_col = config.get("QUESTIONNAIRE", "OPTIONS")

        filtered_data = all_questions[all_questions[question_col] == question]
        if filtered_data.empty:
            raise ValueError(f"Question {question} not found in data")

        options = filtered_data[options_col].values[0]

        # Handle NA/NaN values
        if pd.isna(options):
            raise ValueError(f"No options available for question {question}")

        # First try to parse as JSON if it's a string
        if isinstance(options, str):
            try:
                options = json.loads(
                    options
                )  # Use json.loads instead of ast.literal_eval
            except json.JSONDecodeError:
                # If not valid JSON, try literal_eval as fallback
                try:
                    options = ast.literal_eval(options)
                except (ValueError, SyntaxError) as eval_exc:
                    raise ValueError(
                        f"Invalid options format for question '{question}': {options}"
                    ) from eval_exc

        # Handle different option structures
        if isinstance(options, dict):
            # Case 1: {"restrictions": {"required": false}}
            if "restrictions" in options:
                raise ValueError(
                    f"Question {question} has restrictions, no options available"
                )
            # Case 2: {"items": [{"name": "option1"}, {"name": "option2"}]}
            if "items" in options:
                return [item["name"] for item in options["items"] if "name" in item]
        elif isinstance(options, list):
            # Case 3: Direct list of options
            return options

        return []
    except Exception as e:
        Status.WARNING(
            f"Error while retrieving answer options for question '{question}': {str(e)}"
        )
        return []
