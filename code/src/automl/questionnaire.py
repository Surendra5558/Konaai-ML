# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to manage the questionnaire data"""
import ast
from typing import List

import pandas as pd
from pydantic import BaseModel
from src.automl.utils import config
from src.utils.global_config import GlobalSettings
from src.utils.status import Status


class TemplateQuestion(BaseModel):
    """
    Represents a single question in a questionnaire template.
    Attributes:
    ---------
        question (str): The text of the question.
        options (List[str]): A list of possible answer options for the question.
    """

    question: str = None
    options: List[str] = (
        []
    )  # original plan was to allow multiple, but now we allow only one.


class TemplateQuestionnaire:
    """
    TemplateQuestionnaire is a class for managing and retrieving questionnaire data from a database based on a specific instance ID.
    Attributes:
    ----------
        instance_id (str): The unique identifier for the instance whose questionnaire data is to be accessed.
    """

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id

    def load_all_questions(self, template_id: int = None) -> pd.DataFrame:
        """
        Loads all questions from the questionnaire database for the specified template.
        Connects to the database using the instance ID, retrieves questionnaire data based on a configured query,
        and loads the data into a pandas DataFrame. The method filters the questions to include only those with
        specific render types (radio button or check box) or those matching the provided template ID.
        Args:
        -----
            template_id (int, optional): The ID of the questionnaire template to filter questions by. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered questions. Returns an empty DataFrame if no data is found.
        """
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            return pd.DataFrame()
        instance_db = instance.settings.projectdb
        query = config.get("QUESTIONNAIRE", "QUESTION_QUERY")

        ddf = instance_db.download_table_or_query(query=query)
        if ddf is None or len(ddf.index) == 0:
            # No data found, return empty DataFrame
            return pd.DataFrame()

        # load data from file
        data = ddf.compute()
        render_type_column = config.get("QUESTIONNAIRE", "RENDER_TYPE")
        template_id_column = config.get(
            "QUESTIONNAIRE", "QUESTIONNAIRE_TEMPLATE_ID_COLUMN"
        )
        radio_button = ast.literal_eval(config.get("QUESTIONNAIRE", "RADIO_BUTTON"))
        check_box = ast.literal_eval(config.get("QUESTIONNAIRE", "CHECK_BOX"))

        # filter data based on render type
        return data[
            (data[render_type_column] == radio_button)
            | (data[render_type_column] == check_box)
            | (data[template_id_column] == template_id)
        ]

    def load_questionnaire_template_name(self):
        """
        Loads the questionnaire template names from the database for the current instance.
        This method retrieves the questionnaire template names by executing a predefined query
        from the configuration on the instance's project database. If the instance is not found,
        it returns an empty DataFrame and sets the status to NOT_FOUND.

        Returns:
            pd.DataFrame: A DataFrame containing the questionnaire template names if found,
                          otherwise an empty DataFrame.
        """
        instance = GlobalSettings.instance_by_id(self.instance_id)
        if not instance:
            Status.NOT_FOUND("Instance not found", instance_id=self.instance_id)
            return pd.DataFrame()
        instance_db = instance.settings.projectdb
        query = config.get("QUESTIONNAIRE", "QUESTIONNAIRE_TEMPLATE_NAME_QUERY")

        ddf = instance_db.download_table_or_query(query=query)
        return pd.DataFrame() if ddf is None or len(ddf.index) == 0 else ddf.compute()
