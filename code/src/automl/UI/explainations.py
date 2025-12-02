# # Copyright (C) KonaAI - All Rights Reserved
"""Prediction Explainer UI"""
import asyncio
from typing import List

import numpy as np
import pandas as pd
from nicegui import run
from nicegui import ui
from src.admin.components.spinners import create_loader
from src.admin.components.spinners import create_overlay_spinner
from src.automl.explainer import ExplainationOutput
from src.automl.explainer import ExplainedFeature
from src.automl.explainer import PredictionExplainer
from src.automl.prediction_pipeline import PredictionPipeline
from src.automl.utils import config
from src.utils.global_config import GlobalSettings
from src.utils.submodule import Submodule


class PredictionExplainerUI:
    """
    UI class for displaying model prediction explanations.

    Attributes:
    -----------
        prediction_table_exists (bool): Indicates if the prediction table exists.
        sample_transactions (pd.DataFrame): DataFrame containing sample transactions for explanation.
        explanation (ExplainationOutput): The explanation output for the selected transaction.
        selected_transaction (str): The currently selected transaction ID.
    Args:
        submodule (Submodule): The submodule instance associated with the UI.

    """

    prediction_table_exists: bool = False
    sample_transactions: pd.DataFrame = pd.DataFrame()
    explanation: ExplainationOutput = None
    selected_transaction: str = None

    def __init__(self, submodule: Submodule) -> None:
        self.submodule = submodule

    async def render(self) -> None:
        """
        Asynchronously renders the UI section for model explanations.

        This method displays an expandable section titled "Explainations" with custom styling.
        When expanded, it draws additional widgets related to model explanations.
        It also sets up a callback to check if the prediction table exists whenever the expansion state changes.
        """

        with (
            ui.expansion("Explainations", value=False, icon="insights")
            .classes("w-full border-2 rounded-md")
            .props("outlined") as exp
        ):
            await self._draw_widgets()
        exp.on_value_change(
            lambda: self._prediction_table_exists()  # pylint: disable=unnecessary-lambda
        )

    @ui.refreshable_method
    async def _draw_widgets(self):
        """
        Asynchronously draws the UI widgets for displaying transaction explanations.
        This method manages the display of prediction status, sample transactions, transaction selection, and explanation details.
        It shows a loading spinner if explanations are not yet available, handles the case when no sample transactions exist,
        and dynamically updates the UI based on user interactions such as selecting a transaction or requesting an explanation.
        Widgets rendered include:
        - Status labels for predictions and sample transactions
        - A table of sample transactions
        - A dropdown for selecting a transaction
        - A button to trigger explanation generation
        - Explanation details and features in tabular and chart form
        Visibility and interactivity of widgets are bound to the state of the instance (e.g., prediction table existence, sample transactions).
        """
        spinner = None
        if not self.explanation:
            spinner = create_loader()
            await asyncio.sleep(0.5)
        ui.label("Predictions not available yet.").bind_visibility_from(
            self,
            "prediction_table_exists",
            backward=lambda x: not x,  # pylint: disable=unnecessary-lambda
        )

        with ui.column(align_items="center").classes(
            "w-full gap-2"
        ) as transaction_area:
            if self.sample_transactions is None or len(self.sample_transactions) == 0:
                ui.label("No sample transactions available.").classes("text-red-500")
                spinner.delete()
                return

            # show sample transactions
            sample_trans = ui.table.from_pandas(self.sample_transactions).classes()

            # transaction selection for explanation
            index_col = config.get("DATA", "INDEX")
            trans_options = []
            if index_col in self.sample_transactions.columns:
                trans_options = self.sample_transactions[index_col].values.tolist()

            selected_tran = (
                ui.select(
                    label="Select Transaction",
                    options=trans_options,
                    new_value_mode="add-unique",
                )
                .classes("w-full")
                .props("outlined")
                .bind_value(self, "selected_transaction")
            )
            # Clear explanation when transaction selection changes
            selected_tran.on_value_change(
                lambda: self._clear_explanation()  # pylint: disable=unnecessary-lambda
            )

            ui.button("Explain Transaction", icon="search").classes("w-full").props(
                "outlined"
            ).on_click(lambda: self._explain_transaction(selected_tran.value)).classes(
                "mb-2"
            )

            # show the selected transaction and its explanation
            if self.explanation:
                with ui.grid(columns=2).classes("w-full gap-2 mb-2"):
                    for key, value in self.explanation.model_dump().items():
                        if not value or key == "features":
                            continue

                        ui.label(str(key).replace("_", " ").title()).classes(
                            "font-bold"
                        )
                        ui.label(str(value)).classes("")

            # show details of the selected transaction features
            if self.explanation and self.explanation.features:
                self._draw_echart(self.explanation.features)
                ui.separator()
                features: pd.DataFrame = pd.DataFrame(
                    self.explanation.model_dump().get("features")
                )
                features.columns = [
                    str(col).replace("_", " ").title() for col in features.columns
                ]
                ui.table.from_pandas(features).classes("w-full").props(
                    "outlined"
                ).style("text-align: left;")

            # some empty space
            ui.space()

        # show widgets only if prediction table exists
        transaction_area.bind_visibility_from(self, "prediction_table_exists")

        # show further widgets only if sample transactions exist
        sample_trans.bind_visibility_from(
            self, "sample_transactions", backward=lambda x: len(x) > 0
        )
        selected_tran.bind_visibility_from(
            self, "sample_transactions", backward=lambda x: len(x) > 0
        )
        if spinner:
            spinner.delete()

    def _clear_explanation(self):
        self.explanation = None
        self._draw_widgets.refresh()

    def _draw_echart(self, data: List[ExplainedFeature]):
        if not data:
            return

        data = [d.model_dump() for d in data]

        chart_config = {
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
            },
            "xAxis": {
                "type": "value",
                "name": "Contribution %",
                "nameLocation": "middle",
                "nameGap": 40,
            },
            "yAxis": {
                "type": "category",
                "data": [item["name"] for item in data],
                "name": "Features",
                "axisLabel": {
                    "hideOverlap": False,
                    "overflow": "truncate",
                    "interval": 0,
                    "ellipsis": "...",
                    "rotate": 30,
                },
            },
            "series": [
                {
                    "type": "bar",
                    "data": [
                        {
                            "value": item["contribution_percent"],
                            "name": item["name"],
                            "description": item["description"],
                            "itemStyle": {
                                "color": (
                                    "green"
                                    if item["contribution_percent"] < 0
                                    else "red"
                                ),
                            },
                        }
                        for item in data
                    ],
                    # 'label': {
                    #     'show': True,
                    #     'position': 'center',
                    #     'verticalAlign': 'middle',
                    #     'formatter': '{c}%'
                    # },
                    "grid": {
                        "containLabel": False,
                        "left": "70%",
                        "right": "20px",
                    },
                }
            ],
        }

        ui.echart(
            chart_config,
        ).classes(
            "w-full overflow-auto mb-2"
        ).style(f"height: {len(data)*40}px;")

    def _prediction_table_exists(self):
        """Check if the prediction table exists"""
        table_name = PredictionPipeline(self.submodule).prediction_table

        instance = GlobalSettings().instance_by_id(self.submodule.instance_id)
        if not instance:
            return

        result = instance.settings.projectdb.does_table_exist(table_name=table_name)
        self.prediction_table_exists = result
        if result:
            self._chose_sample_transactions()
            self._draw_widgets.refresh()

    def _chose_sample_transactions(self) -> None:
        """Choose sample transactions from the prediction table"""
        self.sample_transactions = None

        # get index column, prediction column and probability column from config
        index_col = config.get("DATA", "INDEX")
        prediction_col = config.get("OUTPUT", "Prediction_Column")
        probability_col = config.get("OUTPUT", "Prediction_Probability_Column")
        table_name = PredictionPipeline(self.submodule).prediction_table

        # download the prediction data
        query = (
            f"SELECT {index_col}, {prediction_col}, {probability_col} FROM {table_name}"
        )

        instance = GlobalSettings().instance_by_id(self.submodule.instance_id)
        if not instance:
            return

        instance_db = instance.settings.projectdb
        dff = instance_db.download_table_or_query(query=query)
        if dff is None or len(dff) == 0:
            return

        df = dff.compute()

        # chose one transaction per bin of the prediction probability
        sample_transactions = pd.DataFrame()
        _min = df[probability_col].min()
        _max = df[probability_col].max()
        _bins = np.linspace(_min, _max, 10)
        _bins = list(zip(_bins[:-1], _bins[1:]))
        for _min, _max in _bins:
            # select first transaction between min and max
            trans = df[
                (df[probability_col] >= _min) & (df[probability_col] <= _max)
            ].head(1)
            sample_transactions = pd.concat([sample_transactions, trans])

        if len(sample_transactions) > 0:
            sample_transactions = sample_transactions.drop_duplicates(
                subset=[index_col]
            )
            sample_transactions[probability_col] = sample_transactions[
                probability_col
            ].apply(round, args=[2])
            self.sample_transactions = sample_transactions

    async def _explain_transaction(self, transaction_id: str) -> None:
        """Explain the transaction with the given ID"""
        # Create an overlay spinner in the current context
        spinner = create_overlay_spinner("Loading explanation...")

        explainer = PredictionExplainer(self.submodule, transaction_id)
        if explanation := await run.io_bound(explainer.explain):
            self.explanation = explanation

        # # Remove the spinner
        spinner.delete()

        self._draw_widgets.refresh()
