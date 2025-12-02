# # Copyright (C) KonaAI - All Rights Reserved
"""
This script provides a user interface for displaying and managing MLflow results
using the NiceGUI library.
"""
import asyncio
from collections import defaultdict
from datetime import date
from datetime import datetime
from typing import Dict
from typing import List

import humanize
import pandas as pd
from mlflow.entities import Experiment
from nicegui import run
from nicegui import ui
from src.admin.components.spinners import create_loader
from src.automl.model import FeatureImportance
from src.automl.model_tracker import Model  # Import the Model class
from src.automl.model_tracker import ModelTracker
from src.automl.prediction_pipeline import LastPredictionDetails
from src.automl.tasks import model_monitoring
from src.automl.tasks import predict_from_model
from src.automl.tasks import train_model
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule
from src.utils.task_queue import TaskQueue


class ModelTrackerUI:
    """
    ModelTrackerUI provides an interactive user interface for tracking, managing, and deploying machine learning models and experiments within an AutoML workflow.
    This class enables users to:
    - View and manage modelling experiments, including listing, selecting, and deleting experiments.
    - View, select, and deploy models associated with experiments.
    - Display detailed metrics, parameters, confusion matrices, and feature importances for selected models.
    - Initiate new training experiments and monitor the status of training and prediction tasks.
    - Execute predictions and risk scoring using the currently active model, and view details of the last prediction run.

    The UI is organized into expandable sections for experiments and the active model, and supports asynchronous data loading and UI updates. Integration with background task queues allows for tracking long-running operations such as training and prediction.
    """

    experiments: List[Experiment] = None
    chosen_experiment: str = None
    models: List[Model] = None
    models_metrics: pd.DataFrame = None
    chosen_model_name: str = None
    chosen_model: Model = None
    run_risk_scoring: bool = True

    def __init__(self, submodule: Submodule) -> None:
        self.submodule = submodule
        self.model_tracker = ModelTracker(submodule_obj=submodule)

    async def render(self):
        """
        Asynchronously renders the model tracking user interface.
        Displays two expandable sections:
        1. "Modelling Experiments" - Shows widgets related to modelling experiments.
        2. "Active Model" - Shows widgets for the currently active model.

        Loads experiment data when the "Modelling Experiments" section is expanded.
        """
        # show active model

        with (
            ui.expansion("Modelling Experiments", value=False, icon="sym_s_experiment")
            .classes("w-full border-2 rounded-md")
            .props("outlined") as experiments_exp
        ):
            await self._draw_experiment_widgets()

        with (
            ui.expansion("Active Model", value=False, icon="smart_toy")
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            await self._draw_active_model_widgets()

        with (
            ui.expansion("Model Monitoring", value=False, icon="monitor")
            .classes("w-full border-2 rounded-md")
            .props("outlined")
        ):
            await self._draw_model_monitoring_widgets()

        # load experiments when experiments expansion is opened
        experiments_exp.on_value_change(
            lambda: self._load_experiments()  # pylint: disable=unnecessary-lambda
        )

    @ui.refreshable_method
    async def _draw_active_model_widgets(self):
        """Draw the widgets for the active model"""
        spinner = create_loader()
        await asyncio.sleep(0.5)  # Allow spinner to show up
        ui.label("Active Model not available yet.").bind_visibility_from(
            self.submodule, "active_model", backward=lambda x: not x
        )
        if not self.submodule.active_model:
            spinner.delete()
            return

        # show active model details
        try:
            if model := await run.io_bound(
                lambda: self.model_tracker.get_model_by_name(
                    self.submodule.active_model
                )
            ):
                if isinstance(model, Model):
                    self._draw_model_widgets(model)
            else:
                ui.label("Active Model set but not present.").classes("")
                self.submodule.set_active_model(None)
        except Exception as e:
            ui.notify("Error loading active model. ", type="negative")
            Status.FAILED("Error loading active model", error=e, traceback=False)

        with ui.card().classes("w-full border-2 rounded-md overflow-auto"):
            ui.label("Prediction and Risk Scoring").classes(
                "text-lg font-semibold mb-2"
            )
            ui.separator()
            # track last prediction task
            tq = TaskQueue(self.submodule.instance_id)
            if tid := tq.get_task_id_by_name(self.prediction_task_name):
                # check if task has completed
                complete, _ = tq.task_status_by_id(tid)
                if not complete:
                    ui.label(
                        f"Previously submitted prediction task {tid} is still running..."
                    )

            # Prediction Execution
            ui.checkbox("Run Risk Scoring", value=True).bind_value(
                self, "run_risk_scoring"
            )
            ui.button(
                "Execute Predictions",
                icon="batch_prediction",
                on_click=self._execute_predictions,
            )

            # Last Prediction Data Section
            pdata = LastPredictionDetails(self.submodule)
            if pdata.Date:
                with ui.grid(columns=2).classes("w-full gap-2"):
                    ui.label("Last Prediction Details").classes("font-semibold mb-2")
                    ui.separator()
                    ui.label("Date")
                    ui.label(pdata.Date.strftime("%d %B %Y %H:%M:%S")).classes()
                    ui.label("Model Used")
                    ui.label(pdata.Model).classes()
                    ui.label("Predicted Concern Count")
                    ui.label(pdata.Predicted_Concern_Count).classes()

        spinner.delete()

    @property
    def prediction_task_name(self) -> str:
        """
        Returns the name of the prediction task in the format:
        'AutoML_{module}_{submodule}_Prediction'.

        Returns:
            str: The formatted prediction task name based on the module and submodule.
        """
        return f"AutoML_{self.submodule.module}_{self.submodule.submodule}_Prediction"

    def _execute_predictions(self):
        """Execute the predictions"""
        if task := predict_from_model.delay(
            module=self.submodule.module,
            submodule=self.submodule.submodule,
            instance_id=self.submodule.instance_id,
            risk_scoring=self.run_risk_scoring,
        ):
            tq = TaskQueue(self.submodule.instance_id)
            tq.add_task(self.prediction_task_name, task.id)
            self._draw_active_model_widgets.refresh()
            ui.notify("Prediction started", type="positive")

    @ui.refreshable_method
    async def _draw_experiment_widgets(self):
        """Draw the widgets for the experiments"""
        spinner = create_loader()
        await asyncio.sleep(0.5)  # Allow spinner to show up

        with ui.column().classes("w-full gap-2"):
            ui.label("Experiments not available yet.").bind_visibility_from(
                self, "experiments", backward=lambda x: not x
            )
            if self.experiments:
                selected_experiment = (
                    ui.select(
                        label="Select Experiment",
                        options={
                            exp.name: f"{exp.name} [ID {exp.experiment_id}]"
                            for exp in self.experiments
                        },
                    )
                    .classes("w-full")
                    .props("outlined")
                    .bind_value(self, "chosen_experiment")
                )
                selected_experiment.on_value_change(
                    lambda: self.on_experiment_change()  # pylint: disable=unnecessary-lambda
                )

            if self.chosen_experiment:
                ui.button(
                    "Delete Experiment",
                    icon="delete_forever",
                    on_click=self._delete_experiment,
                    color="red",
                ).classes("justify-end")

            if self.models_metrics is not None and len(self.models_metrics) > 0:
                # show model metrics
                ui.label(f"Selected Experiment: {self.chosen_experiment}").classes(
                    "text-md font-semibold"
                )
                ui.table.from_pandas(
                    self.models_metrics.reset_index(drop=False, names="Model")
                ).classes("w-full").bind_visibility_from(
                    self, "models_metrics", backward=lambda x: len(x) > 0
                )

            if self.models:
                # show models list
                selected_model = (
                    ui.select(
                        label="Select Model",
                        options={model.name: model.name for model in self.models},
                    )
                    .classes("w-full")
                    .props("outlined")
                    .bind_value(self, "chosen_model_name")
                )
                selected_model.on_value_change(
                    lambda: self._set_chosen_model()  # pylint: disable=unnecessary-lambda
                )

            if self.chosen_model:
                self._draw_model_widgets(self.chosen_model)
                ui.button(
                    "Deploy Model",
                    icon="assignment_turned_in",
                    on_click=self._deploy_as_active_model,
                    color="green",
                ).classes("justify-end")

            # Training Section
            with ui.card().classes("w-full border-2 rounded-md overflow-auto"):
                ui.label("Models Training").classes("text-lg font-semibold mb-2")
                ui.separator()
                # track last prediction task
                tq = TaskQueue(self.submodule.instance_id)
                if tid := tq.get_task_id_by_name(self.training_task_name):
                    # check if task has completed
                    complete, _ = tq.task_status_by_id(tid)
                    if not complete:
                        ui.label(
                            f"Previously submitted training task {tid} is still running..."
                        )
                # Training Execution
                ui.button(
                    "Start Experiment",
                    icon="psychology",
                    on_click=self._start_experiment,
                )
        spinner.delete()

    @ui.refreshable_method
    async def _draw_model_monitoring_widgets(self):
        """Draw the widgets for the model monitoring"""
        spinner = create_loader()
        await asyncio.sleep(0.5)  # Allow spinner to show up

        with ui.card().classes("w-full border-2 rounded-md overflow-auto"):
            ui.label("Model Monitoring").classes("text-lg font-semibold mb-2")
            ui.separator()

            # track last monitoring task
            tq = TaskQueue(self.submodule.instance_id)

            if tid := tq.get_task_id_by_name(self.training_task_name):
                # check if task has completed
                complete, _ = tq.task_status_by_id(tid)
                if not complete:
                    with ui.column().classes("w-full p-4 bg-yellow-50 rounded-md"):
                        ui.label(
                            f"Model Monitoring task {tid} is still running..."
                        ).classes("text-yellow-800")

            # Monitoring Execution
            ui.button(
                "Start Monitoring",
                icon="psychology",
                on_click=self._start_monitoring,
            )

            # Display monitoring data if available
            database = (
                GlobalSettings()
                .instance_by_id(self.submodule.instance_id)
                .settings.masterdb
            )
            table_name = "MLS.model_monitoring"

            try:
                # get data from database
                df = database.download_table_or_query(table_name=table_name)

                if df is not None and not df.empty:
                    # Filter the dataframe based on instance_id and submodule
                    df = df.compute()

                    filtered_df = df[
                        (df["instance_id"] == self.submodule.instance_id)
                        & (
                            df["submodule"] == self.submodule.submodule
                        )  # or appropriate submodule identifier
                    ]

                    # Display the table only if filtered data exists
                    if not filtered_df.empty:
                        ui.label("Monitoring Data").classes(
                            "text-lg font-semibold mt-4 mb-2"
                        )
                        ui.table.from_pandas(filtered_df).classes("w-full")
                else:
                    ui.label(
                        "No monitoring data available for this instance and submodule"
                    ).classes("text-gray-500 italic")

            except Exception as e:
                ui.label(f"Error loading monitoring data: {str(e)}").classes(
                    "text-red-500"
                )
                Status.FAILED("Error loading monitoring data", error=e, traceback=False)

        spinner.delete()

    def _delete_experiment(self):
        """Delete the experiment"""
        experiment_id = None
        if self.chosen_experiment:
            experiment_id = next(
                (
                    exp.experiment_id
                    for exp in self.experiments
                    if exp.name == self.chosen_experiment
                ),
                None,
            )

        if not experiment_id:
            ui.notify("Experiment not found", type="negative")
            return

        if self.model_tracker.delete_experiment(experiment_id):
            self._load_experiments()
            ui.notify("Experiment deleted successfully", type="positive")
            self._draw_experiment_widgets.refresh()
        else:
            ui.notify("Failed to delete experiment", type="negative")

    @property
    def training_task_name(self) -> str:
        """
        Returns the name of the training task as a formatted string.

        The training task name is constructed using the module and submodule names
        from the associated submodule object, following the pattern:
        'AutoML_{module}_{submodule}_Training'.

        Returns:
            str: The formatted training task name.
        """
        return f"AutoML_{self.submodule.module}_{self.submodule.submodule}_Training"

    def _start_experiment(self):
        """Start the experiment"""

        if task := train_model.delay(
            module=self.submodule.module,
            submodule=self.submodule.submodule,
            instance_id=self.submodule.instance_id,
        ):
            tq = TaskQueue(self.submodule.instance_id)
            tq.add_task(self.training_task_name, task.id)
            ui.notify("Experiment started", type="positive")
            self._draw_experiment_widgets.refresh()

    async def _start_monitoring(self):
        """Start the monitoring"""

        if task := model_monitoring.delay():
            tq = TaskQueue(self.submodule.instance_id)
            tq.add_task(self.training_task_name, task.id)
            ui.notify("Monitoring started", type="positive")

            # Set monitoring in progress
            self._draw_model_monitoring_widgets.refresh()

            # wait for completion and refresh
            await self._wait_for_moitoring_completion(task.id, self.training_task_name)

    async def _wait_for_moitoring_completion(self, task_id: str):
        """Wait for the monitoring task to complete and refresh the UI"""
        tq = TaskQueue(self.submodule.instance_id)
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            complete, _ = await run.io_bound(lambda: tq.task_status_by_id(task_id))
            if complete:
                await self._load_models()

            self._draw_model_monitoring_widgets.refresh()

    def _deploy_as_active_model(self):
        """Deploy the chosen model as the active model"""
        if self.chosen_model:

            self.submodule.set_active_model(self.chosen_model.name)
            ui.notify(
                f"Model {self.chosen_model.name} deployed successfully", type="positive"
            )
            # Refresh the active model section

            self._draw_active_model_widgets.refresh()

    async def _set_chosen_model(self):
        """Set the chosen model"""
        if self.models:
            self.chosen_model = await run.io_bound(
                lambda: next(
                    (m for m in self.models if m.name == self.chosen_model_name), None
                )
            )
            self._draw_experiment_widgets.refresh()

    def _draw_model_widgets(self, model: Model):
        """Draw the widgets for the model"""
        if not model:
            ui.notify("Model not found", type="negative")
            return

        ui.label(f"Model Name: {model.name}").classes("text-md font-semibold")
        ui.label(f"Created On {model.created_on.strftime('%B %d %Y %H:%M:%S')}")
        ui.label(f"Run ID {model.run_id}")
        self._draw_dictionary(model.model_dump().get("metrics", {}), "Model Metrics")
        self._draw_dictionary(model.model_dump().get("params", {}), "Model Parameters")
        self._draw_dictionary(
            model.serialize_confusion_matrix(model.confusion_matrix).model_dump() or {},
            "Confusion Matrix",
        )
        self._draw_feature_importance(model.model_dump().get("feature_importance", {}))

    def _draw_dictionary(self, data: Dict, title: str) -> None:
        with ui.card().classes("w-full border-2 rounded-md"):
            ui.label(title).classes("text-lg font-semibold mb-2")
            ui.separator()
            with ui.grid(columns=2).classes("w-full gap-4"):
                for key, value in data.items():
                    if key in ["features_to_keep", "features_to_drop"]:
                        continue

                    if isinstance(value, (int, float)):
                        value = f"{value:.3f}"
                    elif isinstance(value, (datetime, date)):
                        value = humanize.naturaldate(value)
                    elif isinstance(value, str):
                        value = value.replace("_", " ").title()
                    else:
                        continue

                    ui.label(key.replace("_", " ").title()).classes("font-medium")
                    ui.label(str(value)).classes(
                        "flex-1 text-gray-800 bg-gray-50 px-3 py-1 rounded"
                    )

    def _generate_hsl_color(self, index: int, total: int) -> str:
        """Generate an HSL color based on the index and total number of items."""
        return f"hsl({index * 360 / total}, 70%, 50%)"

    def _draw_feature_importance(
        self, feature_importance: List[FeatureImportance]
    ) -> None:
        """Draw the feature importance chart"""
        # filter out features with zero importance or None feature names
        feature_importance = [
            fi
            for fi in feature_importance
            if isinstance(fi.importance, (int, float))
            and fi.importance != 0.0
            and fi.feature_name is not None
        ]

        # sort feature importance by importance value
        feature_importance = sorted(feature_importance, key=lambda x: x.importance)

        features = [fi.feature_name for fi in feature_importance]
        feature_types = [fi.type for fi in feature_importance]
        # descriptions = [fi.description for fi in feature_importance]

        # Define unique feature types and assign colors
        unique_types = set(feature_types)  # Get unique feature types
        # Assign colors to each feature type (you can customize these colors)
        type_color_map = {
            # Generate distinct colors for each feature type using HSL.
            # The lightness (70%) and saturation (50%) values were chosen to ensure
            # the colors are visually distinct and balanced for most displays.
            type_name: self._generate_hsl_color(i, len(unique_types))
            for i, type_name in enumerate(unique_types)
        }

        # Group features by type
        type_to_data = defaultdict(
            lambda: [0] * len(features)
        )  # Placeholder for stacking
        for idx, _ in enumerate(features):
            t = feature_importance[idx].type
            type_to_data[t][idx] = feature_importance[idx].importance

        # Create one series per feature type
        series_data = []
        series_data.extend(
            {
                "name": t,
                "type": "bar",
                "stack": "total",  # Optional stacking
                "itemStyle": {"color": type_color_map[t]},
                "data": type_to_data[t],
            }
            for t in unique_types
        )

        # chart configuration
        chart_config = {
            "title": False,
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"},
            },
            "legend": {
                "show": True,
                "type": "scroll",  # Optional: scroll if too many types
                "selectedMode": True,  # Allow legend interaction
            },
            "yAxis": {
                "type": "category",
                "data": features,
                "axisLabel": {
                    "overflow": "break",
                    "interval": 0,
                },
            },
            "xAxis": {
                "type": "value",
                "name": "Importance",
                "nameLocation": "middle",
                "nameGap": 30,
                "nameTextStyle": {
                    "align": "center",
                    "verticalAlign": "top",
                },
            },
            "series": series_data,  # Use the series data created above
            "grid": {"containLabel": True},
        }

        total_features = len(features)
        single_feature_height = 22  # Height of each feature row in pixels
        min_height = 400  # Minimum height for the chart
        max_height = 1200  # Maximum height for the chart
        adjusted_total_height = min(
            max(min_height, total_features * single_feature_height), max_height
        )  # Adjusted total_features * single_feature_height
        with ui.card().classes("w-full border-2 rounded-md overflow-auto"):
            ui.label("Feature Importance").classes("text-lg font-semibold mb-2")
            ui.separator()
            ui.echart(chart_config, renderer="svg").classes("w-full").style(
                f"height: {adjusted_total_height}px; overflow-y: auto;"
            )

    async def on_experiment_change(self):
        """
        Handles the event when the selected experiment changes.

        If an experiment is selected (`self.chosen_experiment` is truthy),
        asynchronously loads the models associated with the new experiment.

        Returns:
            None
        """
        if not self.chosen_experiment:
            return
        await self._load_models()

    async def _load_models(self):
        """Load the models from the model tracker"""
        if not self.chosen_experiment:
            return

        self.models = None
        self.models_metrics = None
        self.chosen_model_name = None
        self.chosen_model = None

        try:
            experiment = [
                exp for exp in self.experiments if exp.name == self.chosen_experiment
            ][0]

            models = await run.io_bound(
                lambda: (
                    self.model_tracker.get_ml_models([experiment])
                    if experiment
                    else None
                )
            )
            if models:
                self.models = models
                self.models_metrics = await run.io_bound(
                    lambda: self.model_tracker.all_model_metrics(self.models)
                )
                self.models_metrics = self.models_metrics.sort_values(
                    by="F1", ascending=False
                ).round(3)
        except Exception as e:
            ui.notify(f"Failed to load models: {e}", type="negative")
        finally:
            self._draw_experiment_widgets.refresh()

    async def _load_experiments(self):
        """Load the experiments from the model tracker"""

        self.experiments = None
        if experiments := await run.io_bound(lambda: self.model_tracker.ml_experiments):
            self.experiments = experiments
            self._draw_experiment_widgets.refresh()
