# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the tasks for the automl submodule"""
import pathlib
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from src.automl.fetch_data import ModelData
from src.automl.model import Model
from src.automl.model_tracker import ModelTracker
from src.automl.montioring import ModelPerformanceMonitor
from src.automl.prediction_pipeline import LastPredictionDetails
from src.automl.prediction_pipeline import PredictionPipeline
from src.automl.training_pipeline import TrainingPipeline
from src.automl.utils import config
from src.butler.celery_app import celery
from src.butler.celery_result_backend import ModelMonitoring
from src.tools.application_callback import APPLICATION_STATUS
from src.tools.application_callback import call_etl_update_endpoint
from src.tools.dask_tools import compute
from src.utils.global_config import GlobalSettings
from src.utils.metadata import Metadata
from src.utils.notification import EmailNotification
from src.utils.status import Status
from src.utils.submodule import Submodule


@celery.task(
    name="AutoML - Train Models",
    soft_time_limit=int(timedelta(hours=36).total_seconds()),
    bind=True,
)
def train_model(self, module: str, submodule: str, instance_id: str):
    """
    Starts the training process for a specified module and submodule instance.
    This function orchestrates the end-to-end training workflow, including:
    - Initializing the training environment and parameters.
    - Sending email notifications at the start and end (or failure) of training.
    - Collecting and validating training data.
    - Running the training pipeline and fitting models.
    - Handling errors and reporting training status.
    Args:
    ----
        module (str): The name of the module to train.
        submodule (str): The name of the submodule to train.
        instance_id (str): The unique identifier for the training instance.

    Returns:
        dict: A dictionary representing the final status of the training task.

    Raises:
        RuntimeError: If training data cannot be collected or no models are trained.
        Exception: For any other errors encountered during training.
    """
    task_id = self.request.id
    notifier = EmailNotification(instance_id=instance_id)
    try:
        # collect training data
        start_time = datetime.now(timezone.utc)
        submodule_obj = Submodule(module, submodule, instance_id)
        s = Status.INFO(
            "Starting training",
            task_id=task_id,
            start_time=start_time,
            training_parameters=submodule_obj.model_dump(by_alias=True),
        )

        # create notifier
        notifier.add_content("Training Parameters", s.to_dict())
        notifier.send(f"{module} {submodule} AutoML Training Started")

        data = ModelData(submodule_obj=submodule_obj)
        X, y, _error = data.get_training_data()

        if _error is not None:
            raise RuntimeError(_error.message)

        # create training pipeline
        pipeline = TrainingPipeline(submodule_obj=submodule_obj)
        models = pipeline.fit(X, y)

        if not pipeline.training_complete:
            raise RuntimeError("Contact support.")

        # update submodule with trained models
        s = Status.SUCCESS(
            "Training complete",
            task_id=task_id,
            module=submodule_obj.module,
            submodule=submodule_obj.submodule,
            instance_id=submodule_obj.instance_id,
            total_models_trained=len(models),
            experiment_name=(pipeline.experiment.name if pipeline.experiment else None),
            experiment_id=(
                pipeline.experiment.experiment_id if pipeline.experiment else None
            ),
            concern_questionnaire=[
                q.model_dump() for q in submodule_obj.concern_questionnaire
            ],
            no_concern_questionnaire=[
                q.model_dump() for q in submodule_obj.no_concern_questionnaire
            ],
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
        )

        # send notification
        notifier.add_content("Training Outcome", s.to_dict())
        notifier.send(f"{module} {submodule} AutoML Training Completed")

        return s.task_status()
    except Exception as _e:
        s = Status.FAILED(
            f"Training failed. {str(_e)}",
            task_id=task_id,
            module=submodule_obj.module,
            submodule=submodule_obj.submodule,
            instance_id=submodule_obj.instance_id,
            concern_questionnaire=[
                q.model_dump() for q in submodule_obj.concern_questionnaire
            ],
            no_concern_questionnaire=[
                q.model_dump() for q in submodule_obj.no_concern_questionnaire
            ],
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
        )

        # send notification
        if s.log_file_path:
            notifier.attach(s.log_file_path)
        notifier.add_content("Failure Details", s.to_dict())
        notifier.send(f"{module} {submodule} AutoML Training Failed")

        return s.task_status()


@celery.task(
    name="AutoML - Predict",
    soft_time_limit=int(timedelta(hours=36).total_seconds()),
    bind=True,
)
def predict_from_model(
    self, module: str, submodule: str, instance_id: str, risk_scoring: bool = False
):
    """
    Starts the prediction process using the selected model for a given module, submodule, and instance.

    This function performs the following steps:
    1. Retrieves the instance and validates its existence.
    2. Updates the ETL endpoint to indicate prediction is in progress.
    3. Loads the active model for the specified submodule.
    4. Sends a notification that prediction has started.
    5. Collects and prepares input data for prediction.
    6. Runs the prediction pipeline on the input data.
    7. Sends a notification with the prediction outcome and results.
    8. Optionally, performs risk scoring if requested and updates the ETL endpoint accordingly.
    9. Handles all exceptions by updating the ETL endpoint, sending failure notifications, and returning failure status.
    Args:
    ----
        module (str): The name of the module for which prediction is to be performed.
        submodule (str): The name of the submodule for which prediction is to be performed.
        instance_id (str): The unique identifier of the instance.
        risk_scoring (bool, optional): Whether to run risk scoring after prediction. Defaults to False.

    Returns:
        dict: The status of the prediction task, including details such as success/failure, timing, and error information if any.
    """
    task_id = getattr(getattr(self, "request", None), "id", None)
    notifier = EmailNotification(instance_id=instance_id)

    instance = GlobalSettings.instance_by_id(instance_id)
    if not instance:
        raise ValueError(
            f"Instance with ID {instance_id} not found. Cannot proceed with AutoML Prediction."
        )

    submodule_obj = Submodule(module, submodule, instance_id)
    try:
        # update etl endpoint
        call_etl_update_endpoint(
            endpoint=GlobalSettings()
            .instance_by_id(instance_id)
            .settings.automl_callback_endpoint,
            instance=GlobalSettings().instance_by_id(instance_id),
            submodule=Submodule(module, submodule, instance_id),
            task_id=task_id,
            status=APPLICATION_STATUS.IN_PROGRESS,
            description="automl",
        )

        start_time = datetime.now(timezone.utc)

        model = submodule_obj.active_model
        if not model:
            raise NameError("No active model found.")

        # load model
        mt = ModelTracker(submodule_obj=submodule_obj)
        model_obj: Model = mt.get_model_by_name(model_name=model)
        if model_obj is None or not model_obj.model:
            raise FileNotFoundError("Model file not found.")

        s = Status.INFO(
            "Starting prediction",
            task_id=task_id,
            module=module,
            submodule=submodule,
            instance_id=instance_id,
            model=model,
            start_time=start_time,
        )

        # create notifier
        notifier.add_content("Prediction Parameters", s.to_dict())
        notifier.send(f"{module} {submodule} AutoML Prediction Started")

        # collect data
        data = ModelData(submodule_obj=submodule_obj)
        X = data.get_submodule_data(archived_alerts=False)
        X = X.drop_duplicates()
        X = compute(X)
        if X is None or len(X.index) == 0:
            raise SystemError("No input data found for prediction.")

        if not PredictionPipeline(submodule_obj=submodule_obj).predict(X):
            raise RuntimeError("Contact support.")

        s = Status.SUCCESS(
            "Prediction complete",
            task_id=task_id,
            module=module,
            submodule=submodule,
            model=model,
            instance_id=instance_id,
            run_risk_scoring=risk_scoring,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
        )

        # send notification
        notifier.add_content("Prediction Outcome", s.to_dict())
        # get last prediction data
        pdata = LastPredictionDetails(submodule_obj)
        if pdata.Date:
            notifier.add_content("Prediction Result", pdata.model_dump())
        else:
            notifier.add_content(
                "Prediction Result", "Prediction results could not be validated."
            )

        notifier.send(
            f"{module} {submodule} AutoML Prediction Completed. Risk Scoring Pending."
        )

        # sourcery skip: merge-nested-ifs
        if risk_scoring:
            if automl_risk_scoring(
                module=module, submodule=submodule, instance_id=instance_id
            ):
                call_etl_update_endpoint(
                    endpoint=instance.settings.automl_callback_endpoint,
                    instance=instance,
                    submodule=submodule_obj,
                    task_id=task_id,
                    status=APPLICATION_STATUS.SUCCESS,
                    description="automl",
                )
            else:
                raise RuntimeError("Risk scoring failed. Contact support.")

        return s.task_status()
    except Exception as _e:
        call_etl_update_endpoint(
            endpoint=instance.settings.automl_callback_endpoint,
            instance=instance,
            submodule=submodule_obj,
            task_id=task_id,
            status=APPLICATION_STATUS.FAILED,
            description="automl",
        )

        s = Status.FAILED(
            "Prediction failed",
            task_id=task_id,
            module=module,
            submodule=submodule,
            instance_id=instance_id,
            model=model,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            error=_e,
            traceback=False,
        )

        # send notification
        notifier.add_content("Failure Details", s.to_dict())
        if s.log_file_path:
            notifier.attach(s.log_file_path)
        notifier.send(f"{module} {submodule} AutoML Prediction Failed")

        return s.task_status()


@celery.task(
    name="AutoML - Monitoring",
    soft_time_limit=int(timedelta(hours=36).total_seconds()),
    bind=True,
)
def model_monitoring(self):
    """This function is used for Active model monitoring"""
    task_id = getattr(getattr(self, "request", None), "id", None)
    Status.INFO("Starting model monitoring task", task_id=task_id)

    instance_id = None
    module = None
    submodule_name = None
    start_time = datetime.now(timezone.utc)

    try:  # pylint: disable=too-many-nested-blocks
        Status.INFO("Starting model monitoring", task_id=task_id)

        # Get all instances
        instances = GlobalSettings().instances
        if not instances:
            raise ValueError("No instances found for model monitoring.")

        for instance in instances:
            instance_id = instance.instance_id

            # get all module and submodules for the instance
            Status.INFO(
                "Fetching submodules for instance",
                instance_id=instance_id,
            )

            # Get all modules for the instance
            metadata = Metadata(instance_id=instance_id)
            modules = metadata.modules

            for module in modules:
                submodules = metadata.get_submodule_names(module)
                for submodule_name in submodules:
                    # Create Submodule object using the actual class
                    sub = Submodule(module, submodule_name, instance_id)
                    s = Status.INFO(
                        "Starting monitoring",
                        task_id=task_id,
                        module=module,
                    )

                    # Initialize the monitor
                    monitor = ModelPerformanceMonitor(sub)

                    # Call the monitoring method - now returns a dictionary
                    monitoring_results = monitor.monitor_active_model()

                    # Check if monitoring was successful
                    if monitoring_results["old_f1_score"] is None:
                        # monitor_active_model failed and logged the reason inside itself.
                        Status.INFO(
                            f"Skipping performance checks for {sub.module} - {sub.submodule} because monitor_active_model returned no results."
                        )
                        continue  # Skip to the next submodule

                    # Extract values from the results dictionary
                    old_f1 = monitoring_results["old_f1_score"]
                    new_f1 = monitoring_results["new_f1_score"]
                    f1_change_value = monitoring_results["percentage_change"]
                    concern_count = monitoring_results["concern_count"]
                    no_concern_count = monitoring_results["no_concern_count"]

                    # Check for performance changes
                    if f1_change_value < -5:
                        Status.WARNING(
                            f"Model monitoring - Performance drop detected for submodule {sub.module} - {sub.submodule}. "
                            f"F1 score decreased by {abs(f1_change_value):.2f}%",
                            instance_id=instance_id,
                            task_id=task_id,
                            old_f1_score=old_f1,
                            new_f1_score=new_f1,
                            f1_change=f1_change_value,
                            concern_count=concern_count,
                            no_concern_count=no_concern_count,
                        )

                        continue

                    if f1_change_value > 5:

                        save_data = ModelMonitoring(
                            task_id=task_id,
                            instance_id=instance_id,
                            module=module,
                            submodule=submodule_name,
                            model_name=sub.active_model,
                            active_f1_score=old_f1,
                            new_f1_score=new_f1,
                            f1_score_change=f1_change_value,
                            concern_count=concern_count,  # Add these new fields if your ModelMonitoring class supports them
                            no_concern_count=no_concern_count,
                            date_run=datetime.now(timezone.utc),
                        )

                        if not save_data.upsert():
                            Status.FAILED(
                                f"Failed to save monitoring data for {sub.module} - {sub.submodule}"
                            )
                        s = Status.SUCCESS(
                            "Model monitoring - Performance increase detected.",
                            task_id=task_id,
                            instance_id=instance_id,
                            module=module,
                            submodule=submodule_name,
                            model_name=sub.active_model,
                            active_f1_score=old_f1,
                            new_f1_score=new_f1,
                            f1_score_change=f1_change_value,
                            concern_count=concern_count,
                            no_concern_count=no_concern_count,
                            start_time=start_time,
                            end_time=datetime.now(timezone.utc),
                        )
                        notifier = EmailNotification(instance_id=instance_id)
                        notifier.add_content("Monitoring Parameters", s.to_dict())
                        highlight = {
                            "ACTION REQUIRED": "The newly calculated performance (New F1) significantly exceeds the Active Model's performance (Active F1). Consider promoting/deploying the new model immediately to capture this gain.",
                        }
                        notifier.add_content(
                            "Performance Action - Highlighter", highlight
                        )
                        notifier.send(
                            f"AutoML Model Monitoring -{sub.module}-{sub.submodule}"
                        )

                    Status.SUCCESS(
                        f"Completed model monitoring for {module} - {submodule_name}",
                        instance_id=instance_id,
                    )

    except Exception as e:
        Status.FAILED(
            "Model monitoring task failed",
            task_id=task_id,
            error=e,
            module=module,
            submodule=submodule_name,
            instance_id=instance_id,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
        )
    Status.SUCCESS(
        "Model monitoring completed for all instances/modules/submodules.",
        task_id=task_id,
    )


def automl_risk_scoring(module: str, submodule: str, instance_id: str) -> bool:
    """
    Executes the AutoML risk scoring process for a given module, submodule, and instance.

    This function performs the following steps:
    1. Initializes an email notifier for process updates.
    2. Logs the start of the risk scoring process and sends a notification.
    3. Loads and formats a SQL query for risk scoring using parameters from the submodule and configuration.
    4. Executes the risk scoring query on the instance's project database.
    5. Validates that prediction data has been updated in the UDM table.
    6. Logs and notifies the outcome (success or failure) of the process.
    Args:
    ----
        module (str): The name of the module for which risk scoring is to be performed.
        submodule (str): The name of the submodule for which risk scoring is to be performed.
        instance_id (str): The unique identifier for the instance.

    Returns:
        bool: True if the risk scoring process completes successfully, False otherwise.

    Raises:
        FileNotFoundError: If the risk scoring SQL file is not found.
        SystemError: If the risk scoring query execution fails.
        RuntimeError: If the prediction data is not updated in the UDM table.
        Exception: For any other unexpected errors during the process.
    """
    notifier = EmailNotification(instance_id=instance_id)
    try:
        start_time = datetime.now(timezone.utc)

        s = Status.INFO(
            "AutoML Risk scoring started",
            module=module,
            submodule=submodule,
            instance_id=instance_id,
            start_time=start_time,
        )

        # create notifier
        notifier.add_content("Process Parameters", s.to_dict())
        notifier.send(f"{module} {submodule} AutoML Risk Scoring Started")

        submodule_obj = Submodule(module, submodule, instance_id)
        udm_table_name = submodule_obj.get_data_table_name()

        # Construct the SQL query to execute the stored procedure
        # get current file path
        file_dir_path = pathlib.Path(__file__).parent.absolute()
        sql_file_path = pathlib.Path(file_dir_path, "risk_scoring.sql").absolute()

        sql_query = None
        with open(sql_file_path, encoding="utf-8") as file:
            sql_query = file.read()
        if sql_query is None:
            raise FileNotFoundError("Risk Scoring SQL file not found.")

        sql_query = sql_query.format(
            UDM_TABLE_NAME=udm_table_name,
            Module=module,
            SubModule=submodule,
            PredictionTableName=PredictionPipeline(submodule_obj).prediction_table,
            PredictionColumnName=config.get("OUTPUT", "Prediction_Column"),
            PredictionProbabilityColumnName=config.get(
                "OUTPUT", "Prediction_Probability_Column"
            ),
            ML_Risk_Score_Column=config.get("OUTPUT", "ML_Risk_Score_Column"),
        )

        # Execute the query
        instance = GlobalSettings().instance_by_id(instance_id)
        if not instance:
            Status.NOT_FOUND(
                "Instance not found",
                instance_id=instance_id,
                alert_status=submodule_obj.alert_status,
            )
            return Status.NOT_FOUND("Instance not found").task_status()
        instance_db = instance.settings.projectdb
        Status.INFO(
            "Executing risk scoring query",
            module=module,
            submodule=submodule,
            instance_id=instance_id,
        )
        if instance_db.execute_query(sql_query):
            Status.INFO(
                "Risk scoring completed",
                module=module,
                submodule=submodule,
                instance_id=instance_id,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
            )
        else:
            raise SystemError("Contact support.")

        if not is_prediction_updated(submodule_obj):
            raise RuntimeError("Prediction data not updated in UDM table.")

        s = Status.SUCCESS(
            "AutoML Risk scoring validation successful.",
            module=module,
            submodule=submodule,
            instance_id=instance_id,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
        )

        # send notification
        notifier.add_content("Risk Scoring Outcome", s.to_dict())
        notifier.send(f"{module} {submodule} AutoML Risk Scoring Completed")

        return True
    except Exception as _e:
        s = Status.FAILED(
            "AutoML Risk Scoring failed",
            module=module,
            submodule=submodule,
            instance_id=instance_id,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            error=_e,
        )

        # send notification
        notifier.add_content("Failure Details", s.to_dict())
        if s.log_file_path:
            notifier.attach(s.log_file_path)
        notifier.send(f"{module} {submodule} AutoML Risk Scoring Failed")

        return False


def is_prediction_updated(submodule_obj: Submodule) -> bool:
    """
    Check if the prediction is updated for a given submodule.

    Args:
        submodule_obj (Submodule): The submodule object.

    Returns:
        bool: True if the prediction is updated, False otherwise.
    """
    prediction_col = config.get("OUTPUT", "Prediction_Column")
    udm_table_name = submodule_obj.get_data_table_name()

    prediction_data = LastPredictionDetails(submodule_obj).model_dump()
    if not prediction_data:
        return False

    query2 = config.get("OUTPUT", "VALIDATION_QUERY2").format(
        prediction_col=prediction_col,
        table_name=udm_table_name,
    )

    instance = GlobalSettings().instance_by_id(submodule_obj.instance_id)
    if not instance:
        Status.NOT_FOUND(
            "Instance not found",
            instance_id=submodule_obj.instance_id,
            alert_status=submodule_obj.alert_status,
        )
        return False

    instance_db = instance.settings.projectdb
    ddf = instance_db.download_table_or_query(query=query2)
    if ddf is None or len(ddf.index) == 0:
        return False
    validation_data = ddf.compute()

    udm_concern_count = None
    if validation_data is not None and len(validation_data.index) > 0:
        udm_concern_count = validation_data.values[0][0]

    return prediction_data["Predicted_Concern_Count"] == udm_concern_count
