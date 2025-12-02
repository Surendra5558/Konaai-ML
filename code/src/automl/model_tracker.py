# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to create the model tracker class"""
import json
import os
import pathlib
import platform
import secrets
import shutil
import urllib.parse
from datetime import date
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional

import joblib
import mlflow
import pandas as pd
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from src.automl.ml_params import MLParameters
from src.automl.model import Model
from src.automl.model import ModelMetrics
from src.tools.dask_tools import is_numeric
from src.utils.conf import Setup
from src.utils.file_mgmt import FileHandler
from src.utils.status import Status
from src.utils.submodule import Submodule


class ModelTracker(BaseModel):
    """
    ModelTracker is a utility class for managing machine learning model tracking, registration, and experiment management using MLflow.

    This class provides methods to:
    - Initialize and configure MLflow tracking for a specific submodule and instance.
    - Generate and manage MLflow experiments, including creation, retrieval, and deletion.
    - Register models and associated artifacts (such as metrics, feature importance, confusion matrix, encoders, training data, etc.) with MLflow.
    - Retrieve models and their metadata by name or run ID, including loading associated artifacts.
    - Fetch and display experiment and model metrics in a structured format.
    - Handle experiment and model lifecycle operations, including permanent deletion of experiments and their associated models and files.
    """

    submodule_obj: Submodule = None
    params: Dict = None
    artifacts: Dict = None

    def __init__(self, submodule_obj: Submodule, **data) -> None:
        """
        Initializes the class with a given Submodule object and additional data.
        Args:
            submodule_obj (Submodule): The submodule instance to be associated with this class.
            **data: Arbitrary keyword arguments for additional initialization parameters.
        """

        super().__init__(**data)
        self.submodule_obj = submodule_obj

        # set tracking uri
        mlflow.set_tracking_uri(self.tracking_uri)

    @property
    def base_dir(self) -> str:
        """
        Returns the base directory used for storing machine learning run artifacts.

        Returns:
            str: The name of the base directory, typically "mlruns".
        """
        return "mlruns"

    @property
    def tags(self):
        """
        Returns a dictionary containing the module and submodule tags for the model tracker.

        Returns:
            dict: A dictionary with the following keys:
                - "module": The name of the module associated with the model tracker.
                - "submodule": The name of the submodule associated with the model tracker.
        """
        return {
            "module": self.submodule_obj.module,
            "submodule": self.submodule_obj.submodule,
        }

    @property
    def tracking_uri(self) -> str:
        """
        Returns the tracking URI as a file URL for the model tracking directory.
        This method constructs a directory path for tracking model artifacts by joining the database path,
        the instance ID of the submodule object, and a base directory. If the directory does not exist,
        it is created. The method then returns the file URI corresponding to this directory.

        Returns:
            str: The file URI pointing to the tracking directory.
        """
        tracking_dir = os.path.join(
            Setup().db_path, self.submodule_obj.instance_id, self.base_dir
        )

        # create the directory if it does not exist
        pathlib.Path(tracking_dir).mkdir(parents=True, exist_ok=True)

        return urllib.parse.urljoin("file:", urllib.request.pathname2url(tracking_dir))

    def generate_experiment(self) -> Experiment:
        """
        Generates and registers a new MLflow experiment for the current submodule.
        This function creates a unique experiment name using the submodule's module name,
        submodule name, and the current timestamp. It then creates the experiment in MLflow,
        sets it as the active experiment, and returns the corresponding Experiment object.

        Returns:
            Experiment: The MLflow Experiment object corresponding to the newly created experiment.
        """
        Status.INFO(
            "Creating experiment",
            module=self.submodule_obj.module,
            submodule=self.submodule_obj.submodule,
            instance_id=self.submodule_obj.instance_id,
        )

        # create today's date in human readable format
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # mlflow.set_tracking_uri(tracking_uri)
        experiment_name = (
            f"{self.submodule_obj.module}_{self.submodule_obj.submodule}_{today}"
        )

        mlflow.create_experiment(experiment_name, tags=self.tags)
        mlflow.set_experiment(experiment_name)

        return mlflow.get_experiment_by_name(experiment_name)

    def register(self, model: Model, experiment_id: str, params: Dict = None):
        """
        Registers a model run with MLflow under a specified experiment.
        This function initiates a new MLflow run for the provided model, logs the given parameters,
        and handles the registration process. It also logs informational and success status messages.

        Args:
            model (Model): The model object to be registered.
            experiment_id (str): The MLflow experiment ID under which the run will be created.
            params (Dict, optional): A dictionary of parameters to log with the run. Defaults to None.

        Returns:
            None
        """
        Status.INFO("Creating run", instance_id=self.submodule_obj.instance_id)

        # set params and artifacts
        self.params = params if params is not None else {}

        with mlflow.start_run(
            experiment_id=experiment_id,
            log_system_metrics=False,
        ) as flow_run:
            self.__initiate_registration(flow_run, model)
        Status.SUCCESS(
            f"Model registration completed for {model.name}",
            instance_id=self.submodule_obj.instance_id,
        )

    def __initiate_registration(self, flow_run, model: Model):
        # set tags
        mlflow.set_tags(self.tags)
        self.__register_params()

        # log model metrics
        for metric, value in model.metrics.model_dump().items():
            mlflow.log_metric(key=metric, value=value)

        # log missing summary
        if model.missing_summary is not None:
            _, file_path = FileHandler().get_new_file_name("xlsx", "missing_summary")
            model.missing_summary.to_excel(file_path, index=False, engine="openpyxl")
            mlflow.log_artifact(file_path, "missing_summary")

        # log feature importance
        if model.feature_importance is not None:
            _, file_path = FileHandler().get_new_file_name("json", "feature_importance")
            # save as JSON
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(model._feature_importance, file)
            mlflow.log_artifact(file_path, "feature_importance")

        # log confusion matrix
        if model.confusion_matrix is not None:
            _, file_path = FileHandler().get_new_file_name("pkl", "confusion_matrix")
            with open(file_path, "wb") as file:
                joblib.dump(model.confusion_matrix, file)
            mlflow.log_artifact(file_path, "confusion_matrix")

        # log encoders
        if model.encoders is not None:
            _, file_path = FileHandler().get_new_file_name("pkl", "encoders")
            with open(file_path, "wb") as file:
                joblib.dump(model.encoders, file)
            mlflow.log_artifact(file_path, "encoders")

        # log training data
        if model.training_data is not None and len(model.training_data) > 0:
            _, file_path = FileHandler().get_new_file_name("parquet", "training_data")
            model.training_data.to_parquet(file_path, compression="gzip")
            mlflow.log_artifact(file_path, "training_data")

        # log base shap value
        if model.base_shap_value is not None:
            mlflow.log_param("base_shap_value", model.base_shap_value)

        # log algo type
        if model.algo_type is not None:
            mlflow.log_param("algo_type", model.algo_type)

        # register model
        self.__register_model(flow_run.info.run_id, model)

    def __register_model(self, run_id: str, model: Model):
        """This function registers a model with mlflow"""
        tags = {
            "model_run_id": run_id,
            **self.tags,
        }

        unique_id = (
            date.today().strftime("%Y%m%d") + "_" + str(secrets.randbelow(9000) + 1000)
        )
        model.name = f"{model.name}_{unique_id}"

        # register model
        mlflow.sklearn.log_model(
            model.model,
            registered_model_name=model.name,
            artifact_path="model",
        )

        # set tags
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_tag(model.name, "category", model.category)
        for tag, value in tags.items():
            client.set_registered_model_tag(model.name, tag, value)

    def __register_params(self):
        """This function registers the parameters with mlflow"""
        Status.INFO(
            "Registering parameters", instance_id=self.submodule_obj.instance_id
        )
        for key, value in self.params.items():
            mlflow.log_param(key, value)

        # also log ml_params
        for key, value in self.submodule_obj.ml_params.__dict__.items():
            mlflow.log_param(key, value)

    @property
    def ml_experiments(self) -> List[Experiment]:
        """
        Fetches machine learning experiments filtered by module and submodule tags.
        This method queries MLflow for experiments that match the current submodule's
        module and submodule tags, ordering them by creation time in descending order.
        If no experiments are found, it logs a NOT_FOUND status and returns an empty list.
        In case of an error during the fetch operation, it logs a FAILED status with error details
        and returns an empty list.

        Returns:
            List[Experiment]: A list of MLflow Experiment objects matching the filter criteria,
            or an empty list if none are found or an error occurs.
        """
        Status.INFO(
            "Fetching ML experiments",
            self.submodule_obj,
        )
        try:
            # list all experiments
            filter_string = f"tags.module='{self.submodule_obj.module}' and tags.submodule='{self.submodule_obj.submodule}'"
            result = mlflow.search_experiments(
                filter_string=filter_string,
                order_by=["creation_time DESC"],
            )
            if not result:
                Status.NOT_FOUND(
                    "No experiments found",
                    self.submodule_obj,
                )
                return []
            return result

        except BaseException as _e:
            Status.FAILED(
                "Error while fetching ML experiments",
                instance_id=self.submodule_obj.instance_id,
                module=self.submodule_obj.module,
                submodule=self.submodule_obj.submodule,
                error=_e,
            )
            return []

    def get_ml_models(self, experiments: List[Experiment]) -> List[Model]:
        """
        Retrieves machine learning models associated with the provided experiments.

        Args:
            experiments (List[Experiment]): A list of Experiment objects for which to fetch associated ML models.

        Returns:
            List[Model]: A list of Model objects corresponding to the runs in the given experiments.
            Only models that could be successfully retrieved are included in the list.
        """
        Status.INFO(
            "Fetching ML models",
            module=self.submodule_obj.module,
            submodule=self.submodule_obj.submodule,
            instance_id=self.submodule_obj.instance_id,
        )

        all_runs: pd.DataFrame = self.get_ml_runs(experiments)
        all_run_ids = all_runs["run_id"].values.tolist()

        if models := [self.__get_model_by_run_id(run_id) for run_id in all_run_ids]:
            return [model for model in models if model is not None]

        return []

    def all_model_metrics(self, models: List[Model]) -> pd.DataFrame:
        """
        Displays the metrics of a list of models in a formatted pandas DataFrame.
        For each model in the provided list, this function extracts its metrics, formats the metric names for readability,
        and multiplies all numeric metric values by 100 (except for the 'Seconds To Train' column).
        The resulting DataFrame presents each model's metrics as rows with human-friendly column names.

        Args:
            models (List[Model]): A list of Model objects, each containing a 'name' and a 'metrics' attribute.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a model and each column to a formatted metric.
        """
        metrics = {model.name: model.metrics.model_dump() for model in models}
        # create a dataframe
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.columns = [
            col.replace("_", " ").title() for col in metrics_df.columns
        ]
        # Specify the column to exclude from multiplication
        exclude_col = "Seconds To Train"

        # Select numeric columns (excluding the one to exclude)
        numeric_cols = [
            col
            for col in metrics_df.columns
            if col not in exclude_col and is_numeric(metrics_df[col])
        ]

        # Multiply numeric columns by 100
        metrics_df[numeric_cols] = metrics_df[numeric_cols] * 100
        return metrics_df

    def get_ml_runs(self, experiments: List[Experiment]) -> pd.DataFrame:
        """
        Fetches machine learning runs for the provided list of experiments.
        This function sets the MLflow tracking URI and retrieves all runs associated with the given experiments,
        ordered by their start time in descending order.
        Args:
            experiments (List[Experiment]): A list of Experiment objects for which to fetch ML runs.
        Returns:
            pd.DataFrame: A DataFrame containing the ML runs for the specified experiments.
        Logs:
            Logs an informational message indicating that ML runs are being fetched.
        """
        Status.INFO(
            "Fetching ML runs",
            module=self.submodule_obj.module,
            submodule=self.submodule_obj.submodule,
            instance_id=self.submodule_obj.instance_id,
        )

        # list all experiments
        # set tracking uri
        mlflow.set_tracking_uri(self.tracking_uri)
        # list all run in each experiment
        return mlflow.search_runs(
            experiment_ids=[exp.experiment_id for exp in experiments],
            order_by=["start_time DESC"],
        )

    def get_experiment_by_model_name(self, model_name: str) -> Experiment:
        """
        Retrieve the MLflow experiment associated with a registered model by its name.

        Args:
        ----
            model_name (str): The name of the registered model to search for.

        Returns:
            Experiment or None: The MLflow Experiment object associated with the latest version
            of the registered model if found, otherwise None.

        Notes:
            - Assumes that the registered model has a tag 'model_run_id' pointing to the run ID.
            - Only the latest version of the model is considered.
        """
        # search for the model using the name
        filter_string = f"name='{model_name}'"
        results: List[RegisteredModel] = mlflow.search_registered_models(
            filter_string=filter_string
        )

        # if no results are found
        if not results:
            return None

        # as a practice we only use the latest version of the model
        result_model: RegisteredModel = results[0]

        # get model by run ID
        run_id = result_model.tags["model_run_id"]

        if not run_id:
            return None

        run = mlflow.get_run(run_id)

        experiment_id = run.info.experiment_id

        experiment = mlflow.get_experiment(experiment_id)

        return experiment if isinstance(experiment, Experiment) else None

    def get_model_by_name(self, model_name: str) -> Optional[Model]:
        """
        Retrieve a model by its registered name.
        Args:
        ----
            model_name (str): The name of the registered model to retrieve.
        Returns:
            Optional[Model]: The latest version of the model if found, otherwise None.

        Notes:
            - Searches for the registered model using the provided name.
            - Returns the latest version of the model by retrieving it using its run ID.
        """
        # search for the model using the name
        filter_string = f"name='{model_name}'"
        results: List[RegisteredModel] = mlflow.search_registered_models(
            filter_string=filter_string
        )

        # if no results are found
        if not results:
            return None

        # as a practice we only use the latest version of the model
        result_model = results[0]
        # get model by run ID
        return self.__get_model_by_run_id(result_model.tags["model_run_id"])

    def __get_model_by_run_id(self, run_id: str) -> Optional[List[Model]]:
        """This function is used to get the model by run ID"""
        # create run ID filter string
        client = mlflow.tracking.MlflowClient()

        # get the run
        run = mlflow.get_run(run_id)

        artifact_uri = (
            run.info.artifact_uri
        )  # This is the hard registered path in the metadata files

        # incase the new version is installed, the base path changes, hence we need to get the latest path
        tracking_base_path = self.tracking_uri.split(self.base_dir)[0]
        artifact_rel_path = artifact_uri.split(self.base_dir)[1]
        artifact_uri = str(self.base_dir).join([tracking_base_path, artifact_rel_path])

        # get the model
        filter_string = f"tags.model_run_id='{run_id}'"
        results: List[RegisteredModel] = client.search_registered_models(
            filter_string=filter_string
        )

        # if no results are found
        if not results:
            return None

        # as a practice we only use the latest version of the model
        result_model = results[0]
        model_name = result_model.name
        # model_version = result_model.latest_versions[0].version

        # TODO: Need to change how model is loaded because model created by older scikit learn versions
        # can not be loaded when newer scikit learn is installed.
        model_uri = f"{artifact_uri}/model/model.pkl"

        local_path = mlflow.artifacts.download_artifacts(model_uri)
        model = joblib.load(local_path)

        params = run.data.params
        # convert back to required types
        # params = {key: ast.literal_eval(value) for key, value in params.items()}
        metrics = run.data.metrics
        result_model = Model(
            name=model_name,
            created_on=run.info.start_time,
            category=result_model.tags["category"],
            model=model,
            metrics=ModelMetrics(**metrics),
            params=MLParameters(**params),
            base_shap_value=run.data.params.get("base_shap_value"),
            algo_type=run.data.params.get("algo_type"),
        )

        # set run ID
        result_model.run_id = run_id

        def first_file_in_dir(dir_path: str):
            """List first file in the directory"""
            files = os.listdir(dir_path)
            for file in files:
                if os.path.isfile(os.path.join(dir_path, file)):
                    # return full path of the file
                    return os.path.join(dir_path, file)
            return None

        def uri_to_path(uri: str) -> str:
            parsed_uri = urllib.parse.urlparse(uri)

            if parsed_uri.scheme != "file":
                raise ValueError(
                    f"URI scheme is not file when converting to path. Found scheme {parsed_uri.scheme}"
                )

            path = None
            if platform.system() == "Windows":
                path = urllib.parse.unquote(parsed_uri.path.lstrip("/"))
                return pathlib.Path(pathlib.PureWindowsPath(path))

            return pathlib.Path(pathlib.PurePath(parsed_uri.path))

        artifacts_path = uri_to_path(artifact_uri)
        # list all directories in the artifact path. all our registered artifacts are inside directory, hence filter them
        artifact_dirs = [
            dir
            for dir in os.listdir(artifacts_path)
            if os.path.isdir(os.path.join(artifacts_path, dir))
        ]

        for artifact in artifact_dirs:
            try:
                artifact_dir = os.path.join(artifacts_path, artifact)

                file_path = first_file_in_dir(artifact_dir)
                # continue loop if file is zero size
                if file_path and os.path.getsize(file_path) == 0:
                    continue

                # load missing summary
                if artifact == "missing_summary":
                    result_model.missing_summary = pd.read_excel(file_path)
                elif artifact == "feature_importance":
                    with open(file_path, encoding="utf-8") as file:
                        result_model.feature_importance = json.load(file)
                elif artifact == "confusion_matrix":
                    with open(file_path, "rb") as file:
                        result_model.confusion_matrix = joblib.load(file)
                elif artifact == "encoders":
                    with open(file_path, "rb") as file:
                        result_model.encoders = joblib.load(file)
                elif artifact == "training_data":
                    # get immediate children of artifact directory that end with word parquet
                    parquet_items = [
                        item
                        for item in os.listdir(artifact_dir)
                        if item.endswith(".parquet")
                    ]
                    if not parquet_items:
                        continue

                    # get full path of parquet items
                    parquet_items = [
                        pathlib.Path(artifact_dir).joinpath(item)
                        for item in parquet_items
                    ]

                    result_model.training_data = pd.DataFrame()
                    for parquet_item in parquet_items:
                        result_model.training_data = pd.concat(
                            [result_model.training_data, pd.read_parquet(parquet_item)]
                        )

            except BaseException as _e:
                Status.FAILED(
                    "Error while fetching model by run ID",
                    run_id=run_id,
                    error=_e,
                    traceback=False,
                )
                continue

        return result_model

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Permanently deletes an MLflow experiment and all associated models and files.
        This method performs the following steps:
        1. Retrieves experiment details using the provided experiment ID.
        2. Finds all runs within the experiment.
        3. Deletes all registered models associated with each run, including their custom model folders.
        4. Deletes the experiment from MLflow (moves it to the `.trash` directory).
        5. Removes all physical files related to the experiment from both the original and trash locations.

        Args:
            experiment_id (str): The unique identifier of the experiment to delete.

        Returns:
            bool: True if the experiment and all associated data were deleted successfully, False otherwise.

        Raises:
            Handles all exceptions internally and logs the status. Returns False on failure.
        """
        try:
            client = MlflowClient()

            # 1. Get experiment details
            experiment: Experiment = mlflow.get_experiment(experiment_id)
            exp_name = experiment.name

            # 2. Get all runs in the experiment
            runs_df: pd.DataFrame = mlflow.search_runs(experiment_ids=[experiment_id])

            # 3. Delete registered models
            for _, run in runs_df.iterrows():
                run_id = run["run_id"]

                registered_models: List[RegisteredModel] = (
                    client.search_registered_models(
                        filter_string=f"tags.model_run_id = '{run_id}'"
                    )
                )

                for model in registered_models:
                    client.delete_registered_model(model.name)

                    # Delete from custom models folder
                    model_path = os.path.join("models", model.name)
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)

            # 4. Delete experiment (moves to .trash)
            mlflow.delete_experiment(experiment_id)

            # 5. Delete physical files from BOTH locations
            tracking_dir = os.path.join(
                Setup().db_path, self.submodule_obj.instance_id, self.base_dir
            )

            # Original experiment path
            exp_path = os.path.join(tracking_dir, experiment_id)
            if os.path.exists(exp_path):
                shutil.rmtree(exp_path)

            # Trash path (where MLflow moves deleted experiments)
            trash_path = os.path.join(tracking_dir, ".trash", experiment_id)
            if os.path.exists(trash_path):
                shutil.rmtree(trash_path)

            Status.SUCCESS(
                "Experiment deleted permanently.",
                experiment_id=experiment_id,
                experiment_name=exp_name,
            )
            return True
        except BaseException as e:
            Status.FAILED(
                "Error deleting experiment", experiment_id=experiment_id, error=e
            )
            return False
