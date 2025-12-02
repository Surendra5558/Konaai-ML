# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the pipeline for the automl submodule predictions"""
from datetime import datetime
from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas import Timestamp
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from src.automl.feature_builder import FeatureBuilder
from src.automl.feature_encoder import CustomFeatureEncoder
from src.automl.model import Model
from src.automl.model_tracker import ModelTracker
from src.automl.preprocess import PreProcess
from src.automl.utils import config
from src.automl.validator import DataValidator
from src.utils.file_mgmt import file_handler
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule


class LastPredictionDetails(BaseModel):
    """LastPredictionDetails is a class that retrieves and stores the details of the last prediction
    made by a model, including the model name, prediction date, and the count of predicted concerns.
    Attributes:
    ----------
        Model (str): The name of the model used for the last prediction.
        Date (datetime): The date when the last prediction was made.
        Predicted_Concern_Count (int): The count of predicted concerns from the last prediction.

    Raises:
        Status.NOT_FOUND: If an error occurs while fetching the last prediction data, this exception
        is raised with an appropriate error message."""

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: (
                v.isoformat() if isinstance(v, datetime) and v else None
            ),
        }
    )

    Model: str = None
    Date: datetime = None
    Predicted_Concern_Count: int = 0
    sub_: Submodule = Field(default=None, exclude=True)

    def __init__(self, submodule: Submodule):
        super().__init__()
        self.sub_ = submodule
        self.Model = None
        self.Date = None
        self.Predicted_Concern_Count = None
        self._get_last_prediction_data()

    def _get_last_prediction_data(self):
        try:
            # get table name
            table_name = PredictionPipeline(self.sub_).prediction_table

            date_col = config.get("OUTPUT", "Date_Column")
            model_col = config.get("OUTPUT", "Model_Column")
            prediction_col = config.get("OUTPUT", "Prediction_Column")

            # Formating table name
            table_name = PredictionPipeline(self.sub_).prediction_table

            # get distinct model name and date
            query1 = config.get("OUTPUT", "VALIDATION_QUERY1").format(
                model_col=model_col, date_col=date_col, table_name=table_name
            )
            # get count of predictions
            query2 = config.get("OUTPUT", "VALIDATION_QUERY2").format(
                prediction_col=prediction_col,
                table_name=table_name,
            )

            # upload predictions
            instance = GlobalSettings.instance_by_id(self.sub_.instance_id)
            if not instance:
                Status.NOT_FOUND(
                    "Instance not found", self.sub_, alert_status=self.sub_.alert_status
                )
                return
            instance_db = instance.settings.projectdb
            q1_data = instance_db.download_table_or_query(query=query1)
            if q1_data is None or len(q1_data) == 0:
                Status.NOT_FOUND("No data found for the last prediction query.")
                return

            q1_result = q1_data.compute().dropna()

            if q1_result is not None and len(q1_result.index) > 0:
                self.Model = q1_result[model_col].values[0]
                date_ = q1_result[date_col].values[0]

                # convert numpy datetime to datetime
                date_: Timestamp = pd.to_datetime(date_)
                self.Date = date_.to_pydatetime()

            q2_data = instance_db.download_table_or_query(query=query2)
            if q2_data is None or len(q2_data) == 0:
                Status.NOT_FOUND("No data found for the last prediction query.")
                return

            q2_result = q2_data.compute()

            if q2_result is not None and len(q2_result.index) > 0:
                self.Predicted_Concern_Count = int(q2_result.values[0][0])
        except BaseException as _e:
            Status.NOT_FOUND(
                "Error while fetching last prediction data.",
                self.sub_,
                error=str(_e),
            )
            self.Model = None
            self.Date = None
            self.Predicted_Concern_Count = None


class PredictionPipeline(BaseEstimator, TransformerMixin):
    """This class is used to create the prediction pipeline for the submodule"""

    def __init__(self, submodule_obj: Submodule) -> None:
        """This function initializes the class"""
        self.submodule_obj = submodule_obj
        self.current_model: Model = None
        self.experiment_name: str = None
        super().__init__()

    @property
    def prediction_table(self) -> str:
        """
        The table name is retrieved from the configuration using the "OUTPUT" section and "OUTPUT_TABLE" key.
        It is then formatted with the current module and submodule names, and spaces are replaced with underscores.

        Returns:
            str: The formatted name of the prediction output table.
        """
        # get table name
        return (
            config.get("OUTPUT", "OUTPUT_TABLE")
            .format(
                module=self.submodule_obj.module, submodule=self.submodule_obj.submodule
            )
            .replace(" ", "_")
        )

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """This function is used to fit the pipeline"""
        return self

    def predict(
        self, X: dd.DataFrame
    ) -> Union[bool, str]:  # pylint: disable=unused-argument
        """This function is used to predict from the selected model"""
        mt = ModelTracker(submodule_obj=self.submodule_obj)

        # load model
        self.current_model: Model = mt.get_model_by_name(
            self.submodule_obj.active_model
        )
        if self.current_model is None or not self.current_model.model:
            Status.FAILED(
                "Model not found",
                self.submodule_obj,
            )
            return False

        Status.INFO(
            f"Predicting using model: {self.current_model.name}", self.submodule_obj
        )

        # update experiment name
        if experiment_name := mt.get_experiment_by_model_name(self.current_model.name):
            self.experiment_name = experiment_name.name

        # predict
        output_path = self._predict(X, self.current_model)

        return False if output_path is None else self._upload_predictions(output_path)

    def _predict(self, X: dd.DataFrame, model: Model) -> Union[str, None]:
        """
        Generate predictions using the provided model and input data.
        Args:
            X (dd.DataFrame): Input data in the form of a Dask DataFrame.
            model (Model): The model object used for making predictions.
        Returns:
            str: The file path where the predictions are saved, or None if the model does not have feature names.
        Raises:
            Status.FAILED: If the model does not have feature names.
        Steps:
            1. Check if the model has feature names.
            2. Prepare the input data using model encoders.
            3. Make predictions using the model.
            4. Add today's date to the predictions.
            5. Add the model name to the predictions.
            6. Save the predictions to a parquet file.
        """
        # check if model has feature names
        expected_features = list(model._feature_importance.keys())
        if not expected_features:
            Status.FAILED("Model does not have feature names", self.submodule_obj)
            return None

        # prepare data
        # X = dd.from_pandas(
        #     X.head(1000), npartitions=1
        # )  # Limit to 1000 rows for prediction
        X = self._prepare_data(X, model.encoders)

        # check if X is empty
        if X is None or len(X) == 0:
            Status.FAILED("No data to predict", self.submodule_obj)
            return None

        # get the model
        trained_model: TunedThresholdClassifierCV = model.model

        # Check for missing features
        if missing_features := set(expected_features) - set(X.columns.tolist()):
            Status.INFO(
                f"Missing features in input data: {missing_features}",
                self.submodule_obj,
            )

            # create missing features with nan values
            for feature in missing_features:
                X[feature] = np.nan

        # make predictions
        Status.INFO(f"Making predictions using model: {model.name}", self.submodule_obj)
        prediction_col = config.get("OUTPUT", "Prediction_Column")
        X[prediction_col] = X[expected_features].map_partitions(
            trained_model.predict, meta=(prediction_col, "f8")
        )

        # get prediction probabilities
        prediction_prob_col = config.get("OUTPUT", "Prediction_Probability_Column")
        X[prediction_prob_col] = X[expected_features].map_partitions(
            lambda x: trained_model.predict_proba(x)[:, 1],
            meta=(prediction_prob_col, "f8"),
        )

        # Add today's date to the predictions
        date_col = config.get("OUTPUT", "Date_Column")
        X[date_col] = None
        X[date_col] = X[date_col].fillna(pd.to_datetime(datetime.now()))
        X[date_col] = dd.to_datetime(X[date_col], errors="coerce")

        # Add model name to predictions
        model_col = config.get("OUTPUT", "Model_Column")
        X[model_col] = None
        X[model_col] = X[model_col].fillna(model.name).astype(str)

        # Add experiment name to predictions
        experiment_col = config.get("OUTPUT", "EXPERIMENT_NAME_COLUMN")

        # Get the experiment name, using a placeholder string if it is None
        # The placeholder could be an empty string, 'N/A', or a similar marker.
        experiment_name_to_use = (
            self.experiment_name if self.experiment_name is not None else "N/A"
        )

        # Initialize the column and fill the 'None' values with the resolved experiment name
        X[experiment_col] = None
        X[experiment_col] = X[experiment_col].fillna(experiment_name_to_use).astype(str)

        _, output_path = file_handler.get_new_file_name(file_extension="parquet")
        X.to_parquet(output_path)

        Status.SUCCESS(f"Predictions saved to {output_path}", self.submodule_obj)

        return output_path

    def _upload_predictions(self, file_path: str) -> bool:
        # upload predictions
        instance = GlobalSettings.instance_by_id(self.submodule_obj.instance_id)
        if not instance:
            Status.NOT_FOUND(
                "Instance not found",
                self.submodule_obj,
                alert_status=self.submodule_obj.alert_status,
            )
            return False
        instance_db = instance.settings.projectdb

        return instance_db.upload_table(
            data_file_path=file_path, table_name=self.prediction_table
        )

    def _prepare_data(
        self, X: dd.DataFrame, encoders: CustomFeatureEncoder
    ):  # pylint: disable=unused-argument
        """This method runs the pipeline for the submodule"""
        Status.INFO("Preparing data for prediction", self.submodule_obj)

        fb = FeatureBuilder(
            submodule=self.submodule_obj,
            preprocessor=PreProcess(self.submodule_obj),
            encoder=encoders,
        )

        data_pipeline = Pipeline(
            steps=[
                ("Custom Features", encoders.custom_features_transformer),
                ("Feature Building", fb),
            ],
            verbose=False,
        )

        # prepare input data
        X = data_pipeline.transform(X)
        return DataValidator(submodule=self.submodule_obj).validate_index(X)
