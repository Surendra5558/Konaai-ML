# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to explain the prediction for a given transaction ID"""
from datetime import datetime
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import shap
from pandas import Timestamp
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from src.automl.classifiers import AlgoType
from src.automl.feature_encoder import CustomFeatureEncoder
from src.automl.model import FeatureImportance
from src.automl.model import Model
from src.automl.model_tracker import ModelTracker
from src.automl.prediction_pipeline import PredictionPipeline
from src.automl.utils import config
from src.tools.dask_tools import validate_column
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule


class ExplainedFeature(BaseModel):
    """
    Feature is a class that represents a feature in the model.
    Attributes:
    ----------
        name (str): The name of the feature.
        description (str): The description of the feature.
    """

    name: str = Field(None, description="Name of the feature")
    description: str = Field(None, description="Description of the feature")
    contribution_percent: float = Field(None, description="Importance of the feature")
    value: Union[str, float, bool] = Field(None, description="Value of the feature")


class ExplainationOutput(BaseModel):
    """
    ExplainationOutput
    This class represents the output of an explanation process for a machine learning model's prediction.
    It contains detailed information about the prediction, the model, and the features used in the explanation.
    Attributes:
    -----------
        transaction_id (str): Transaction ID for which the prediction is to be explained.
        transaction_id_field_name (str): Name of the field used as the transaction ID in the database.
        model_name (str): Name of the model used for prediction.
        experiment_name (str): Name of the experiment associated with the model.
        is_model_active (bool): Indicates if the predictor model is currently active.
        amount_value (float): Amount associated with the transaction.
        amount_field_name (str): Name of the field used for the amount in the database.
        prediction_date (datetime): Date of the prediction in ISO format.
        predicted_concern (bool): Indicates if the prediction is a concern.
        prediction_probability (float): Probability of the prediction.
        decision_threshold (float): Decision threshold of the model.
        features (List[ExplainedFeature]): List of features and their explanations.
        error (str): Error message if any issue occurs during the process.
    Config:
        model_config (ConfigDict): Custom configuration for JSON encoding, particularly for datetime objects.
    """

    # custom config
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else None,
        },
    )

    transaction_id: str = Field(
        None, description="Transaction ID for which the prediction is to be explained"
    )
    transaction_id_field_name: str = Field(
        None, description="Name of the field used as the transaction ID in the database"
    )
    model_name: str = Field(None, description="Name of the model used for prediction")
    experiment_name: str = Field(
        None, description="Name of the experiment associated with the model"
    )
    is_model_active: bool = Field(
        None, description="Indicates if the predictor model is currently active"
    )
    amount_value: float = Field(
        None, description="Amount associated with the transaction"
    )
    amount_field_name: str = Field(
        None, description="Name of the field used for the amount in the database"
    )
    prediction_date: datetime = Field(
        None, description="Date of the prediction in ISO format"
    )
    predicted_concern: bool = Field(
        None, description="Indicates if the prediction is a concern"
    )
    prediction_probability: float = Field(
        None, description="Probability of the prediction"
    )
    decision_threshold: float = Field(
        None, description="Decision threshold of the model"
    )
    features: List[ExplainedFeature] = Field(
        None, description="List of features and their explanations"
    )
    error: str = Field(
        None, description="Error message if any issue occurs during the process"
    )


class PredictionExplainer:
    """
    PredictionExplainer is a class that provides explanations for predictions made by a machine learning model.
    Attributes:
    ---------
        submodule_obj (Submodule): An object representing the submodule.
        transaction_id (str): The transaction ID for which the prediction explanation is to be generated.
    """

    def __init__(self, submodule_obj: Submodule, transaction_id: str) -> None:
        """This function initializes the class"""
        self.submodule_obj = submodule_obj
        self.transaction_id = transaction_id

    def explain(self) -> Optional[ExplainationOutput]:
        """
        Provides an explanation for the prediction using the explainer data.

        Returns:
            Optional[ExplainationOutput]: The explanation output if successful,
            otherwise None in case of an error.

        Raises:
            Exception: Captures and logs any exception that occurs during the
            explanation process, marking the status as FAILED.
        """
        try:
            return self._get_explainer_data()
        except Exception as e:
            Status.FAILED(
                "Failed to explain the prediction",
                self.submodule_obj,
                transaction_id=self.transaction_id,
                error=str(e),
            )
            return None

    def _get_explainer_data(self) -> ExplainationOutput:

        output = ExplainationOutput()
        output.transaction_id = self.transaction_id
        output.transaction_id_field_name = config.get("DATA", "INDEX")

        data = self._get_transaction_data()
        # read only 1 row of data from parquet file that matches the transaction ID
        if data is None or len(data) == 0:
            return self._handle_explanation_error(
                "No data found for the given transaction ID.", output
            )

        if len(data) > 1:
            return self._handle_explanation_error(
                "Multiple transactions for given transaction id found.", output
            )

        Status.INFO(
            "Features data loaded for AutoML explaination",
            self.submodule_obj,
            transaction_id=self.transaction_id,
        )

        # get the transaction ID field name
        transaction_id_field = config.get("DATA", "INDEX")
        transaction_id_field = validate_column(transaction_id_field, data)

        # now its safe to convert to a single row of data
        data_series: pd.Series = data.iloc[0]

        # get the transaction ID
        output.transaction_id = int(data_series[transaction_id_field])
        output.transaction_id_field_name = transaction_id_field

        # get the model
        model_col = config.get("OUTPUT", "Model_Column")
        model_name = data_series[model_col]
        output.model_name = model_name
        output.is_model_active = self.submodule_obj.active_model == model_name

        mt = ModelTracker(submodule_obj=self.submodule_obj)
        model: Model = mt.get_model_by_name(model_name)
        if model.training_data is None or len(model.training_data) == 0:
            return self._handle_explanation_error(
                "Older models do not support explanations. Train a new model to get explanations.",
                output,
            )

        # get the metadata
        experiment_col = config.get("OUTPUT", "EXPERIMENT_NAME_COLUMN")
        date_col = config.get("OUTPUT", "Date_Column")
        prediction_col = config.get("OUTPUT", "Prediction_Column")
        prediction_prob_col = config.get("OUTPUT", "Prediction_Probability_Column")

        # get the experiment name
        experiment_name = data_series[experiment_col]
        output.experiment_name = experiment_name

        # get the prediction conclusion
        output.predicted_concern = bool(data_series[prediction_col])

        # get the prediction date
        predicted_date: Timestamp = pd.to_datetime(data_series[date_col])
        output.prediction_date = (
            predicted_date.to_pydatetime() if predicted_date else None
        )

        # get the amount value
        amount_column, amount_value = self._get_amount_info()
        output.amount_value = amount_value or None
        output.amount_field_name = amount_column

        Status.INFO(
            "Model loaded for AutoML explaination",
            self.submodule_obj,
            transaction_id=self.transaction_id,
        )
        output.decision_threshold = float(model.metrics.decision_threshold)

        # get prediction probability
        output.prediction_probability = float(data_series[prediction_prob_col])
        if output.prediction_probability == 0 or output.prediction_probability is None:
            return self._handle_explanation_error(
                "No contributing features because the prediction probability is 0",
                output,
            )

        # get the feature explanations
        features_explaination = self._explain_transaction(
            model, data, output.prediction_probability, model.training_data
        )
        if features_explaination is None or len(features_explaination) == 0:
            return self._handle_explanation_error(
                "No contributing features found.", output
            )

        # generate values for features explaination
        features_explaination = self._generate_values(
            features_explaination, data, model, amount_value
        )

        Status.INFO(
            "Explaination ready for AutoML explaination",
            self.submodule_obj,
            transaction_id=self.transaction_id,
        )

        # create features list
        features = [
            ExplainedFeature(
                name=feature["Feature"],
                description=feature["Description"],
                contribution_percent=float(feature["Contribution_Percent"]),
                value=feature["Value"],
            )
            for feature in features_explaination.to_dict(orient="records")
        ]
        output.features = features
        return output

    def _handle_explanation_error(self, error: str, output: ExplainationOutput):
        output.error = error
        Status.WARNING(error, self.submodule_obj, transaction_id=self.transaction_id)
        return output

    def _explain_transaction(
        self,
        model: Model,
        data: pd.DataFrame,
        prediction_prob_value: float,
        training_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate SHAP (SHapley Additive exPlanations) values to explain the predictions of a given model on a transaction.
        Parameters:
        -----------
        model : Model
            The trained model object containing the estimator and feature importance.
        data : pd.DataFrame
            The input data for which predictions are made.
        prediction_prob_value : float
            The predicted probability value for the transaction.
        training_data : pd.DataFrame
            The training data used to fit the model, required for certain types of SHAP explainers.
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the feature contributions with their respective percentages and descriptions.
        Raises:
        -------
        RuntimeError
            If the explainer cannot be created for the given model type.
        Notes:
        ------
        - Supports TREE, LINEAR, and KERNEL algorithm types for SHAP explainers.
        - Filters out features with zero contribution.
        - Logs the creation of the explainer and feature contributions.
        """

        base_estimator = model.model.estimator_

        features: List[FeatureImportance] = model.feature_importance
        feature_names = [feature.feature_name for feature in features]

        explainer = None
        shap_values = None
        if model.algo_type == AlgoType.TREE.value:
            explainer = shap.TreeExplainer(base_estimator)
            shap_values = explainer.shap_values(
                data[feature_names], check_additivity=False
            )
        elif model.algo_type == AlgoType.LINEAR.value:
            explainer = shap.LinearExplainer(
                base_estimator, training_data[feature_names]
            )
            shap_values = explainer.shap_values(data[feature_names])
        elif model.algo_type == AlgoType.KERNEL.value:
            # reduce the number of background data points to 100 or the number of training data points
            bg_data = shap.kmeans(
                training_data[feature_names], min(100, len(training_data))
            )
            explainer = shap.KernelExplainer(base_estimator.predict, bg_data)
            shap_values = explainer.shap_values(data[feature_names])

        if explainer is None:
            raise RuntimeError("Can not create explainer")
        Status.INFO(
            "Explainer created for AutoML explaination",
            self.submodule_obj,
            transaction_id=self.transaction_id,
        )

        # if shap values are of shape (1, len(feature_names), 2) then we need to convert it to (1, len(feature_names))
        if shap_values.shape == (1, len(feature_names), 2):
            shap_values = shap_values[0, :, 1]
            shap_values = shap_values[np.newaxis, :]

        # create feature contributions
        total_shap_value = np.sum(abs(shap_values[0]))
        contributions = {}
        for i, feature_name in enumerate(feature_names):
            cont_value = (
                shap_values[0][i] * prediction_prob_value * 100 / total_shap_value
            )
            contributions[feature_name] = round(cont_value, 2)

        con_df = (
            pd.Series(contributions)
            .reset_index(drop=False, name="Contribution_Percent")
            .sort_values(ascending=True, by="Contribution_Percent")
        )

        con_df = con_df.rename(columns={"index": "Feature"})

        # remove all zero contributions
        con_df = con_df[con_df["Contribution_Percent"] != 0]
        # assign description to features
        con_df["Description"] = con_df["Feature"].apply(
            lambda x: next(
                (
                    feature.description
                    for feature in features
                    if feature.feature_name == x
                ),
                None,
            )
        )
        Status.INFO(
            "Feature contributions created for AutoML explaination",
            self.submodule_obj,
            transaction_id=self.transaction_id,
        )
        return con_df

    def _generate_values(
        self,
        features_explaination: pd.DataFrame,
        data: pd.DataFrame,
        model: Model,
        amount: float,
    ) -> pd.DataFrame:
        """
        Generate a DataFrame with feature explanations and their corresponding values.
        This method captures the actual values of features from the provided data,
        merges them with the feature explanations, and processes the values based on
        whether they are categorical, anomaly features, or numeric.
        Args:
            features_explaination (pd.DataFrame): DataFrame containing feature explanations.
            data (pd.DataFrame): DataFrame containing the actual data.
            model (Model): The model object containing encoders and transformers.
            amount (float): The amount value to be used for updating numeric feature values.
        Returns:
            pd.DataFrame: DataFrame with updated feature explanations and their corresponding values.
        """
        # capture actual values of features
        actual_values: pd.Series = data[features_explaination["Feature"]].iloc[0]
        actual_values.index.name = "Feature"
        actual_values = actual_values.reset_index(drop=False, name="Value")

        # merge features_explaination with actual values
        features_explaination = features_explaination.merge(
            actual_values, on="Feature", how="left"
        )
        # categorical columns
        encoders: CustomFeatureEncoder = model.encoders
        categorical_columns = {
            col.lower() for col in encoders.onehot_encoder.get_feature_names_out()
        }
        # assign boolean value for categorical columns
        features_explaination["Value"] = features_explaination[
            ["Feature", "Value"]
        ].apply(
            lambda x: (
                bool(x["Value"])
                if x["Feature"].lower() in categorical_columns
                else x["Value"]
            ),
            axis=1,
        )

        # change anomaly feature values to %
        anmly_prefix = encoders.custom_features_transformer.anomaly_prefix
        features_explaination["Value"] = features_explaination[
            ["Feature", "Value"]
        ].apply(
            lambda x: (
                f"{x['Value']:.2%}"
                if x["Feature"].startswith(anmly_prefix)
                else x["Value"]
            ),
            axis=1,
        )

        # update actual value for amount column
        numeric_encoder = encoders.numeric_encoder
        features_explaination["Value"] = features_explaination[
            ["Feature", "Value"]
        ].apply(
            lambda x: numeric_encoder.get_polynomial_value(amount, x["Feature"])
            or x["Value"],
            axis=1,
        )

        return features_explaination

    def _get_transaction_data(self) -> pd.DataFrame:
        """
        Retrieves transaction data from the database based on the transaction ID.
        This method constructs a table name using the module and submodule information,
        retrieves the transaction ID field from the configuration, and then downloads
        the corresponding data from the database. The data is read from a parquet file
        and returned as a pandas DataFrame.
        Returns:
            pd.DataFrame: The transaction data as a pandas DataFrame.
        """
        # read explainer data
        table_name = PredictionPipeline(self.submodule_obj).prediction_table
        transaction_id_field = config.get("DATA", "INDEX")

        # download predictions data
        instance = GlobalSettings().instance_by_id(self.submodule_obj.instance_id)
        if not instance:
            Status.FAILED("Instance not found", self.submodule_obj)
            return pd.DataFrame()
        instance_db = instance.settings.projectdb

        # read only 1 row of data from table to validate the transaction ID field
        query = f"SELECT * FROM {table_name} WHERE {transaction_id_field} = {self.transaction_id}"
        dff = instance_db.download_table_or_query(query=query)
        if dff is None or len(dff) == 0:
            Status.FAILED(
                "Error in downloading data or no data found for the given transaction ID",
                self.submodule_obj,
            )
            return pd.DataFrame()

        return dff.compute()

    def _get_amount_info(self) -> Tuple[str, float]:
        # sourcery skip: extract-method
        """
        Retrieves the amount information for a specific transaction.
        This method fetches the amount column and its value from the database
        based on the transaction ID. It uses the configuration settings to
        determine the transaction ID field and establishes a connection to
        the database using the DBHandler class.
        Returns:
            Tuple[str, float]: A tuple containing the name of the amount column
            and the amount value. If the amount column or table name is not found,
            or if the query does not return any data, it returns (None, None).
        """
        transaction_id_field = config.get("DATA", "INDEX")

        instance_db = GlobalSettings.instance_by_id(
            self.submodule_obj.instance_id
        ).settings.projectdb

        if amount_column := self.submodule_obj.get_amount_column():
            if table_name := self.submodule_obj.get_data_table_name():
                query = f"SELECT {amount_column} FROM {table_name} WHERE {transaction_id_field} = {self.transaction_id}"
                dff = instance_db.download_table_or_query(query=query)
                if dff is None or len(dff) == 0:
                    return None, None

                amount_data = dff.compute()
                amount_value = float(amount_data[amount_column].iloc[0])

                return amount_column, amount_value

        return None, None
