# # Copyright (C) KonaAI - All Rights Reserved
# # # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to monitor the model performance"""
from typing import Any
from typing import Dict
from typing import Optional

import dask.dataframe as dd
from sklearn.metrics import f1_score
from src.automl.fetch_data import ModelData
from src.automl.model import Model
from src.automl.model_tracker import ModelTracker
from src.automl.prediction_pipeline import PredictionPipeline
from src.automl.utils import config
from src.tools.dask_tools import validate_column
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule


def calculate_f1_score(y_true, y_pred) -> float:
    """Calculates the F1 score and scales it to a percentage."""
    # This directly mirrors the original function's logic
    return f1_score(y_true, y_pred) * 100


class ModelPerformanceMonitor:
    """
    Monitors the active model's performance by executing the
    prediction pipeline on the current data and comparing the F1 score
    against the model's historical F1 score.
    """

    def __init__(self, submodule: Submodule):
        """Initializes the monitor with the target submodule."""
        self.sub = submodule
        self.index_col = config.get("DATA", "INDEX")
        self.module_info = f"submodule {self.sub.module} - {self.sub.submodule}"

    def monitor_active_model(self) -> Dict[str, Optional[float]]:
        """
        Runs the monitoring process to calculate the new F1 score and its
        percentage change from the old F1 score.

        Returns:
            A dictionary containing:
            - old_f1_score: historical F1 score
            - new_f1_score: current F1 score
            - percentage_change: percentage change between old and new F1
            - concern_count: number of concern predictions (1)
            - no_concern_count: number of no concern predictions (0)
            - total_predictions: total number of predictions made
        """
        # Initialize result dictionary with None values
        result = {
            "old_f1_score": None,
            "new_f1_score": None,
            "percentage_change": None,
            "concern_count": None,
            "no_concern_count": None,
        }

        active_model_name = self.sub.active_model
        if not active_model_name:
            Status.INFO(f"No active model found for {self.module_info}")
            return result

        instance = GlobalSettings.instance_by_id(self.sub.instance_id)
        if not instance:
            Status.INFO(f"No instance found for id {self.sub.instance_id}")
            return result

        table_name = self.sub.get_data_table_name()
        if not table_name:
            Status.INFO(f"No data table found for {self.module_info}")
            return result

        current_model: Model = ModelTracker(submodule_obj=self.sub).get_model_by_name(
            active_model_name
        )
        if not current_model:
            Status.INFO(f"No active model found for {self.module_info}")
            return result

        db = instance.settings.projectdb
        X: Optional[Any] = db.download_table_or_query(
            table_name=table_name
        )  # X can be dd.DataFrame or pd.DataFrame

        if X is None or len(X) == 0:
            Status.INFO(f"No data found in table {table_name} for {self.module_info}")
            return result

        # Fetch true labels
        _, y_true, error = ModelData(submodule_obj=self.sub).get_training_data()

        if error:
            Status.FAILED(
                f"Error fetching training data for {self.module_info}: {error}"
            )
            return result

        # Get indexes of true labels to filter the main data table X
        indexes_values = y_true.index.compute().tolist()

        # update index column
        index_col = validate_column(self.index_col, X)
        X = X.set_index(index_col)

        # Filter X to include only records present in y_true
        if isinstance(X, dd.DataFrame):
            X_computed = X.compute()
            X_new = X_computed[X_computed.index.isin(indexes_values)]
            X_new = dd.from_pandas(X_new, npartitions=X.npartitions)
        else:
            # If X is already pandas
            X_new = X[X.index.isin(indexes_values)]

        if len(X_new) == 0:
            Status.FAILED(
                f"No new data found for prediction in table {table_name} for {self.module_info}"
            )
            return result

        # Run Prediction Pipeline
        prediction_pipeline = PredictionPipeline(self.sub)
        output_path = prediction_pipeline._predict(X_new, current_model)

        if output_path is None:
            Status.FAILED(f"Prediction failed for {self.module_info}")
            return result

        # Get data from output path and calculate f1 score
        # Use .compute() if dd.read_parquet returns dask immediately, to handle indexing.
        X_new_final = dd.read_parquet(output_path).compute()
        y_predicted = X_new_final.get("ML_Prediction")

        if y_predicted is None:
            Status.FAILED(
                f"Prediction failed: 'ML_Prediction' column missing for {self.module_info}"
            )
            return result

        # Calculate concern and no concern counts
        y_predicted_computed = (
            y_predicted.compute()
            if isinstance(y_predicted, (dd.DataFrame, dd.Series))
            else y_predicted
        )

        concern_count = int((y_predicted_computed == 1).sum())
        no_concern_count = int((y_predicted_computed == 0).sum())
        total_predictions = len(y_predicted_computed)

        Status.INFO(
            f"Prediction completed for {self.module_info}. Total Predictions: {total_predictions}, Concern: {concern_count}, No Concern: {no_concern_count}"
        )

        # Filter and align y_true with the predicted results (X_new_final)
        # Use the computed index from the predicted data for precise alignment.
        y_true_computed = (
            y_true.compute()
            if isinstance(y_true, (dd.DataFrame, dd.Series))
            else y_true
        )
        # y_true_filtered must match the index of y_predicted, which is X_new_final.index
        y_true_filtered = y_true_computed.loc[X_new_final.index]

        new_f1 = calculate_f1_score(y_true_filtered, y_predicted)

        # Compare scores
        old_f1 = (current_model.metrics.f1) * 100

        # Calculate percentage change
        percentage_change = ((new_f1 - old_f1) / old_f1) * 100 if old_f1 != 0 else 0

        # Update result dictionary
        result.update(
            {
                "old_f1_score": round(old_f1, 2),  # Round old_f1,
                "new_f1_score": round(new_f1, 2),  # Round new_f1,
                "percentage_change": round(percentage_change, 2),
                "concern_count": concern_count,
                "no_concern_count": no_concern_count,
            }
        )

        Status.INFO(
            f"Old F1 Score {old_f1:.4f}% and New F1 Score {new_f1:.4f}% with Percentage Change: {percentage_change:.4f}% concern_count: {concern_count} no_concern_count: {no_concern_count} , module={self.sub.module},submodule={self.sub.submodule}"
        )

        return result
