# # Copyright (C) KonaAI - All Rights Reserved
from typing import List

import dask.dataframe as dd
import pandas as pd
import pretty_errors
import pytest
from data_generators.PO_data_generation import generate_po_dataframe
from src.automl.model import Model
from src.automl.model_tracker import ModelTracker
from src.automl.prediction_pipeline import PredictionPipeline
from src.automl.training_pipeline import TrainingPipeline
from src.automl.utils import config
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule


class TestAutoMLPipeline:
    submodule = None

    X, y = generate_po_dataframe(records=50)
    index_col = config.get("DATA", "INDEX")
    X: dd.DataFrame = X.set_index(index_col)
    models: List[Model] = []

    def test_training_pipeline(self):
        """Test the training pipeline."""
        # Set up the submodule
        self.submodule = Submodule(
            module="P2P",
            submodule="Purchase Order",
            instance_id=GlobalSettings().active_instance_id,
        )
        current_iterations = self.submodule.ml_params.n_splits
        self.submodule.ml_params.n_splits = 1
        self.submodule.save_configuration()

        # Generate data
        X = self.X.copy()

        assert len(X) == 50, "Input data size is not equal to expected size"

        pipeline = TrainingPipeline(submodule_obj=self.submodule)
        self.trained_models = pipeline.fit(X, self.y)
        self.models.extend(self.trained_models)

        assert len(self.trained_models) > 0, "no models were trained"

        for model in self.trained_models:  # sourcery skip: no-loop-in-tests
            self._test_metrics_for_model(model)

        # revert number of iterations
        self.submodule.ml_params.n_splits = current_iterations
        self.submodule.save_configuration()

    def _test_metrics_for_model(self, model: Model):
        assert model.metrics.accuracy > 0, "Model accuracy is not greater than 0"
        assert (
            model.metrics.balanced_accuracy > 0
        ), "Model balanced accuracy is not greater than 0"
        assert model.metrics.roc_auc > 0, "Model roc_auc is not greater than 0"
        assert model.metrics.f1 > 0, "Model f1 is not greater than 0"
        assert model.metrics.precision > 0, "Model precision is not greater than 0"
        assert model.metrics.recall > 0, "Model recall is not greater than 0"
        assert (
            model.metrics.seconds_to_train > 0
        ), "Model seconds_to_train is not greater than 0"
        assert (
            model.metrics.decision_threshold >= 0
        ), "Model decision_threshold should be greater than 0"

    def validate_predictions(self, model: Model) -> bool:
        """Validate the predictions."""
        assert len(self.X) == 50, "Input data size is not equal to expected size"

        Status.INFO(f"Validating predictions for model: {model.name}")

        if not self.submodule:
            self.submodule = Submodule(
                module="P2P",
                submodule="Purchase Order",
                instance_id=GlobalSettings().active_instance_id,
            )

        pipeline = PredictionPipeline(self.submodule)

        mt = ModelTracker(submodule_obj=self.submodule)
        experiment = mt.get_experiment_by_model_name(model.name)
        assert experiment is not None, f"Experiment not found for model: {model.name}"
        pipeline.experiment_name = experiment.name

        result_path = pipeline._predict(self.X, model)

        assert result_path is not None, "Prediction failed"

        _df = pd.read_parquet(result_path)
        assert len(_df) == 50, "Number of rows in the result is not equal to the input"
        prediction_col = config.get("OUTPUT", "Prediction_Column")
        assert (
            prediction_col in _df.columns
        ), "Prediction column not found in the result"

        assert (
            _df.index.name == self.X.index.name
        ), "Index column not found in the result"

        return True

    def test_predictions(self):
        # sourcery skip: no-loop-in-tests
        for model in self.models:
            try:
                # sourcery skip: no-conditionals-in-tests
                if self.validate_predictions(model):
                    Status.SUCCESS(
                        f"Prediction successful for model: {model.name}", self.submodule
                    )
                else:
                    pytest.fail(f"Prediction failed for model: {model.name}")
            except Exception as e:
                pretty_errors.excepthook(e.__class__, e, e.__traceback__)
                pytest.fail(f"Prediction failed for model: {model.name}")
