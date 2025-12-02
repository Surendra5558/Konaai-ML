# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the pipeline for the automl submodule"""
import logging
from typing import Dict
from typing import List
from typing import Tuple

import dask.dataframe as dd
from mlflow.entities import Experiment
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from src.automl.classifiers import BinaryRiskClassifier
from src.automl.custom_transformer import CustomFeaturesTransformer
from src.automl.datetime_encoder import DateTimeEncoder
from src.automl.feature_builder import FeatureBuilder
from src.automl.feature_encoder import CustomFeatureEncoder
from src.automl.feature_selection import FeatureSelection
from src.automl.model import Model
from src.automl.model_tracker import ModelTracker
from src.automl.numeric_encoder import NumericEncoder
from src.automl.preprocess import PreProcess
from src.automl.validator import DataValidator
from src.tools.ml_feature_mapping import MLFeatureMappings
from src.utils.conf import Setup
from src.utils.status import Status
from src.utils.submodule import Submodule

# disable mlflow logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

feature_mapper = MLFeatureMappings()


class TrainingPipeline(BaseEstimator, TransformerMixin):
    """
    TrainingPipeline is a machine learning pipeline class designed to orchestrate the end-to-end training process for a given submodule. It integrates data preparation, feature engineering, model training, feature description assignment, and experiment tracking into a unified workflow.
        submodule_obj (Submodule): The submodule instance containing configuration, data access, and metadata required for the pipeline.
    Attributes:
    ------------
        submodule_obj (Submodule): The submodule instance associated with this pipeline.
        params (dict): Stores parameters and statistics related to the training process.
        missing_summary (dict): Summary of missing values in the training data.
        encoders (CustomFeatureEncoder or None): The feature encoder used during preprocessing.
        models (List[Model]): List of trained model instances.
        experiment (Experiment or None): The experiment object tracking this training run.
        patterns_data (pandas.DataFrame or None): DataFrame containing test pattern IDs and descriptions.
    """

    submodule_obj: Submodule

    def __init__(self, submodule_obj: Submodule) -> None:
        """This function initializes the class"""
        self.submodule_obj = submodule_obj
        self.params = {}
        self.missing_summary = {}
        self.encoders = None
        self.models: List[Model] = []
        self.experiment: Experiment = None
        self.patterns_data = self._get_patterns_data(submodule_obj)
        self.training_complete = False

        super().__init__()

    def _get_patterns_data(self, sub: Submodule):
        """
        Retrieve test patterns data from the given submodule.

        Args:
        ----
            sub (Submodule): The submodule from which to retrieve test patterns.

        Returns:
            pandas.DataFrame: A DataFrame containing the test patterns with columns
            for pattern ID and pattern description.
        """
        # get the test patterns
        patterns_data = sub.get_test_patterns()
        if patterns_data is None:
            Status.FAILED(
                "No test patterns found",
                self.submodule_obj,
            )
            return None

        pattern_id_col = (
            Setup().global_constants.get("TEST_PATTERNS", {}).get("PATTERN_ID_COLUMN")
        )
        pattern_description_col = (
            Setup()
            .global_constants.get("TEST_PATTERNS", {})
            .get("PATTERN_DESCRIPTION_COLUMN")
        )
        patterns_data = patterns_data[[pattern_id_col, pattern_description_col]]
        return patterns_data

    def fit(self, X: dd.DataFrame, y: dd.Series) -> List[Model]:
        """
        Fits the training pipeline by training models on the provided data and tracking the experiment.
        Args:
        ----
            X (dd.DataFrame): The input features as a Dask DataFrame.
            y (dd.Series): The target variable as a Dask Series.

        Returns:
            List[Model]: A list of trained model instances.
        """
        # train models
        self.train(X, y)

        # track models
        self.experiment = self.track()

        self.training_complete = True

        return self.models

    def _assign_desc(
        self, feature_name: str, encoder: CustomFeatureEncoder
    ) -> Tuple[str, str]:
        """
        Assigns a description to a given feature name based on its encoding and transformation.
        Args:
            feature_name (str): The name of the feature to describe.
            encoder (CustomFeatureEncoder): An instance of CustomFeatureEncoder containing various encoders.
        Returns:
            str: A description of the feature.
            str: The type of the feature (e.g., "Test Pattern", "Numeric", "Anomaly", "Categorical", "Other").
        Description:
            The method follows these steps to assign a description:
            1. Checks if the feature name is from a pattern ID and assigns a pattern description.
            2. Checks if the feature name is from a numeric encoder and assigns a numeric description.
            3. Checks if the feature name is from a custom features transformer and assigns a custom feature description.
            4. Checks if the feature name is from datetime encoding and assigns a datetime description.
            5. Checks if the feature name is from one-hot encoding and assigns a one-hot encoding description.
            6. If none of the above, assigns a default description indicating that the feature description is not defined.
        """
        desc: str = None
        feature_type = None
        range_suffix = NumericEncoder.range_suffix
        power_sign = NumericEncoder.power_sign
        ohe_separator = FeatureBuilder.prefix_separator
        anomaly_prefix = CustomFeaturesTransformer.anomaly_prefix
        datetime_suffixes = [
            DateTimeEncoder.quarter_suffix,
            DateTimeEncoder.month_suffix,
            DateTimeEncoder.day_suffix,
            DateTimeEncoder.weekday_suffix,
        ]

        pattern_id_col = (
            Setup().global_constants.get("TEST_PATTERNS", {}).get("PATTERN_ID_COLUMN")
        )
        pattern_description_col = (
            Setup()
            .global_constants.get("TEST_PATTERNS", {})
            .get("PATTERN_DESCRIPTION_COLUMN")
        )

        # Step 1: Check if feature name is from pattern id
        if any(
            pattern_id.lower() in feature_name.lower()
            for pattern_id in self.patterns_data[pattern_id_col].values
        ):
            pattern_id = [
                pattern_id
                for pattern_id in self.patterns_data[pattern_id_col].values
                if pattern_id.lower() in feature_name.lower()
            ][0]
            description = self.patterns_data[
                self.patterns_data[pattern_id_col] == pattern_id
            ][pattern_description_col].values[0]
            desc = f'PatternID: "{feature_name}". Pattern Description: {description}'
            feature_type = "Test Pattern"

        # Step 2: Check if feature name is from numeric encoder
        elif any(
            en_ft_name.lower() in feature_name.lower()
            for en_ft_name in encoder.numeric_encoder.get_feature_names_out()
        ):

            if range_suffix in feature_name:
                col_name = feature_name.split(range_suffix)[0]
                value = feature_name.split(range_suffix)[1]
                desc = f'Data column "{col_name}" has value in range of "{value.split(ohe_separator)[1]}"'
            elif power_sign in feature_name:
                col_name = feature_name.split(power_sign)[0]
                power = feature_name.split(power_sign)[1]
                desc = f'This feature represents "{col_name}" adjusted to fit a bell-shaped curve, making the values more comparable and meaningful. Then the resulting values are raised to the power of {power}.'
            else:
                desc = f'This feature represents "{feature_name}" adjusted to fit a bell-shaped curve, making the values more comparable and meaningful.'
            feature_type = "Numeric"

        # Step 3: Check if feature name is from custom features transformer
        elif any(
            en_ft_name.lower() in feature_name.lower()
            for en_ft_name in encoder.custom_features_transformer.get_feature_names_out()
        ):
            if feature_name.startswith(anomaly_prefix):
                base_feature = feature_name.split(anomaly_prefix)[1]
                feature_type = "Anomaly"
                return feature_mapper.get_description(base_feature), feature_type

        # Step 4: Check if feature name is from datetime encoding (MOVED BEFORE ONE-HOT ENCODER CHECK)
        elif any(
            dt_suffix.lower() in feature_name.lower() for dt_suffix in datetime_suffixes
        ):
            for suffix in datetime_suffixes:

                if suffix in feature_name:
                    parts = feature_name.split(suffix)
                    if len(parts) >= 1:
                        col_name = parts[0]
                        value = parts[1].lstrip("=") if len(parts) > 1 else ""
                        feature_type = "Datetime"
                        if suffix == DateTimeEncoder.quarter_suffix:
                            desc = f'This feature represents the quarter "{value}" of datetime column "{col_name}".'
                        elif suffix == DateTimeEncoder.month_suffix:
                            desc = f'This feature represents the month "{value}" of datetime column "{col_name}".'
                        elif suffix == DateTimeEncoder.day_suffix:
                            desc = f'This feature represents the day of datetime column "{col_name}".'
                        elif suffix == DateTimeEncoder.weekday_suffix:
                            desc = f'This feature represents the weekday "{value}" of datetime column "{col_name}".'
                        break

        # Step 5: Check if feature name is from one hot encoding
        elif any(
            ohe_ft_name.lower() in feature_name.lower()
            for ohe_ft_name in encoder.onehot_encoder.get_feature_names_out()
        ):

            col_name = feature_name.split(ohe_separator)[0]
            value = feature_name.split(ohe_separator)[1]
            # Additional check to ensure this isn't actually a datetime feature
            if not any(
                dt_suffix.lower() in col_name.lower() for dt_suffix in datetime_suffixes
            ):
                feature_type = "Categorical"
                return (
                    f'Data column "{col_name}" with a value of "{value}"',
                    feature_type,
                )

        # Step 6: All other features
        else:
            desc = "Feature description not defined"
            feature_type = "Other"
        return desc, feature_type

    def _update_feature_descriptions(self):
        """
        Updates the feature descriptions for each model in the self.models list.
        For each model, it retrieves the feature importance dictionary, assigns a description to each feature using the
        assign_desc method, and updates the model's feature_importance attribute with the new information.
        If an error occurs during the update process, it logs the failure status with the relevant error message.
        Raises:
            BaseException: If an error occurs during the update process, it logs the failure status with the relevant error message.
        """
        for model in self.models:
            try:
                feature_importance = model._feature_importance
                # feature importance is a dictionary with feature names as keys and importance as values
                updated_feature_importance = {}
                for feature, importance in feature_importance.items():
                    desc, f_type = self._assign_desc(feature, model.encoders)
                    importance: Dict = {
                        "importance": importance,
                        "description": desc,
                        "type": f_type,
                    }
                    updated_feature_importance[feature] = importance
                model.feature_importance = updated_feature_importance
            except BaseException as _e:
                Status.FAILED(
                    "Can not update feature definitions",
                    self.submodule_obj,
                    model=model.name,
                    error=_e,
                )
                continue

    def train(self, X: dd.DataFrame, y: dd.Series):
        """
        Trains machine learning models for the submodule using the provided feature matrix and target.
        This method prepares the input data, initializes a binary risk classifier, trains models for each classifier,
        sets the necessary encoders for each trained model, and updates feature descriptions.
        Args:
        ----
            X (dd.DataFrame): The input feature matrix as a Dask DataFrame.
            y (dd.Series): The target variable as a Dask Series.

        Returns:
            None
        """
        # prepare data
        X, y = self._prepare_data(X, y)

        Status.INFO(
            "Training models",
            instance_id=self.submodule_obj.instance_id,
            module=self.submodule_obj.module,
            submodule=self.submodule_obj.submodule,
        )

        name_prefix = f"{self.submodule_obj.module}_{self.submodule_obj.submodule}"
        bclf = BinaryRiskClassifier(
            name_prefix=name_prefix,
            test_size=self.submodule_obj.ml_params.test_size,
            n_splits=self.submodule_obj.ml_params.n_splits,
            synthentic_data=self.submodule_obj.ml_params.need_synthetic,
        )

        # train models
        self.models = bclf.train_per_classifier(X, y)

        # set encoders
        for model in self.models:
            model.set_encoders(self.encoders)

        # update feature definitions
        self._update_feature_descriptions()

    def track(self) -> Experiment:
        """
        Tracks the experiment by generating an experiment and registering models.
        This method creates an instance of `ModelTracker` using the provided submodule object,
        generates an experiment, and attempts to register each model in the experiment.
        If a model registration fails, it logs the failure and continues with the next model.

        Returns:
            Experiment: The generated experiment object.

        Raises:
            BaseException: If there is an error during model registration.
        """
        # generate experiment
        model_tracker = ModelTracker(submodule_obj=self.submodule_obj)
        experiment = model_tracker.generate_experiment()

        if len(self.models) == 0:
            Status.FAILED(
                "No models to register",
                self.submodule_obj,
            )
            return experiment

        for model in self.models:
            try:
                model_tracker.register(model, experiment.experiment_id, self.params)
            except BaseException as _e:
                Status.FAILED(
                    "Can not register model",
                    self.submodule_obj,
                    model=model.name,
                    error=_e,
                )
                continue

        return experiment

    def _prepare_data(
        self, X: dd.DataFrame, y: dd.Series
    ) -> Tuple[dd.DataFrame, dd.Series]:
        """
        Prepares the data for training by applying a series of transformations and feature engineering steps.
        Args:
            X (dd.DataFrame): The input features as a Dask DataFrame.
            y (dd.Series): The target variable as a Dask Series.
        Returns:
            Tuple[dd.DataFrame, dd.Series]: The transformed features and target variable.
        Steps:
            1. Initializes a custom feature encoder.
            2. Creates a custom features transformer.
            3. Sets up a preprocessing pipeline.
            4. Builds features using the feature builder.
            5. Selects features using the feature selection module.
            6. Constructs a data pipeline with the above steps.
            7. Fits and transforms the input data using the data pipeline.
            8. Extracts and stores parameters related to features and records.
            9. Retrieves and stores a summary of missing data.
            10. Validates the transformed data.
        Raises:
            ValidationError: If the data validation fails.
        """
        Status.INFO("Preparing data for training", self.submodule_obj)
        # create feature encoder
        encoder = CustomFeatureEncoder()

        # create the pipeline
        cft = CustomFeaturesTransformer(
            submodule=self.submodule_obj, index_column=X.index.name
        )
        encoder.custom_features_transformer = cft
        pp = PreProcess(submodule=self.submodule_obj, encoder=encoder)
        fb = FeatureBuilder(
            submodule=self.submodule_obj, preprocessor=pp, encoder=encoder
        )
        fs = FeatureSelection(submodule=self.submodule_obj, feature_builder=fb)

        data_pipeline = Pipeline(
            steps=[
                ("Custom Features", cft),
                ("PreProcessing", pp),
                ("Feature Building", fb),
                ("Feature Selection", fs),
            ],
            verbose=False,
        )

        # prepare input data
        X = data_pipeline.fit_transform(X, y)
        # create parameters
        self.params = {
            "features_to_keep": fs.get_features_to_keep(),
            "features_to_drop": fs.get_features_to_drop(),
            "total_records": len(X.index),
            "total_concern_records": len(y[y == 1]),
            "total_no_concern_records": len(y[y == 0]),
        }

        self.missing_summary = pp.get_missing_summary()
        self.encoders = encoder

        return DataValidator(submodule=self.submodule_obj).validate_data(X, y)
