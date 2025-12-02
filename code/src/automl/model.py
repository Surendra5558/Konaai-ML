# # Copyright (C) KonaAI - All Rights Reserved
"""This module is used to create the model class"""
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic import computed_field
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import ValidationError
from src.automl.ml_params import MLParameters
from src.utils.status import Status

# Commented out to avoid OpenAPI schema issues - using Any instead
# from mlflow.entities.model_registry import RegisteredModel


class ModelMetrics(BaseModel):
    """
    ModelMetrics is a Pydantic model that encapsulates various evaluation metrics for a machine learning model.
    Attributes:
    ----------
        accuracy (float): The accuracy of the model, ranging from 0 to 1.
        balanced_accuracy (float): The balanced accuracy of the model, ranging from 0 to 1.
        roc_auc (float): The ROC AUC (Receiver Operating Characteristic Area Under Curve) of the model, ranging from 0 to 1.
        f1 (float): The F1 score of the model, ranging from 0 to 1.
        precision (float): The precision of the model, ranging from 0 to 1.
        recall (float): The recall of the model, ranging from 0 to 1.
        seconds_to_train (float): The time taken to train the model, in seconds (must be >= 0).
        decision_threshold (float): The decision threshold used by the model, ranging from 0 to 1.
    """

    # define the metrics
    accuracy: float = Field(
        None, title="Accuracy", description="The accuracy of the model", ge=0, le=1
    )
    balanced_accuracy: float = Field(
        None,
        title="Balanced Accuracy",
        description="The balanced accuracy of the model",
        ge=0,
        le=1,
    )
    roc_auc: float = Field(
        None, title="ROC AUC", description="The ROC AUC of the model", ge=0, le=1
    )
    f1: float = Field(
        None, title="F1 Score", description="The F1 score of the model", ge=0, le=1
    )
    precision: float = Field(
        None, title="Precision", description="The precision of the model", ge=0, le=1
    )
    recall: float = Field(
        None, title="Recall", description="The recall of the model", ge=0, le=1
    )
    seconds_to_train: float = Field(
        None,
        title="Time to Train",
        description="The time taken to train the model",
        ge=0,
    )
    decision_threshold: float = Field(
        None,
        title="Decision Threshold",
        description="The decision threshold of the model",
        ge=0,
        le=1,
    )

    def to_dict(self):
        """
        Converts the current object instance into a dictionary representation.

        Returns:
            dict: A dictionary containing all the fields and their values from the object.
        """
        # convert to dynamic dictionary
        return self.model_dump()

    def __str__(self):
        return f"[{', '.join(f'{str(k).upper()}: {float(v)}' for k, v in self.model_dump().items())}]"


class FeatureImportance(BaseModel):
    """
    Represents the importance of a feature in a machine learning model.

    Attributes:
    ----------
        feature_name (str): The name of the feature.
        importance (float): The importance score of the feature. Must be greater than or equal to 0.
        description (Optional[str]): A brief description of the feature.
        type (Optional[str]): The type of the feature. Defaults to "Other".
    """

    feature_name: str = Field(
        title="Feature Name", description="The name of the feature"
    )
    importance: float = Field(
        title="Feature Importance",
        description="The importance score of the feature",
        ge=0,
    )
    description: Optional[str] = Field(
        None,
        title="Feature Description",
        description="A brief description of the feature",
    )
    type: Optional[str] = Field(
        "Other", title="Feature Type", description="The type of the feature"
    )


class ConfusionMatrix(BaseModel):
    """
    Represents the percentage breakdown of a confusion matrix.

    Attributes:
    ---------
        true_negative_percent (float): The percentage of true negatives (correctly predicted negatives) in the confusion matrix. Must be between 0 and 100.
        false_positive_percent (float): The percentage of false positives (incorrectly predicted positives) in the confusion matrix. Must be between 0 and 100.
        false_negative_percent (float): The percentage of false negatives (incorrectly predicted negatives) in the confusion matrix. Must be between 0 and 100.
        true_positive_percent (float): The percentage of true positives (correctly predicted positives) in the confusion matrix. Must be between 0 and 100.
    """

    true_negative_percent: float = Field(
        title="True Negative Percent",
        description="The percentage of true negatives in the confusion matrix",
        ge=0,
        le=100,
    )
    false_positive_percent: float = Field(
        title="False Positive Percent",
        description="The percentage of false positives in the confusion matrix",
        ge=0,
        le=100,
    )
    false_negative_percent: float = Field(
        title="False Negative Percent",
        description="The percentage of false negatives in the confusion matrix",
        ge=0,
        le=100,
    )
    true_positive_percent: float = Field(
        title="True Positive Percent",
        description="The percentage of true positives in the confusion matrix",
        ge=0,
        le=100,
    )


class Model(BaseModel):
    """
    This class represents a machine learning model and its associated metadata, configuration, and artifacts.
    It is designed to be compatible with Pydantic for data validation and serialization, supporting arbitrary types
    such as numpy arrays and pandas DataFrames.
    Attributes:
    ---------
        name (str): The name of the model.
        created_on (datetime): The date and time when the model was created.
        category (str): The category/type of the model (e.g., classifier, regressor).
        model (Any): The actual model object, which can be any type (MLflow RegisteredModel, sklearn model, etc.).
        metrics (ModelMetrics): The evaluation metrics associated with the model.
        base_shap_value (Optional[float]): The base SHAP value for the model, if available.
        algo_type (Optional[str]): The algorithm type used by the model.
        params (MLParameters): The parameters used to train the model.
        confusion_matrix (np.ndarray): The confusion matrix for the model (not serialized to JSON).
        missing_summary (pd.DataFrame): Summary of missing values in the training data (not serialized to JSON).
        encoders (Dict): Encoders used in the model (not serialized to JSON).
        training_data (pd.DataFrame): The training data used for the model (not serialized to JSON).
        _feature_importance (Dict): Internal storage for feature importance information.
    """

    # define model configuration. This is pydantic internal configuration to allow arbitrary types such as numpy arrays
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # mandatory fields
    name: str = Field(title="Model Name", description="The name of the model")
    created_on: datetime = Field(
        title="Created On",
        description="The date the model was created in UTC",
        default_factory=lambda: datetime.now(timezone.utc),
    )
    category: str = Field(
        title="Model Category", description="The category of the model"
    )
    model: Any = Field(
        default=None, title="Model", description="The model object", exclude=True
    )
    metrics: ModelMetrics = Field(
        title="Model Metrics", description="The metrics of the model"
    )
    base_shap_value: Optional[float] = Field(
        None,
        title="Base SHAP Value",
        description="The base SHAP value of the model",
    )
    algo_type: Optional[str] = Field(
        None,
        title="Algorithm Type",
        description="The algorithm type of the model",
    )

    # optional fields
    params: MLParameters = Field(
        None, title="Model Parameters", description="The parameters of the model"
    )
    confusion_matrix: np.ndarray = Field(
        None,
        title="Confusion Matrix",
        description="The confusion matrix of the model",
        exclude=True,  # excluding since numpy array can not be serialized to json
    )
    missing_summary: pd.DataFrame = Field(
        None,
        title="Missing Summary",
        description="The missing summary of the model",
        exclude=True,
    )
    encoders: Dict = Field(
        None,
        title="Encoders",
        description="The encoders used in the model",
        exclude=True,
    )
    training_data: pd.DataFrame = Field(
        None,
        title="Training Data",
        description="The training data used in the model",
        exclude=True,
    )
    _feature_importance: Dict = None

    def __init__(
        self,
        name: str,
        category: str,
        model: Any,
        metrics: ModelMetrics,
        **data,
    ):
        # call the parent constructor
        super().__init__(
            name=name,
            category=category,
            model=model,
            metrics=metrics,
            **data,
        )

    def set_encoders(self, encoder: Dict):
        """
        Sets the encoders for the model.

        Args:
            encoder (Dict): A dictionary containing encoder objects or mappings to be used by the model.

        Returns:
            None
        """
        self.encoders = encoder

    def serialize_confusion_matrix(
        self, confusion_matrix: np.ndarray
    ) -> ConfusionMatrix:
        """
        Serializes a confusion matrix into a ConfusionMatrix object with rounded percentage values.

        Args:
            confusion_matrix (np.ndarray): A 2x2 numpy array representing the confusion matrix,
                expected in the order [[tn, fp], [fn, tp]] or as a flat array [tn, fp, fn, tp].

        Returns:
            ConfusionMatrix: An object containing the rounded (to 3 decimal places) values for
                true negatives, false positives, false negatives, and true positives as percentages.
        """
        tn, fp, fn, tp = confusion_matrix.ravel()

        return ConfusionMatrix(
            true_negative_percent=round(tn, 3) or 0,
            false_positive_percent=round(fp, 3) or 0,
            false_negative_percent=round(fn, 3) or 0,
            true_positive_percent=round(tp, 3) or 0,
        )

    @computed_field
    @property
    def feature_importance(self) -> List[FeatureImportance]:
        """
        Returns the feature importance for the model as a list of FeatureImportance objects.
        This method supports both the old and new formats of feature importance:
        - Old format: a dictionary with feature names as keys and importance values as values.
        - New format: a dictionary with feature names as keys and values as dictionaries containing
          'importance', 'description', and 'type' fields.

        Returns:
            List[FeatureImportance]: A list of FeatureImportance objects with non-zero importance.

        Raises:
            Status.INVALID_INPUT: If feature importance is not available or is invalid for the model.
        """

        _feature_importance = self._feature_importance

        # check if the feature importance is available
        if (
            _feature_importance is None
            or not isinstance(_feature_importance, dict)
            or len(_feature_importance) == 0
        ):
            Status.INVALID_INPUT(
                f"Feature importance is not available for model {self.name}"
            )
            return []

        # get first value type of the dictionary
        first_key = next(iter(_feature_importance))
        first_value = _feature_importance[first_key]

        transformed_importance: List[FeatureImportance] = []
        for key, value in _feature_importance.items():
            # This is done to enable backward compatibility with old feature importance format
            # old format was a dictionary with feature names as keys and importance values as values
            if not isinstance(first_value, dict):
                transformed_importance.append(
                    FeatureImportance(
                        feature_name=key,
                        importance=value,
                        description="",  # empty string if not provided
                        type="Other",  # default type for backward compatibility
                    )
                )
            else:
                transformed_importance.append(
                    FeatureImportance(
                        feature_name=key,
                        importance=value.get("importance", 0),
                        description=value.get("description", ""),
                        type=value.get("type", "Other"),
                    )
                )

        return transformed_importance

    @feature_importance.setter
    def feature_importance(self, value):
        """This function is used to set the feature importance"""
        self._feature_importance = value

    def model_dump(self):
        """
        Serializes the model's attributes into a dictionary for export or storage.
        Returns:
            dict: A dictionary containing the model's name, creation date (as ISO string),
            category, serialized metrics, serialized parameters, feature importance,
            and a serialized confusion matrix (if available). Fields with None values
            are included as None.
        """
        return {
            "name": self.name,
            "created_on": (
                self.created_on.isoformat() if self.created_on is not None else None
            ),
            "category": self.category,
            "metrics": self.metrics.model_dump() if self.metrics is not None else None,
            "params": self.params.model_dump() if self.params is not None else None,
            "feature_importance": (
                self.feature_importance if self.feature_importance is not None else None
            ),
            "confusion_matrix": (
                self.serialize_confusion_matrix(self.confusion_matrix)
                if self.confusion_matrix is not None
                else None
            ),
        }

    @field_validator("category")
    @classmethod
    def validate_model_type(cls, value):
        """
        Validates that the provided model type is one of the allowed categories.

        Args:
            value (str): The model type to validate.

        Returns:
            str: The validated model type if it is among the allowed categories.

        Raises:
            ValidationError: If the provided model type is not in the list of allowed categories.
        """
        type_choices = [
            "classifier",
            "regressor",
            "anomaly_detector",
            "clustering",
            "dimensionality_reduction",
            "ensemble",
            "transformer",
        ]
        if value not in type_choices:
            raise ValidationError(
                f"Model category should be one of {type_choices}. Got {value} instead."
            )
        return value
