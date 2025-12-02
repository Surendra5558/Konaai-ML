# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the code for classifiers."""
import copy
import logging
import sys
import time
import warnings
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Self
from typing import Tuple

import dask.dataframe as dd
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.tree import DecisionTreeClassifier
from src.automl.model import Model
from src.automl.model import ModelMetrics
from src.automl.splitter import split_train_test
from src.utils.status import Status
from tqdm import tqdm
from xgboost import XGBClassifier

# import classifiers

# change lightgbm logging to error only
lgb_logger = logging.getLogger("lightgbm")
lgb_logger.setLevel(logging.CRITICAL)

# change optuna logging to error only
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.ERROR)


# ignore lightgbm warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AlgoType(Enum):
    """
    Enumeration of supported classifier algorithm types.
    Attributes:
    ----------
        TREE (str): Represents tree-based algorithms (e.g., decision trees, random forests).
        LINEAR (str): Represents linear algorithms (e.g., logistic regression, linear SVM).
        KERNEL (str): Represents kernel-based algorithms (e.g., SVM with RBF kernel).
    """

    TREE = "tree"
    LINEAR = "linear"
    KERNEL = "kernel"


class CustomClassifier(BaseEstimator):
    """
    Custom classifier class.
    Attributes:
    ----------
        name (str): The name of the classifier.
        model: The classifier model.
        feature_importance_property (str): The property for feature importance.
    """

    name: str
    model: None
    feature_importance_property: str
    params: Dict = {}  # hyperparameters
    n_trials: int = 150  # number of trials for hyperparameter tuning
    algo_type: AlgoType = None

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        name,
        model,
        feature_importance_property,
        params=None,
        n_trials=150,
        algo_type=None,
    ):
        if params is None:
            params = {}
        self.name = name
        self.model = model
        self.feature_importance_property = feature_importance_property
        self.params = params
        self.n_trials = n_trials
        self.algo_type = algo_type

    def get_feature_importance(self, X_train: dd.DataFrame) -> Dict:
        """
        Calculate and return the feature importances of the trained model.

        Parameters:
        X_train (dd.DataFrame): The training data used to fit the model. It should be a Dask DataFrame.

        Returns:
        Dict: A dictionary where the keys are feature names and the values are their corresponding importance scores.
              If the model does not have the specified feature importance property, an empty dictionary is returned.

        Raises:
        Status.INVALID_INPUT: If the length of features and importances are not the same.
        """
        __importances = []

        if hasattr(self.model, self.feature_importance_property):

            if self.feature_importance_property == "coef_":
                __importances = [abs(item) for item in self.model.coef_[0]]
            elif self.feature_importance_property == "feature_importances_":
                __importances = [abs(item) for item in self.model.feature_importances_]

            if len(X_train.columns) != len(__importances):
                Status.INVALID_INPUT("Length of features and importances is not same")
                return {}

            _importances = list(zip(X_train.columns, __importances))
            # NOTE: Do not sort the importances as it will change the order of features,
            # when we are using the feature_importance in the prediction pipeline,
            # if the order of features is changed, it will cause an error.
            return {f: round(float(i), 4) for f, i in _importances}

        return {}

    def fit(self, X: dd.DataFrame, y: dd.Series) -> Self:
        """This function fits the model"""
        # tune the decision threshold
        self.model = self.model.fit(X, y)
        return self


class BinaryRiskClassifier:
    """
    This class provides an automated framework for training, tuning, and evaluating multiple binary classification models. It supports hyperparameter optimization, decision threshold tuning, and model selection based on cross-validation metrics.
    Attributes:
    ----------
        name_prefix (str): Prefix for naming trained models.
        random_state (int): Random seed for reproducibility.
        test_size (float): Proportion of the dataset to include in the test split.
        n_splits (int): Number of cross-validation splits.
        synthentic_data (bool): Whether to use synthetic data for training.
        verbose (bool): Verbosity flag for logging.
    Properties:
        _classifiers (List[CustomClassifier]): List of supported classifiers with their configurations.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        name_prefix: str,
        random_state=42,
        test_size=0.2,
        n_splits=5,
        synthentic_data=False,
        verbose=True,
    ):
        self.name_prefix = name_prefix
        self.verbose = verbose
        self.random_state = random_state
        self.test_size = test_size
        self.n_splits = n_splits
        self.synthentic_data = synthentic_data

    @property
    def _classifiers(self) -> List[CustomClassifier]:
        return [
            CustomClassifier(
                name="Logistic Regression",
                model=LogisticRegression(random_state=self.random_state),
                feature_importance_property="coef_",
                params={
                    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "penalty": [
                        "l1",
                        "l2",
                    ],
                    "solver": ["liblinear"],
                },
                n_trials=50,
                algo_type=AlgoType.LINEAR,
            ),
            CustomClassifier(
                name="Decision Tree",
                model=DecisionTreeClassifier(random_state=self.random_state),
                feature_importance_property="feature_importances_",
                params={
                    "max_depth": [3, 5, 7, 9, 11, 13],
                    "min_samples_split": [2, 4, 6, 8, 10],
                    "min_samples_leaf": [1, 2, 4, 6, 8],
                },
                n_trials=100,
                algo_type=AlgoType.TREE,
            ),
            CustomClassifier(
                name="Random Forest",
                model=RandomForestClassifier(random_state=self.random_state),
                feature_importance_property="feature_importances_",
                params={
                    "n_estimators": [50, 100, 200, 300, 400, 500],
                    "max_depth": [3, 5, 7, 9, 11, 13],
                    "min_samples_split": [2, 4, 6, 8, 10],
                    "min_samples_leaf": [1, 2, 4, 6, 8],
                },
                n_trials=150,
                algo_type=AlgoType.TREE,
            ),
            CustomClassifier(
                name="XGBoost",
                model=XGBClassifier(random_state=self.random_state),
                feature_importance_property="feature_importances_",
                params={
                    "n_estimators": [50, 100, 200, 300, 400, 500],
                    "max_depth": [3, 5, 7, 9, 11, 13],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bynode": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                },
                n_trials=150,
                algo_type=AlgoType.TREE,
            ),
            CustomClassifier(
                name="LightGBM",
                model=lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                feature_importance_property="feature_importances_",
                params={
                    "n_estimators": [50, 100, 200, 300, 400, 500],
                    "max_depth": [3, 5, 7, 9, 11, 13],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bynode": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                },
                n_trials=150,
                algo_type=AlgoType.TREE,
            ),
            CustomClassifier(
                name="AdaBoost",
                model=AdaBoostClassifier(random_state=self.random_state),
                feature_importance_property="feature_importances_",
                params={
                    "n_estimators": [50, 100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                },
                n_trials=100,
                algo_type=AlgoType.KERNEL,
            ),
            CustomClassifier(
                name="Gradient Boosting",
                model=GradientBoostingClassifier(random_state=self.random_state),
                feature_importance_property="feature_importances_",
                params={
                    "n_estimators": [50, 100, 200, 300, 400, 500],
                    "max_depth": [3, 5, 7, 9, 11, 13],
                    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "colsample_bynode": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                },
                n_trials=150,
                algo_type=AlgoType.TREE,
            ),
        ]

    def tune_hyperparameters(  # pylint: disable=too-many-positional-arguments
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        classifier: CustomClassifier,
    ) -> Tuple[CustomClassifier, float]:
        """
        Tunes the hyperparameters of the given classifier using Optuna and returns the classifier with the best hyperparameters and the tuned decision threshold.
        Args:
        ----
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training target data.
            X_test (pd.DataFrame): Testing feature data.
            y_test (pd.Series): Testing target data.
            classifier (CustomClassifier): The classifier to be tuned.
        Returns:
            Tuple[CustomClassifier, float]: The classifier with the best hyperparameters and the tuned decision threshold.
        """
        Status.INFO(f"Tuning hyperparameters for {classifier.name}")

        def objective(trial):
            # set the hyperparameters
            for key, value in classifier.params.items():
                setattr(classifier.model, key, trial.suggest_categorical(key, value))

            # fit the model
            classifier.fit(X_train, y_train)

            # predict
            y_pred = classifier.model.predict(X_test)

            # calculate f1 score
            f1 = f1_score(y_test, y_pred)

            return f1

        # create a study
        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )

        # optimize the study
        study.optimize(
            objective,
            timeout=10 * 60,  # 10 minutes
            n_trials=classifier.n_trials,
            show_progress_bar=False,
            # n_jobs=-1,
        )

        # get the best hyperparameters
        best_params = study.best_params

        # print best f1 score
        Status.INFO(f"F1 score: {study.best_value}")

        # set the hyperparameters
        for key, value in best_params.items():
            setattr(classifier.model, key, value)

        # tune the decision threshold
        tuned_clf, threshold = self.tune_decision_threshold(
            classifier.model, X_train, y_train
        )

        Status.INFO(f"Decision Threshold: {threshold}")

        # set the model
        classifier.model = tuned_clf

        return classifier, threshold

    def tune_decision_threshold(
        self, model: Any, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Any, float]:
        """
        Tunes the decision threshold of a given model to maximize the precision score.
        Parameters:
        -----------
        model : Any
            The machine learning model to be tuned.
        X_train : pd.DataFrame
            The training data features.
        y_train : pd.Series
            The training data labels.
        Returns:
        --------
        Tuple[Any, float]
            A tuple containing the tuned classifier and the best decision threshold rounded to 3 decimal places.
        """
        Status.INFO("Tuning decision threshold")
        # Create a scorer
        # Our objective is to maximize precision score
        # Originally tried with f1 score, but it decision threshold were very low
        _scorer = make_scorer(precision_score, pos_label=1, zero_division=np.nan)

        # Wrap the classifier with TunedThresholdClassifierCV to optimize the threshold for precision
        tuned_clf = TunedThresholdClassifierCV(model, scoring=_scorer).fit(
            X_train, y_train
        )

        return tuned_clf, round(tuned_clf.best_threshold_, 3)

    def calculate_metrics(self, y_pred: dd.Series, y_test: dd.Series) -> ModelMetrics:
        """This function calculates the metrics"""
        metrics = ModelMetrics()
        metrics.accuracy = round(accuracy_score(y_test, y_pred), 3)
        metrics.balanced_accuracy = round(balanced_accuracy_score(y_test, y_pred), 3)
        metrics.roc_auc = round(roc_auc_score(y_test, y_pred), 3)
        metrics.f1 = round(f1_score(y_test, y_pred), 3)
        metrics.precision = round(precision_score(y_test, y_pred), 3)
        metrics.recall = round(recall_score(y_test, y_pred), 3)
        return metrics

    def train_with_iterations(
        self, X: dd.DataFrame, y: dd.Series, classifier: CustomClassifier
    ) -> Model:
        """
        Train a classifier with multiple iterations using cross-validation.
        Parameters:
        -----------
        X : dd.DataFrame
            The input features for training.
        y : dd.Series
            The target labels for training.
        classifier : CustomClassifier
            The classifier to be trained.
        Returns:
        --------
        Model
            The best model obtained after training with cross-validation.
        Raises:
        -------
        ValueError
            If all the predictions are 0 or if the precision of the current model is less than the best model.
        BaseException
            If any other exception occurs during training, the iteration is skipped.
        """
        start_time = time.perf_counter()
        best_model: Model = None
        for i, split in enumerate(
            split_train_test(
                X,
                y,
                test_size=self.test_size,
                n_splits=self.n_splits,
                synthentic_data=self.synthentic_data,
            )
        ):
            X_train, X_test, y_train, y_test = split

            try:
                Status.INFO(
                    f"Training {classifier.name}", iteration=f"{i+1}/{self.n_splits}"
                )

                # create a copy of the classifier
                clf = copy.deepcopy(classifier)

                # train the model
                trained_clf = clf.fit(X_train, y_train)

                # get feature importance
                feature_importance = trained_clf.get_feature_importance(X_train)

                # hyperparameter tuning
                tuned_clf, decision_threshold = self.tune_hyperparameters(
                    X_train, y_train, X_test, y_test, trained_clf
                )

                # if decision threshold is None or 0, then skip this iteration
                if not decision_threshold:
                    Status.WARNING(f"Invalid decision threshold: {decision_threshold}")
                    continue

                # predict
                y_pred = tuned_clf.model.predict(X_test)

                # check if all the predictions are 0
                if y_pred.sum() == 0:
                    Status.WARNING("All the predictions are 0")
                    continue

                # check if all the predictions are 1
                if y_pred.sum() == len(y_pred):
                    Status.WARNING("All the predictions are 1")
                    continue

                # calculate metrics
                metrics = self.calculate_metrics(y_pred, y_test)

                # check if f1 score is 0, underfitting
                if metrics.f1 in [0, 1]:
                    Status.WARNING(f"Unacceptable F1 score: {metrics.f1}")
                    continue

                metrics.decision_threshold = decision_threshold

                # if previous model is not None, then compare the metrics
                best_precision = best_model.metrics.precision if best_model else 0
                if (
                    best_model
                    and best_precision > metrics.precision
                    and best_precision != 1.0
                ):
                    Status.WARNING(
                        f"Precision {metrics.precision} is less than the best model: {best_precision}"
                    )
                    continue

                # if previous model is not None, then compare the metrics
                best_f1 = best_model.metrics.f1 if best_model else 0
                if (
                    best_model
                    and best_precision == metrics.precision
                    and best_f1 > metrics.f1
                    and best_f1 != 1.0
                ):
                    Status.WARNING(
                        f"Precision is same as best model. F1 score {metrics.f1} is less than the best model: {best_f1}"
                    )
                    continue

                # store model
                model = Model(
                    name=tuned_clf.name,
                    category="classifier",
                    model=tuned_clf.model,
                    metrics=metrics,
                )

                # get feature importance
                model._feature_importance = feature_importance

                model.confusion_matrix = confusion_matrix(
                    y_test, y_pred, normalize="all"
                )

                # calculate base shap value as the mean of the predicted probabilities
                model.base_shap_value = tuned_clf.model.predict_proba(X_test)[
                    :, 1
                ].mean()

                # store the best model
                best_model = model
            except BaseException as _e:
                Status.WARNING("Skipping this iteration.", error=_e)
                continue

        # assign time to train
        if best_model:
            best_model.metrics.seconds_to_train = round(
                time.perf_counter() - start_time, 3
            )
            Status.SUCCESS(
                f"Training completed for {classifier.name}",
                metrics=best_model.metrics,
            )
        else:
            Status.FAILED(f"No model trained for {classifier.name}")

        return best_model

    def train_per_classifier(self, X: dd.DataFrame, y: dd.Series) -> List[Model]:
        """
        Trains multiple classifiers on the provided feature set and target, returning a list of trained models.
        Args
        ----
            X (dd.DataFrame): The input features as a Dask DataFrame.
            y (dd.Series): The target variable as a Dask Series.
        Returns:
            List[Model]: A list of successfully trained Model instances.

        Notes:

            - Each classifier in `self._classifiers` is trained using `train_with_iterations`.
            - The trained model's name and algorithm type are updated.
            - Training data (features and target) is stored in each model for later use (e.g., SHAP explainers).
            - Any errors during training are logged, and failed classifiers are skipped.
            - Only successfully trained models (not None) are included in the returned list.
        """

        models = []
        for classifier in tqdm(
            self._classifiers, desc="Training classifiers", file=sys.stdout
        ):
            try:
                Status.INFO(f"Training {classifier.name}")
                if model := self.train_with_iterations(X, y, classifier):
                    model.name = f"{self.name_prefix}_{model.name}"
                    model.algo_type = classifier.algo_type.value

                    # Save training data to be used for shap explainer
                    X_copy = X.copy()
                    X_copy["target"] = y
                    model.training_data = X_copy
                    del X_copy

                    models.append(model)
            except BaseException as _e:
                Status.FAILED(f"Error while training {classifier.name}", error=_e)

        # remove None models
        models = [model for model in models if model is not None]
        Status.INFO(
            f"Training completed. Total {len(models)} models trained",
            # models=", ".join([model.name for model in models]),
        )

        return models

    def fit(self, X: dd.DataFrame, y: dd.Series) -> List[Model]:
        """This function fits the model and returns the metrics"""
        return self.train_per_classifier(X, y)
