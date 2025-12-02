# # Copyright (C) KonaAI - All Rights Reserved
from typing import Dict

from src.automl.model_tracker import ModelTracker
from src.utils.global_config import GlobalSettings
from src.utils.submodule import Submodule


def get_headers(authorization):
    instance_id = GlobalSettings().active_instance_id
    client_id = GlobalSettings.instance_by_id(instance_id).ClientUID
    project_id = GlobalSettings.instance_by_id(instance_id).ProjectUID

    return {
        "clientId": client_id,
        "projectId": project_id,
        "Module": "P2P",
        "Submodule": "Purchase Order",
        **authorization,
    }


def validate_model_results(result, key, value_type):
    if value_type == "str":
        assert isinstance(result.get(key, ""), str), f"Expected '{key}' to be a string"
        assert len(result.get(key, "")) > 0, f"'{key}' should not be empty"

    elif value_type == "dict":
        assert isinstance(result.get(key, {}), dict), f"Expected '{key}' to be a dict"
        assert len(result.get(key)) > 0, f"'{key}' should not be empty"

    elif value_type == "list":
        assert isinstance(result.get(key, []), list), f"Expected '{key}' to be a list"
        assert len(result.get(key)) > 0, f"'{key}' should not be empty"

    else:
        raise ValueError(f"Unsupported expected type: {value_type}")


def test_set_active_model(client, authorization):
    headers = get_headers(authorization)
    module = headers.get("Module", "P2P")
    submodule = headers.get("Submodule", "Purchase Order")

    # get a ready model name
    sub = Submodule(module, submodule, GlobalSettings().active_instance_id)
    latest_experiment = ModelTracker(sub).ml_experiments[0]
    models = ModelTracker(sub).get_ml_models([latest_experiment])
    # set modelName in headers
    headers["modelName"] = models[0].name

    response = client.post("/api/automl/activemodel", headers=headers)

    assert response.status_code == 200, "Failed to set active model"


def test_get_activemodel(client, authorization):
    response = client.get(
        "/api/automl/activemodel",
        headers=get_headers(authorization),
    )

    assert response.status_code == 200, "Failed to get active model"

    # Get response JSON
    result = response.json()
    assert isinstance(result, Dict), "Expected result to be a dict"

    # Extract model data from 'active_model data' field
    result = result.get("data", {})
    assert isinstance(result, Dict), "Expected 'data' field to be a dict"

    # Define expected fields and their types
    expected_keys = [
        # (key, value_type),
        ("category", "str"),
        ("confusion_matrix", "dict"),
        ("feature_importance", "list"),
        ("metrics", "dict"),
        ("name", "str"),
        ("params", "dict"),
    ]

    # Check for missing keys
    missing_keys = [key for key, _ in expected_keys if key not in result]
    assert not missing_keys, (
        f"Missing keys in 'data': {missing_keys}. "
        f"Available keys: {list(result.keys())}"
    )

    # Validate the type of each expected field (delegated to a utility function)
    for key, value_type in expected_keys:  # sourcery skip: no-loop-in-tests
        validate_model_results(result, key, value_type)


# def test_get_experiments(client, authorization):
#     response = client.get(
#         "/api/automl/experiments",
#         headers=get_headers(authorization),
#     )
#     assert response.status_code == 200, "Failed to get active model"
#     result = response.json()
#     assert isinstance(result, Dict), "Expected result to be a dict"
#     result = result.get("experiments", [])
#     expected_keys = [
#         # (key, value_type),
#         ("experiments", "list")
#     ]
#     assert all(
#         expected_keys
#     ), f"Following keys missing: {{key[0] for key in expected_keys}} - {set(result.keys())}"

#     for key, value_type in expected_keys:  # sourcery skip: no-loop-in-tests
#         validate_model_results(result, key, value_type)


# def test_get_experiments_results(client, authorization):
#     headers = get_headers(authorization)
#     module = headers.get("Module", "P2P")
#     submodule = headers.get("Submodule", "Invoices")

#     # get a ready model name
#     sub = Submodule(module, submodule, GlobalSettings().active_instance_id)
#     latest_experiment = ModelTracker(sub).ml_experiments[0]
#     headers["experimentName"] = latest_experiment.name

#     response = client.get(
#         "/api/automl/experiments/experiment_name",
#         headers=headers,
#     )
#     assert response.status_code == 200, "Failed to get active model"
#     result = response.json()
#     assert isinstance(result, Dict), "Expected result to be a dict"
#     result = result.get("experiment_result", {})
#     # Assert expected metric keys
#     expected_metrics = {
#         "Accuracy": "int",
#         "Balanced Accuracy": "int",
#         "F1": "int",
#         "Precision": "int",
#         "Recall": "int",
#         "Roc Auc": "int",
#         "Seconds To Train": "int",
#     }

#     # Assert presence of expected metrics and call validation function
#     for model_name, metrics in result.items():  # sourcery skip: no-loop-in-tests
#         assert all(
#             item in metrics.keys() for item in expected_metrics
#         ), f"Missing expected metrics in experiment '{model_name}': {expected_metrics - set(metrics.keys())}"
#         for (  # sourcery skip: no-loop-in-tests
#             key,
#             value_type,
#         ) in expected_metrics.items():  # Iterate over metric names (keys)
#             validate_model_results(result, key, value_type)


# def test_get_questionnaire_details(client, authorization):
#     response = client.get(
#         "/api/automl/questionnaire",
#         headers=get_headers(authorization),
#     )
#     assert response.status_code == 200, "Failed to get active model"
#     result = response.json()
#     assert isinstance(result, Dict), "Expected result to be a dict"
#     result = result.get("questionnaire_details", {})
#     # Expected metrics and data types
#     expected_metrics = {
#         "alert_status": str,
#         "concern_questionnaire": list,
#         "no_concern_questionnaire": list,
#     }
#     # Assert presence of expected metrics and call validation function
#     assert all(
#         item in result.keys() for item in expected_metrics
#     ), f"Missing expected metrics: {expected_metrics.keys() - set(result.keys())}"

#     for key, value_type in expected_metrics.items():  # sourcery skip: no-loop-in-tests
#         validate_model_results(result, key, value_type)
