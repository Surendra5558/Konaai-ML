# # Copyright (C) KonaAI - All Rights Reserved
"""AutoML API Endpoints"""
from datetime import datetime
from datetime import timezone
from typing import Dict
from typing import List
from typing import Optional

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Header
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from pydantic import Field
from src.automl.explainer import ExplainationOutput
from src.automl.explainer import PredictionExplainer
from src.automl.fetch_data import ModelData
from src.automl.model import ConfusionMatrix
from src.automl.model import ModelMetrics
from src.automl.model_tracker import ModelTracker
from src.automl.prediction_pipeline import LastPredictionDetails
from src.automl.questionnaire import TemplateQuestion
from src.automl.tasks import predict_from_model
from src.automl.tasks import train_model
from src.butler.api_setup import validate_token
from src.utils.api_response import APIResponse
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule

# Initialize security scheme
security = HTTPBearer()

automl_router = APIRouter(tags=["AutoML"])


class TrainingDataResponse(APIResponse):
    """Response model for training data validation status."""

    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None
    data: Optional[ModelData.TrainingDataValidationResult] = None


@automl_router.get(
    "/trainingdatavalidation",
    response_model=TrainingDataResponse,
    dependencies=[Depends(validate_token)],
)
async def get_data_validation_status(
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """Get the data validation status for the specified module and submodule."""
    try:
        response = TrainingDataResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )
        data = ModelData(submodule_obj=submodule_obj)
        validation_result: ModelData.TrainingDataValidationResult = (
            data.min_data_validator()
        )
        response.data = validation_result

        s = Status.SUCCESS(
            "Data validation check completed",
        )
        return response.assign_status(status=s, data=validation_result)
    except Exception as e:
        s = Status.FAILED(
            "Data validation fetch failed", submodule_obj, error=str(e), traceback=False
        )
        return APIResponse().assign_status(status=s, error=str(e))


class ModelForDocs(BaseModel):
    """
    A subclass of Model that includes an optional confusion matrix for FastAPI documentation purposes.

    Attributes:
    -----------
        confusion_matrix (Optional[ConfusionMatrix]): Stores the confusion matrix associated with the model, if available.
    """

    confusion_matrix: Optional[ConfusionMatrix] = None
    name: str = Field(title="Model Name", description="The name of the model")
    created_on: datetime = Field(
        title="Created On",
        description="The date the model was created in UTC",
        default_factory=datetime.now(timezone.utc),
    )
    category: str = Field(
        title="Model Category", description="The category of the model"
    )
    metrics: ModelMetrics = Field(
        title="Model Metrics", description="The metrics of the model"
    )
    algo_type: Optional[str] = Field(
        None,
        title="Algorithm Type",
        description="The algorithm type of the model",
    )


class ModelResponse(APIResponse):
    """
    ModelResponse represents the response structure for model-related API endpoints.
    Attributes:
    -----------
        data (Optional[ModelForDocs]): The model details or documentation object, if available.
        client_id (str): Identifier for the client associated with the model.
        project_id (str): Identifier for the project to which the model belongs.
        module (str): The main module name related to the model.
        submodule (str): The submodule name related to the model.
    """

    data: Optional[ModelForDocs] = None
    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None


@automl_router.get(
    "/activemodel", response_model=ModelResponse, dependencies=[Depends(validate_token)]
)
async def get_active_model(
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Retrieve the active model for a given module, submodule, client, and project.

    This endpoint expects the following headers:
        - Module: The name of the module.
        - Submodule: The name of the submodule.
        - clientId: The client identifier.
        - projectId: The project identifier.

    Returns:
        APIResponse: An object containing the status and details of the active model if found,
        or an appropriate error/status message otherwise.

    Raises:
        Exception: If any error occurs during the process, returns a failed status with error details.
    """
    try:
        response = ModelResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        # Check if the active model is set
        if active_model := ModelTracker(submodule_obj=submodule_obj).get_model_by_name(
            submodule_obj.active_model
        ):
            # Get the active model details
            s = Status.SUCCESS("Active model found", submodule_obj)
            return response.assign_status(status=s, data=active_model.model_dump())

        s = Status.NOT_FOUND("No active model found", submodule_obj)
        return response.assign_status(status=s)
    except Exception as e:
        s = Status.FAILED(
            "Model fetch failed", submodule_obj, error=str(e), traceback=False
        )
        return APIResponse().assign_status(status=s, error=str(e))


@automl_router.post(
    "/activemodel", response_model=ModelResponse, dependencies=[Depends(validate_token)]
)
async def set_active_model(
    model_name: str = Header(..., alias="modelName"),
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Sets the active model for a given client, project, module, and submodule.
    This endpoint receives model details via HTTP headers, validates the client and project instance,
    and attempts to set the specified model as active for the provided submodule. Returns a response
    indicating success or failure, along with relevant status and model data.
    Args:
    ----
        model_name (str): Name of the model to set as active (from header 'modelName').
        module (str): Name of the module (from header 'Module').
        submodule (str): Name of the submodule (from header 'Submodule').
        client_id (str): Client identifier (from header 'clientId').
        project_id (str): Project identifier (from header 'projectId').
    Returns:
        APIResponse: Response object containing status, error details (if any), and model data.

    Raises:
        RuntimeError: If unable to set the active model.
    """
    try:
        response = ModelResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        # Validate the model name
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        if submodule_obj.set_active_model(model_name):
            s = Status.SUCCESS("Active model set", submodule_obj)
            return response.assign_status(
                status=s,
                data=ModelTracker(submodule_obj=submodule_obj)
                .get_model_by_name(model_name)
                .model_dump(),
            )

        raise RuntimeError("Contact support")
    except Exception as e:
        s = Status.FAILED(
            "Failed to set active model", submodule_obj, error=str(e), traceback=False
        )
        return APIResponse().assign_status(status=s, error=str(e))


class TaskResponse(APIResponse):
    """
    AutoML response class to handle model training and prediction.
    Attributes:
    ----------
        client_id (str): Identifier for the client.
        project_id (str): Identifier for the project.
        module (str): Name of the module associated with the task.
        submodule (str): Name of the submodule associated with the task.
        task_id (Optional[str]): Unique identifier for the specific task.
    """

    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None
    task_id: Optional[str] = None


@automl_router.post(
    "/train", response_model=TaskResponse, dependencies=[Depends(validate_token)]
)
async def start_experiment_training(
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Starts the training process for an experiment asynchronously.
    This endpoint expects specific headers to identify the module, submodule, client, and project.
    It retrieves the corresponding instance, initializes the submodule object, and triggers the model training task.
    Returns a response indicating the status of the training initiation, including the task ID if successful.
    Handles errors gracefully and returns appropriate status messages.
    Args:
    ----
        module (str): The name of the module, provided via the "Module" header.
        submodule (str): The name of the submodule, provided via the "Submodule" header.
        client_id (str): The client identifier, provided via the "clientId" header.
        project_id (str): The project identifier, provided via the "projectId" header.
    Returns:
        APIResponse: An object containing the status of the training initiation, task ID, and error details if any.
    """
    try:
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = TaskResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )
        task = train_model.delay(
            module=module, submodule=submodule, instance_id=instance.instance_id
        )
        s = Status.SUCCESS("Training started", task_id=task.id, submodule=submodule_obj)
        return response.assign_status(status=s, task_id=task.id)
    except Exception as e:
        s = Status.FAILED(
            "Can not start training",
            submodule=submodule_obj,
            error=str(e),
            traceback=False,
        )
        return APIResponse().assign_status(status=s, error=str(e))


class BulkPredictionResponse(APIResponse):
    """
    BulkPredictionResponse represents the response structure for bulk prediction API calls.
    Attributes:
    ----------
        client_id (str): The client identifier associated with the prediction request.
        project_id (str): The project identifier under which the prediction is performed.
        module (str): The module name for which the prediction is requested.
        data (Optional[Dict[str, str]]): A dictionary mapping submodule names to their corresponding task IDs.
        error (Optional[Dict[str, str]]): A dictionary mapping submodule names to error messages, if any occurred during prediction.
    """

    client_id: str = None
    project_id: str = None
    module: str = None
    data: Optional[Dict[str, str]] = Field(
        None, description="Task IDs for each submodule"
    )
    error: Optional[Dict[str, str]] = Field(
        None, description="List of errors for each submodule"
    )


@automl_router.post(
    "/bulkpredict",
    response_model=BulkPredictionResponse,
    dependencies=[Depends(validate_token)],
)
async def bulk_prediction(
    module: str = Header(..., alias="Module"),
    submodules: str = Header(
        ..., alias="Submodules", description="Comma-separated list of submodules"
    ),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Initiates bulk prediction tasks for specified submodules within a module for a given client and project.
    Args:
    ----
        module (str): The name of the module. Provided via HTTP header "Module".
        submodules (str): Comma-separated list of submodules. Provided via HTTP header "Submodules".
        client_id (str): The client identifier. Provided via HTTP header "clientId".
        project_id (str): The project identifier. Provided via HTTP header "projectId".

    Returns:
        APIResponse or BulkPredictionResponse: Response object containing the status, any errors, and task IDs for started predictions.

    Raises:
        Exception: If bulk prediction cannot be started due to missing instance, no submodules, or other errors.

    Notes:
        - Each submodule is checked for an active model before starting prediction.
        - Prediction tasks are started asynchronously for each valid submodule.
        - Errors for individual submodules are collected and returned in the response.
    """
    try:
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodules_list = []
        if submodules is not None:
            submodules_list = [s.strip() for s in submodules.split(",")]

        if not submodules_list:
            s = Status.FAILED(
                "No submodules provided",
            )
            return BulkPredictionResponse().assign_status(status=s)

        response = BulkPredictionResponse(
            module=module, client_id=client_id, project_id=project_id
        )

        # Iterate through each submodule and start prediction
        task_dict = {}
        errors = {}
        for sub in submodules_list:
            try:
                submodule_obj = Submodule(
                    module=module, submodule=sub, instance_id=instance.instance_id
                )

                # skip if no active model is set
                if not submodule_obj.active_model:
                    Status.NOT_FOUND(f"No active model for {sub}", submodule_obj)
                    continue

                task = predict_from_model.delay(
                    module, sub, instance.instance_id, risk_scoring=True
                )
                task_dict[sub] = task.id
            except Exception as e:
                errors[sub] = f"Predicted can not be started. {str(e)}"
                Status.FAILED(
                    f"Prediction failed for {sub}", error=str(e), traceback=False
                )

        # Check if any tasks were started
        if not task_dict:
            s = Status.NOT_FOUND("No active models found for the provided submodules")
            return response.assign_status(status=s, error=errors)

        s = Status.SUCCESS("Bulk prediction started")
        return response.assign_status(status=s, data=task_dict, error=errors)
    except Exception as e:
        s = Status.FAILED("Can not start bulk prediction", error=str(e))
        return APIResponse().assign_status(status=s, error=str(e))


@automl_router.post(
    "/predict", response_model=TaskResponse, dependencies=[Depends(validate_token)]
)
async def single_prediction(
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Handles a single prediction request for a specified module and submodule.
    This endpoint retrieves the appropriate instance based on client and project IDs,
    validates the existence of an active model, and initiates an asynchronous prediction task.
    Returns a response with the prediction task status and ID.
    Args:
    ----
        module (str): The name of the module, provided via HTTP header "Module".
        submodule (str): The name of the submodule, provided via HTTP header "Submodule".
        client_id (str): The client identifier, provided via HTTP header "clientId".
        project_id (str): The project identifier, provided via HTTP header "projectId".

    Returns:
        APIResponse: An object containing the status of the prediction request, including
        error details if applicable, or the task ID if prediction was successfully started.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = TaskResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        if not submodule_obj.active_model:
            s = Status.NOT_FOUND("No active model found", submodule_obj)
            return response.assign_status(status=s)

        task = predict_from_model.delay(
            module,
            submodule,
            instance.instance_id,
            risk_scoring=True,
        )
        s = Status.SUCCESS(
            "Prediction started", task_id=task.id, submodule=submodule_obj
        )
        return response.assign_status(status=s, task_id=task.id)
    except Exception as e:
        s = Status.FAILED("Can not start prediction", submodule_obj, error=str(e))
        return APIResponse().assign_status(status=s, error=str(e))


class ExperimentNamesReponse(APIResponse):
    """
    ExperimentNamesReponse is a response model that extends the APIResponse class.
    """

    data: Optional[List[str]] = Field(None, description="List of experiment names")
    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None


@automl_router.get(
    "/experiments",
    response_model=ExperimentNamesReponse,
    dependencies=[Depends(validate_token)],
)
async def list_experiments(
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Lists all experiments for a given module and submodule within a specified client and project.
    Args:
    ----
        module (str): The name of the module, provided via HTTP header "Module".
        submodule (str): The name of the submodule, provided via HTTP header "Submodule".
        client_id (str): The client identifier, provided via HTTP header "clientId".
        project_id (str): The project identifier, provided via HTTP header "projectId".

    Returns:
        APIResponse: An API response containing the status and a list of experiment names if successful,
        or an error message if the operation fails.

    Raises:
        BaseException: If any error occurs during the retrieval of experiments.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = ExperimentNamesReponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        experiments = [
            exp._name
            for exp in ModelTracker(submodule_obj=submodule_obj).ml_experiments
        ]
        s = Status.SUCCESS(
            "Experiment list", experiments=experiments, submodule=submodule_obj
        )
        return response.assign_status(status=s, data=experiments)
    except BaseException as e:
        s = Status.FAILED(
            "Experiment list failed", submodule=submodule_obj, error=str(e)
        )
        return APIResponse().assign_status(status=s, error=str(e))


@automl_router.get(
    "/experiments/model",
    response_model=ModelResponse,
    dependencies=[Depends(validate_token)],
)
async def get_experiment_model(
    model_name: str = Header(..., alias="modelName"),
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Retrieve details of a specific experiment model by name.
    This endpoint fetches the model associated with the given module, submodule, client, and project identifiers.
    It returns the model details if found, or an appropriate status message if not found or if an error occurs.
    Args:
    ----
        model_name (str): Name of the model to retrieve (from header "modelName").
        module (str): Name of the module (from header "Module").
        submodule (str): Name of the submodule (from header "Submodule").
        client_id (str): Identifier for the client (from header "clientId").
        project_id (str): Identifier for the project (from header "projectId").

    Returns:
        APIResponse: Response object containing status and model data if found, or error/status information otherwise.

    Raises:
        Exception: If there is an error retrieving model details.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = ModelResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        if model := ModelTracker(submodule_obj=submodule_obj).get_model_by_name(
            model_name
        ):
            s = Status.SUCCESS("Model found", submodule_obj, model=model.model_dump())
            return response.assign_status(status=s, data=model.model_dump())

        s = Status.NOT_FOUND("Model not found", submodule_obj)
        return response.assign_status(status=s)
    except Exception as e:
        s = Status.FAILED("Can not retrieve model details", submodule_obj, error=str(e))
        return APIResponse().assign_status(status=s, error=str(e))


@automl_router.post(
    "/experiments/delete",
    response_model=APIResponse,
    dependencies=[Depends(validate_token)],
)
async def delete_experiments(
    experiment_id: str = Header(..., alias="experimentId"),
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Deletes an experiment for a given client and project.
    This asynchronous endpoint deletes an experiment identified by `experiment_id` for the specified client, project, module, and submodule. It retrieves the instance based on `client_id` and `project_id`, constructs a submodule object, and attempts to delete the experiment using the ModelTracker. Returns a success response if deletion is successful, otherwise returns a failure response with error details.
    Args:
    ----
        experiment_id (str): The ID of the experiment to delete (from header 'experimentId').
        module (str): The module name (from header 'Module').
        submodule (str): The submodule name (from header 'Submodule').
        client_id (str): The client ID (from header 'clientId').
        project_id (str): The project ID (from header 'projectId').

    Returns:
        APIResponse: An API response object with the status of the deletion operation.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        # Delete experiment
        if ModelTracker(submodule_obj=submodule_obj).delete_experiment(experiment_id):
            s = Status.SUCCESS(
                "Experiment deleted successfully", submodule=submodule_obj
            )
            return APIResponse().assign_status(status=s)

        raise RuntimeError("Contact support")
    except Exception as e:
        s = Status.FAILED(
            "Experiment deletion failed", submodule=submodule_obj, error=str(e)
        )
        return APIResponse().assign_status(status=s, error=str(e))


class Questionnaire(BaseModel):
    """
    Questionnaire data class.
    Attributes:
    ----------
        alert_status (Optional[str]): Alert status of the questionnaire.
        concern_questionnaire (Optional[List[TemplateQuestion]]): List of questions for cases with concerns.
        no_concern_questionnaire (Optional[List[TemplateQuestion]]): List of questions for cases with no concerns.
        template_id (Optional[int]): Identifier for the questionnaire template.
    """

    alert_status: Optional[str] = Field(None, description="Alert status")
    concern_questionnaire: Optional[List[TemplateQuestion]] = Field(
        None, description="Concern questionnaire"
    )
    no_concern_questionnaire: Optional[List[TemplateQuestion]] = Field(
        None, description="No concern questionnaire"
    )
    template_id: Optional[int] = Field(None, description="Template ID")


class QuestionnaireResponse(APIResponse):
    """
    Represents the response for a questionnaire API endpoint.
    Attributes:
    ----------
        client_id (str): The unique identifier for the client.
        project_id (str): The unique identifier for the project.
        module (str): The module associated with the questionnaire.
        submodule (str): The submodule associated with the questionnaire.
        data (Optional[Questionnaire]): The details of the questionnaire, if available.
    """

    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None
    data: Optional[Questionnaire] = Field(None, description="Questionnaire details")


@automl_router.get(
    "/questionnaire",
    response_model=QuestionnaireResponse,
    dependencies=[Depends(validate_token)],
)
async def get_questionnaire_details(
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Retrieve questionnaire details for a specified module, submodule, client, and project.

    This endpoint expects the following headers:

        - Module: The name of the module.
        - Submodule: The name of the submodule.
        - clientId: The client identifier.
        - projectId: The project identifier.
    Returns:
        APIResponse: An object containing the questionnaire details, status, and any error information.

    Raises:
        Returns a failed status if the instance is not found or if any exception occurs during processing.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        sub = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = QuestionnaireResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )
        # Get questionnaire details
        questionnaire_details = sub.load_configuration(
            module=module, submodule=submodule, instance_id=instance.instance_id
        )
        data = Questionnaire(
            alert_status=questionnaire_details.get("alert_status", ""),
            concern_questionnaire=questionnaire_details.get(
                "concern_questionnaire", []
            ),
            no_concern_questionnaire=questionnaire_details.get(
                "no_concern_questionnaire", []
            ),
            template_id=questionnaire_details.get("template_id"),
        )

        s = Status.SUCCESS(
            "Questionnaire found",
            sub,
        )
        return response.assign_status(status=s, data=data)
    except BaseException as e:
        s = Status.FAILED(
            "Can not retrieve questionnaire details",
            sub,
            error=str(e),
        )
        return APIResponse().assign_status(status=s, error=str(e))


@automl_router.post(
    "/questionnaire",
    response_model=QuestionnaireResponse,
    dependencies=[Depends(validate_token)],
)
async def set_questionnaire_details(
    input_ques: Questionnaire = Body(...),
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Asynchronously sets questionnaire details for a given module, submodule, client, and project.
    Args:
    -----
        input_ques (Questionnaire): The questionnaire details to be set, provided in the request body.
        module (str): The module name, provided in the request header "Module".
        submodule (str): The submodule name, provided in the request header "Submodule".
        client_id (str): The client identifier, provided in the request header "clientId".
        project_id (str): The project identifier, provided in the request header "projectId".
    Returns:
        APIResponse: An API response object containing the status and data of the operation.

    Raises:
        RuntimeError: If saving the questionnaire configuration fails.

    Notes:
        - Returns a success status and the saved questionnaire data if the operation is successful.
        - Returns a not found status if the instance for the given client and project is not found.
        - Returns a failed status with error details if any exception occurs during the process.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        sub = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = QuestionnaireResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )
        ques: Questionnaire = Questionnaire(
            alert_status=input_ques.alert_status,
            concern_questionnaire=input_ques.concern_questionnaire,
            no_concern_questionnaire=input_ques.no_concern_questionnaire,
            template_id=input_ques.template_id,
        )
        sub.alert_status = ques.alert_status
        sub.concern_questionnaire = ques.concern_questionnaire
        sub.no_concern_questionnaire = ques.no_concern_questionnaire
        sub.template_id = ques.template_id

        # Save configuration
        if sub.save_configuration():
            s = Status.SUCCESS("Questionnaire saved", sub)
            return response.assign_status(status=s, data=ques)

        raise RuntimeError("Contact support")
    except Exception as e:
        s = Status.FAILED("Questionnaire save failed", sub, error=str(e))
        return APIResponse().assign_status(status=s, error=str(e))


class LastPredictionResponse(APIResponse):
    """
    Response model for the last prediction endpoint.
    Attributes:
    ----------
        client_id (str): Identifier for the client. Defaults to None.
        project_id (str): Identifier for the project. Defaults to None.
        module (str): Name of the module. Defaults to None.
        submodule (str): Name of the submodule. Defaults to None.
        data (Optional[LastPredictionDetails]): Details of the last prediction, if available.
    """

    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None
    data: Optional[LastPredictionDetails] = Field(
        None, description="Last prediction details"
    )


@automl_router.get(
    "/activemodel/lastprediction",
    response_model=LastPredictionResponse,
    dependencies=[Depends(validate_token)],
)
async def get_last_prediction(
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Retrieve the last prediction details for a given module, submodule, client, and project.

    This endpoint expects the following headers:

        - Module: The name of the module.
        - Submodule: The name of the submodule.
        - clientId: The client identifier.
        - projectId: The project identifier.
    Returns:

        APIResponse: An object containing the status and last prediction details if found.
            - If the instance is not found, returns a NOT_FOUND status.
            - If last prediction details are found, returns a SUCCESS status with details.
            - If last prediction details are not found, returns a NOT_FOUND status.
            - On exception, returns a FAILED status with error information.
    Raises:
        Exception: If an error occurs during retrieval.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = LastPredictionResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        lpd = LastPredictionDetails(submodule=submodule_obj)
        if lpd.Date:
            s = Status.SUCCESS("Last prediction details found", submodule_obj)
            return response.assign_status(status=s, data=lpd)

        s = Status.NOT_FOUND("Last prediction details not found", submodule_obj)
        return response.assign_status(status=s)
    except Exception as e:
        s = Status.FAILED(
            "Can not retrieve last prediction details",
            submodule_obj,
            error=str(e),
        )
        return APIResponse().assign_status(status=s, error=str(e))


class ExplanationResponse(APIResponse):
    """
    Response model for the explanation endpoint.
    Attributes:
    ----------
        client_id (str): Identifier for the client.
        project_id (str): Identifier for the project.
        module (str): Name of the module related to the explanation.
        submodule (str): Name of the submodule related to the explanation.
        data (Optional[ExplainationOutput]): Prediction explanation output.
    """

    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None
    data: Optional[ExplainationOutput] = Field(
        None, description="Prediction explanation"
    )


@automl_router.get(
    "/explainprediction",
    response_model=ExplanationResponse,
    dependencies=[Depends(validate_token)],
)
async def explain_prediction(
    transaction_id: str = Header(...),
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Asynchronously generates an explanation for a prediction based on provided headers.
    Args:
    -----
        transaction_id (str): Unique identifier for the transaction, passed via HTTP header.
        module (str): Name of the module, passed via HTTP header (alias "Module").
        submodule (str): Name of the submodule, passed via HTTP header (alias "Submodule").
        client_id (str): Identifier for the client, passed via HTTP header (alias "clientId").
        project_id (str): Identifier for the project, passed via HTTP header (alias "projectId").
    Returns:
        APIResponse: An API response object containing the explanation data and status.

    Raises:
        RuntimeError: If explanation generation fails.
        Exception: For any unexpected errors during processing.

    Notes:
        - Returns a NOT_FOUND status if the instance for the given client and project is not found.
        - Returns a FAILED status if any exception occurs during explanation generation.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = ExplanationResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        explainer = PredictionExplainer(
            submodule_obj=submodule_obj, transaction_id=transaction_id
        )

        if explanation := explainer.explain():
            s = Status.SUCCESS("Explanation generated", submodule_obj)
            return response.assign_status(status=s, data=explanation)

        raise RuntimeError("Contact support")
    except Exception as e:
        s = Status.FAILED(
            "Can not generate explanation failed", submodule_obj, error=str(e)
        )
        return APIResponse().assign_status(status=s, error=str(e))


class ModelMetricsDetails(BaseModel):
    """Model metrics data class."""

    model_name: str = Field(..., description="Model name")
    metrics: ModelMetrics = Field(None, description="Model metrics")


class ExperimentWithMetrics(BaseModel):
    """Experiment models with metrics."""

    experiment_name: str = Field(..., description="Experiment name")
    experiment_id: str = Field(..., description="Experiment ID")
    metrics: List[ModelMetricsDetails] = Field(None, description="Experiment metrics")


class ExperimentDetailsResponse(APIResponse):
    """Response model for experiment details endpoint."""

    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None
    data: Optional[ExperimentWithMetrics] = Field(
        None, description="Experiment details"
    )


@automl_router.get(
    "/experiments/{experiment_name}",
    response_model=ExperimentDetailsResponse,
    dependencies=[Depends(validate_token)],
)
async def get_experiment_details(
    experiment_name: str,
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """
    Retrieve details of a specific machine learning experiment, including associated models and their metrics.
    Args:
    ----
        experiment_name (str): The name of the experiment to retrieve details for.
        module (str): The module name, provided via HTTP header "Module".
        submodule (str): The submodule name, provided via HTTP header "Submodule".
        client_id (str): The client identifier, provided via HTTP header "clientId".
        project_id (str): The project identifier, provided via HTTP header "projectId".
    Returns:
        APIResponse: An API response containing the experiment details, including model metrics, or an appropriate status if not found or on error.

    Raises:
        Exception: If any error occurs during retrieval, returns a failed status with error details.

    Notes:
        - Requires valid client and project identifiers to locate the instance.
        - Returns NOT_FOUND status if the instance, experiment, or models are not found.
        - Returns SUCCESS status with experiment details if found.
    """
    try:
        # Get the instance ID
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            s = Status.NOT_FOUND(
                "Instance not found", client_id=client_id, project_id=project_id
            )
            return APIResponse().assign_status(status=s)

        submodule_obj = Submodule(
            module=module,
            submodule=submodule,
            instance_id=instance.instance_id,
        )

        response = ExperimentDetailsResponse(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        tracker = ModelTracker(submodule_obj=submodule_obj)
        experiment = next(
            (exp for exp in tracker.ml_experiments if exp._name == experiment_name),
            None,
        )

        if not experiment:
            s = Status.NOT_FOUND("Experiment not found", submodule_obj)
            return response.assign_status(status=s)

        # Get all models for the experiment and their metrics
        models = tracker.get_ml_models([experiment])

        output = ExperimentWithMetrics(
            experiment_name=experiment_name,
            experiment_id=experiment.experiment_id,
        )
        if not models:
            s = Status.NOT_FOUND(
                "No valid models trained",
                submodule=submodule_obj,
            )
            return response.assign_status(status=s, data=output)

        # Get all model metrics for the experiment
        output.metrics = [
            ModelMetricsDetails(
                model_name=m.name,
                metrics=m.metrics.model_dump() if m.metrics else None,
            )
            for m in models
        ]

        s = Status.SUCCESS(
            "Experiment details found",
            experiment_name=experiment_name,
            total_models=len(models),
            submodule=submodule_obj,
        )
        return response.assign_status(status=s, data=output)
    except Exception as e:
        s = Status.FAILED(
            "Can not retrieve experiment details", submodule_obj, error=str(e)
        )
        return APIResponse().assign_status(status=s, error=str(e))
