# # Copyright (C) KonaAI - All Rights Reserved
"""API routes for the chatbot conversation handling."""
from typing import Optional
from typing import Tuple

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Header
from src.butler.api_setup import validate_token
from src.sql_agent.context_evaluation import ContextEvaluation
from src.sql_agent.data_query import SQLQueryAgent
from src.utils.agent_models import AgentResponseModel
from src.utils.agent_models import FollowUpQuestion
from src.utils.agent_models import Message
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.metadata import Metadata
from src.utils.operators import ValueOperators
from src.utils.status import Status
from src.utils.submodule import Submodule

responder = AgentResponseModel()

sql_agent_router = APIRouter(tags=["SQL Agent"])


# create new api called sqlagent
@sql_agent_router.post(
    "/sqlagent",
    dependencies=[Depends(validate_token)],
    response_model=AgentResponseModel,
)
def sql_agent_conversation(  # pylint: disable=too-many-positional-arguments
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
    model: Optional[AgentResponseModel] = Body(
        None, description="Existing conversation object for continuing chat"
    ),
) -> AgentResponseModel:
    """API endpoint to handle SQL Agent conversations."""
    try:
        Status.INFO("Received request for SQL agent conversation")
        model = model or AgentResponseModel()
        instance = GlobalSettings.instance_by_client_project(client_id, project_id)
        if not instance:
            model.add_message(
                role="system",
                content="Instance not found for the provided clientId and projectId.",
            )
            return model

        # validate context
        needs_context, model = _needs_context(instance, model)
        if needs_context:
            Status.INFO("Asking for more context from user")
            return model

        # process the message and generate response
        model = SQLQueryAgent().execute(model)
        return model
    except Exception as e:
        Status.FAILED(
            "Error while responding to SQL agent chat", error=e, traceback=True
        )

        msg = Message(
            role="agent",
            content="Error while trying to process your request. Please try again later. If the issue persists, contact support.",
        )
        model.add_message(msg)
        return model


def _needs_context(
    instance: Instance, model: AgentResponseModel
) -> Tuple[bool, AgentResponseModel]:
    """Check if more context is needed from the user."""
    if model.context and not isinstance(model.context, Submodule):
        model.context = Submodule(**model.context)

    if isinstance(model.context, Submodule) and model.context.is_valid:
        # Context is already provided and valid
        return False, model

    if not isinstance(model.context, Submodule):
        sub = Submodule(instance_id=instance.instance_id)
    else:
        sub = model.context

    needs_module = True
    module_question = "Please specify the module you would like to query."
    if sub.module:
        needs_module = False
    elif model.follow_up_questions:
        # check if last follow up question was for module
        last_question = model.follow_up_questions[-1]
        if last_question.question == module_question:
            sub.module = last_question.answers or None
            model.context = sub
            needs_module = False

            # remove the follow up question since it has been answered
            model.remove_follow_up_questions(last_question)

    if needs_module:
        context_eval: ContextEvaluation = ContextEvaluation(instance)
        detected_context: Submodule = context_eval.evaluate_context(
            model.last_unsolicited_user_message()
        )
        if detected_context and detected_context.module:
            Status.INFO(
                f"Auto-detected module: {detected_context.module}",
                instance=instance,
            )
            sub.module = detected_context.module
            model.context = sub
        else:
            md = Metadata(instance.instance_id)
            modules = md.modules
            q = FollowUpQuestion(
                question=module_question,
                answer_options=modules,
                operator_options=[ValueOperators.EQUALS],
                multiple_selection=False,
                answer_operator=ValueOperators.EQUALS,
            )
            model.add_follow_up_question(q)
            return True, model

    # check for submodule
    needs_submodule = True
    submodule_question = (
        f"Please specify the transaction type for {sub.module} you would like to query."
    )
    if sub.submodule:
        needs_submodule = False
    elif model.follow_up_questions:
        # check if last follow up question was for submodule
        last_question = model.follow_up_questions[-1]
        if last_question.question == submodule_question:
            sub.submodule = last_question.answers or None
            model.context = sub
            needs_submodule = False

            # remove the follow up question since it has been answered
            model.remove_follow_up_questions(last_question)
    if needs_submodule:
        md = Metadata(instance.instance_id)
        submodules = md.get_submodule_names(sub.module)
        q = FollowUpQuestion(
            question=f"Please specify the transaction type for {sub.module} you would like to query.",
            answer_options=submodules,
            operator_options=[ValueOperators.EQUALS],
            answer_operator=ValueOperators.EQUALS,
            multiple_selection=False,
        )
        model.add_follow_up_question(q)
        return True, model

    # Ensure context is assigned as Submodule object before returning
    model.context = sub
    return False, model
