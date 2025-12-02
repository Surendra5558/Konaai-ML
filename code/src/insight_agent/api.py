# # Copyright (C) KonaAI - All Rights Reserved
"""Audit Agent API Module"""
from typing import Literal
from typing import Optional

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Header
from pydantic import ValidationError
from src.butler.api_setup import validate_token
from src.insight_agent.transaction_agent import TransactionAgent
from src.utils.api_response import APIResponse
from src.utils.global_config import GlobalSettings
from src.utils.status import Status
from src.utils.submodule import Submodule

insight_agent_router = APIRouter(tags=["Insight Agent"])


class AuditResponseModel(APIResponse):
    """Audit Response Model extending APIResponse"""

    data: Optional[str] = None
    client_id: str = None
    project_id: str = None
    module: str = None
    submodule: str = None


@insight_agent_router.post(
    "/transaction",
    response_model=AuditResponseModel,
    dependencies=[Depends(validate_token)],
)
def transaction_summary(  # pylint: disable=too-many-positional-arguments
    transaction_id: str = Body(...),
    report_type: Literal["summary", "full"] = Body("summary"),
    module: str = Header(..., alias="Module"),
    submodule: str = Header(..., alias="Submodule"),
    client_id: str = Header(..., alias="clientId"),
    project_id: str = Header(..., alias="projectId"),
):
    """Generate transaction summary using LLM and audit patterns"""
    submodule_obj = None
    try:
        instance = GlobalSettings.instance_by_client_project(
            client_uid=client_id, project_uid=project_id
        )
        if not instance:
            raise ValidationError("Instance not found")

        submodule_obj = Submodule(
            module=module, submodule=submodule, instance_id=instance.instance_id
        )

        response = AuditResponseModel(
            module=module,
            submodule=submodule,
            client_id=client_id,
            project_id=project_id,
        )

        ta = TransactionAgent(submodule=submodule_obj, transaction_id=transaction_id)

        if report_type == "full":
            summary = ta.generate_full_audit_report()
        elif report_type == "summary":
            summary = ta.generate_summary_report()
        else:
            raise ValidationError("Invalid report type specified")

        s = Status.SUCCESS(
            "Transaction audit summary generated successfully", submodule_obj
        )
        return response.assign_status(status=s, data=summary)
    except Exception as e:
        s = Status.FAILED(
            "Transaction audit summary generation failed. Contact support.",
            submodule_obj,
            error=str(e),
        )
        return APIResponse().assign_status(status=s, error=str(e))
