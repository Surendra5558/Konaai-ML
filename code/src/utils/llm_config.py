# # Copyright (C) KonaAI - All Rights Reserved
"""LLM Configuration Module"""
from typing import Optional

import validators
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from src.insight_agent import constants


class BaseLLMConfig(BaseModel):
    """Base LLM Configuration Model"""

    llm_name: Optional[str] = Field(None, description="Name of the LLM to be used.")
    model_name: Optional[str] = Field(
        None, description="Specific model name or version of the LLM."
    )
    temperature: float = Field(0.7, description="Temperature setting for the LLM.")
    api_key: Optional[str] = Field(
        None,
        description="API key for accessing the LLM service.",
    )
    endpoint: Optional[str] = Field(
        None, description="Optional endpoint URL for the LLM service."
    )
    api_version: Optional[str] = Field(
        None, description="Optional API version for the LLM service."
    )

    # validate endpoint if it exists as a URL
    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v):
        """
        Validate that the provided endpoint is either None or a well-formed URL.

        Args:
            cls: The class this validator is bound to.
            v (str | None): The endpoint value to validate.

        Returns:
            str | None: The original endpoint value if it is None or a valid URL.

        Raises:
            ValueError: If v is not None and is not a valid URL.
        """
        if v is not None and not validators.url(v):
            raise ValueError("Endpoint must be a valid URL.")
        return v

    @field_validator("llm_name")
    @classmethod
    def validate_llm(cls, v):
        """
        Validate that the provided LLM name is one of the allowed models.

        Parameters
        ----------
        cls : type
            The class performing the validation (provided by the validator infrastructure).
        v : str
            The candidate LLM model name to validate.

        Returns
        -------
        str
            The validated LLM name (returned unchanged) when it is allowed.

        Raises
        ------
        ValueError
            If the provided name is not present in constants.LLM_MODELS.keys().
        """
        if v is None:
            return v
        allowed_names = constants.LLM_MODELS.keys()
        if v not in allowed_names:
            raise ValueError(
                f"Invalid LLM name: {v}. Allowed names are: {list(allowed_names)}"
            )
        return v
