# # Copyright (C) KonaAI - All Rights Reserved
"""LLM Factory Module"""
from typing import Union

from anthropic import Anthropic
from langchain_community.llms.cohere import Cohere
from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from src.insight_agent import constants
from src.utils.llm_config import BaseLLMConfig
from src.utils.status import Status


# Claude API integration using anthropic
class ClaudeLLM:
    """Claude LLM Client Wrapper"""

    def __init__(
        self,
        api_key,
        model: str,
        temperature: float = 0.7,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt, max_tokens=1024):
        """Generate response from Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    def invoke(self, prompt, max_tokens=1024):
        """Invoke the Claude model with a prompt."""
        return self.generate(prompt, max_tokens=max_tokens)


def get_llm(
    llm_config: BaseLLMConfig, max_tokens: int = 4000
) -> Union[BaseLanguageModel, ClaudeLLM]:
    """
    Create and return a configured language model instance based on the provided LLM configuration.
    This function inspects llm_config.llm_name (falling back to a default name if not provided),
    normalizes it to uppercase, and initializes the corresponding LLM client. Supported providers
    include: "AZURE", "OPENAI", "COHERE", "GEMINI", and "CLAUDE". For each provider the function
    validates that required configuration fields are present and then constructs the appropriate
    LLM object with parameters such as model name, API key, temperature, and token limits.
    Parameters:
        llm_config (BaseLLMConfig):
            Configuration object containing provider-specific settings such as:
            - llm_name / model identifier (which provider to use)
            - api_key
            - endpoint (Azure)
            - api_version (Azure)
            - model_name / deployment name
            - temperature
            The exact attributes required depend on the selected provider.
        max_tokens (int, optional):
            Maximum number of tokens (output) to allow for the model. If falsy, the function
            uses a default of 4000. Individual client constructors may accept provider-specific
            token parameter names (e.g., max_tokens, max_output_tokens).
    Returns:
        Union[BaseLanguageModel, ClaudeLLM]:
            An initialized language model client instance appropriate for the requested provider.
    Raises:
        ValueError:
            - If the selected model/provider is not supported.
            - If required configuration fields for the selected provider are missing
              (for example: api_key and model_name for OpenAI/Cohere/Gemini,
              or api_key/endpoint/api_version for Azure).
        Exception:
            Any exception raised during client construction is logged and re-raised.
    Side effects:
        - Logs informational messages using the Status logger (Status.INFO) about the
          initialization steps and the chosen model/temperature.
        - On error, logs a failure via Status.FAILED before re-raising the exception.
    Notes:
        - Azure initialization expects azure_deployment (llm_config.model_name) and uses
          explicit endpoint and api_version values.
        - Some providers use different argument names and/or timeouts; this function maps
          llm_config fields into the provider-specific constructor arguments.
        - Temperature from llm_config is passed to the provider client to control randomness.
    """
    Status.INFO(
        f"LLM Call for: {llm_config.llm_name}, temperature: {llm_config.temperature}"
    )
    llm: BaseLanguageModel
    model_name = (llm_config.llm_name or constants.DEFAULT_LLM_NAME).upper()
    max_tokens = max_tokens or 4000

    try:
        if model_name not in constants.LLM_MODELS:
            raise ValueError(f"Unsupported LLM model name: {model_name}")

        if model_name == "AZURE":
            Status.INFO(
                "Initializing Azure OpenAI with GPT-4o-mini model",
                temperature=llm_config.temperature,
            )

            if not all(
                [
                    llm_config.api_key,
                    llm_config.endpoint,
                    llm_config.api_version,
                    llm_config.model_name,
                ]
            ):
                raise ValueError(
                    "Azure LLM configuration requires api_key, endpoint, api_version, and model_name"
                )

            llm = AzureChatOpenAI(
                api_key=llm_config.api_key,
                azure_endpoint=llm_config.endpoint,
                api_version=llm_config.api_version,
                azure_deployment=llm_config.model_name,
                temperature=llm_config.temperature,
                max_tokens=max_tokens,
                timeout=60,
            )

        elif model_name == "OPENAI":
            Status.INFO(
                "Initializing OpenAI with GPT-4o-mini model",
                temperature=llm_config.temperature,
            )

            if not all([llm_config.api_key, llm_config.model_name]):
                raise ValueError(
                    "OpenAI LLM configuration requires api_key and model_name"
                )

            llm = ChatOpenAI(
                api_key=llm_config.api_key,
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                max_tokens=max_tokens,
                timeout=60,
            )

        elif model_name == "COHERE":
            Status.INFO("Initializing Cohere LLM", temperature=llm_config.temperature)

            if not all([llm_config.api_key, llm_config.model_name]):
                raise ValueError(
                    "Cohere LLM configuration requires api_key and model_name"
                )

            llm = Cohere(
                model=llm_config.model_name,
                cohere_api_key=llm_config.api_key,
                max_tokens=max_tokens,
                temperature=llm_config.temperature,
            )

        elif model_name == "GEMINI":
            Status.INFO(
                "Initializing Google Gemini LLM", temperature=llm_config.temperature
            )

            if not all([llm_config.api_key, llm_config.model_name]):
                raise ValueError(
                    "Google Gemini LLM configuration requires api_key and model_name"
                )

            llm = ChatGoogleGenerativeAI(
                google_api_key=llm_config.api_key,
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                max_output_tokens=max_tokens,
            )

        elif model_name == "CLAUDE":
            Status.INFO("Initializing Claude LLM", temperature=llm_config.temperature)

            if not llm_config.api_key:
                raise ValueError("Claude LLM configuration requires api_key")

            llm = ClaudeLLM(
                api_key=llm_config.api_key,
                model=llm_config.model_name,
                temperature=llm_config.temperature,
            )

        else:
            raise ValueError(f"Unsupported LLM model name: {model_name}")

        Status.INFO(
            f"Successfully initialized LLM: {model_name} with temperature: {llm_config.temperature}"
        )
        return llm

    except Exception as e:
        Status.FAILED(f"Error initializing {model_name} LLM: {str(e)}")
        raise e
