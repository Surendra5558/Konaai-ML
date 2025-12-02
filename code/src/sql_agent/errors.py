# # Copyright (C) KonaAI - All Rights Reserved
"""Custom exceptions for chat handler module"""


class BaseError(Exception):
    """Base exception for chat handler custom errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message)
        self.message = message
        self.kwargs = kwargs

        # add additional context to the error message
        if kwargs:
            for key, value in kwargs.items():
                self.message += f" - {key}: {value}"


class HallucinationError(BaseError):
    """Custom exception for handling hallucination errors in data queries."""

    pass


class ConfigurationError(BaseError):
    """Custom exception for handling configuration errors."""

    pass


class NotAllowedError(BaseError):
    """Custom exception for handling not allowed errors."""

    pass
