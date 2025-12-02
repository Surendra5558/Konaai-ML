# # Copyright (C) KonaAI - All Rights Reserved
"""Enum class representing the various states of a Celery task."""
from enum import Enum


class CeleryStates(Enum):
    """
    Enum class representing the various states of a Celery task.
    """

    # define standard celery states
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    RETRY = "RETRY"
    IGNORED = "IGNORED"
