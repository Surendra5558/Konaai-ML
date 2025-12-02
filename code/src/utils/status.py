# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the status class"""
from datetime import date
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from src.utils.custom_logger import app_logger


class StatusType(Enum):
    """
    StatusType is an enumeration that defines standard HTTP-like status codes for application responses.
    Attributes:
    ----------
        SUCCESS (int): Indicates a successful operation (200).
        FAILED (int): Indicates a failed operation (500).
        INFO (int): Indicates an informational response (202).
        NOT_FOUND (int): Indicates that the requested resource was not found (404).
        INVALID_INPUT (int): Indicates that the input provided was invalid (422).
        UNAUTHORIZED (int): Indicates that authentication is required and has failed or not been provided (401).
        FORBIDDEN (int): Indicates that the request was valid, but the server is refusing action (403).
        CONFLICT (int): Indicates a conflict with the current state of the resource (409).
        BAD_REQUEST (int): Indicates that the server could not understand the request due to invalid syntax (400).
        NOT_IMPLEMENTED (int): Indicates that the server does not support the functionality required to fulfill the request (501).
        SERVICE_UNAVAILABLE (int): Indicates that the server is not ready to handle the request (503).
        TROUBLESHOOT (int): Indicates a status requiring troubleshooting (299).
        WARNING (int): Indicates a warning status (299).
    Methods:
        __str__(): Returns a human-readable string representation of the status type.
    """

    SUCCESS = 200
    FAILED = 500
    INFO = 202
    NOT_FOUND = 404
    INVALID_INPUT = 422
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    CONFLICT = 409
    BAD_REQUEST = 400
    NOT_IMPLEMENTED = 501
    SERVICE_UNAVAILABLE = 503
    TROUBLESHOOT = 299
    WARNING = 299
    PENDING = 102
    IN_PROGRESS = 102
    COMPLETED = 200

    def __str__(self):
        return self.name.replace("_", " ").title()


class Status:
    """
    Status class provides a standardized way to represent and log the status of operations within the application.
    Attributes:
    -----------
        status (StatusType): The status type (e.g., SUCCESS, FAILED, INFO, etc.).
        message (str): The main message describing the status.
        code (int): Numeric code representing the status.
        args (tuple): Additional positional arguments to be included in the status message.
        kwargs (dict): Additional keyword arguments to be included in the status message.
    """

    def __init__(self, status: StatusType, message: str, code: int, *args, **kwargs):
        self.status = status
        self.message = message
        self.code = code
        self.args = args
        self.kwargs = kwargs

        if self.status in (
            StatusType.SUCCESS,
            StatusType.INFO,
            StatusType.TROUBLESHOOT,
            StatusType.WARNING,
        ):
            self.__message = self.message
        else:
            self.__message = f"{self.status} - {self.message}"

        if self.__message and self.args:
            for arg in self.args:
                self.__message += f" - {str(arg)}"

        if self.__message and self.kwargs:
            for key, value in self.kwargs.items():
                if key == "traceback":
                    continue
                # add the key value pair to the message
                self.__message += f" - {key.replace('_', ' ').title()}: {str(value)}"  # pylint: disable=unused-private-member

    @property
    def log_file_path(self) -> str:
        """Returns the log file path"""
        return app_logger.log_file_path

    @classmethod
    def log_handlers(cls) -> list:
        """Returns the log handler"""
        return [
            app_logger.file_handler,
            app_logger.stream_handler,
        ]

    # define standard status codes
    @staticmethod
    def SUCCESS(message: str, *args, **kwargs):
        """Success"""
        s = Status(
            StatusType.SUCCESS, message, StatusType.SUCCESS.value, *args, **kwargs
        )
        app_logger.success(s.__message)
        return s

    @staticmethod
    def FAILED(message: str, *args, traceback: bool = False, **kwargs):
        """Failed"""
        s = Status(StatusType.FAILED, message, StatusType.FAILED.value, *args, **kwargs)
        app_logger.error(s.__message, traceback=traceback)
        return s

    @staticmethod
    def INFO(message: str, *args, **kwargs):
        """Info"""
        s = Status(StatusType.INFO, message, StatusType.INFO.value, *args, **kwargs)
        app_logger.info(s.__message)
        return s

    @staticmethod
    def COMPLETED(message: str, *args, **kwargs):
        """Completed"""
        s = Status(
            StatusType.COMPLETED,
            message,
            StatusType.COMPLETED.value,
            *args,
            **kwargs,
        )
        app_logger.info(s.__message)
        return s

    @staticmethod
    def NOT_FOUND(message: str, *args, **kwargs):
        """Not Found"""
        s = Status(
            StatusType.NOT_FOUND, message, StatusType.NOT_FOUND.value, *args, **kwargs
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def WARNING(message: str, *args, **kwargs):
        """Warning"""
        s = Status(
            StatusType.WARNING, message, StatusType.WARNING.value, *args, **kwargs
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def INVALID_INPUT(message: str, *args, **kwargs):
        """Invalid Input"""
        s = Status(
            StatusType.INVALID_INPUT,
            message,
            StatusType.INVALID_INPUT.value,
            *args,
            **kwargs,
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def UNAUTHORIZED(message: str, *args, **kwargs):
        """Unauthorized"""
        s = Status(
            StatusType.UNAUTHORIZED,
            message,
            StatusType.UNAUTHORIZED.value,
            *args,
            **kwargs,
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def FORBIDDEN(message: str, *args, **kwargs):
        """Forbidden"""
        s = Status(
            StatusType.FORBIDDEN, message, StatusType.FORBIDDEN.value, *args, **kwargs
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def CONFLICT(message: str, *args, **kwargs):
        """Conflict"""
        s = Status(
            StatusType.CONFLICT, message, StatusType.CONFLICT.value, *args, **kwargs
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def BAD_REQUEST(message: str, *args, **kwargs):
        """Bad Request"""
        s = Status(
            StatusType.BAD_REQUEST,
            message,
            StatusType.BAD_REQUEST.value,
            *args,
            **kwargs,
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def NOT_IMPLEMENTED(message: str, *args, **kwargs):
        """Not Implemented"""
        s = Status(
            StatusType.NOT_IMPLEMENTED,
            message,
            StatusType.NOT_IMPLEMENTED.value,
            *args,
            **kwargs,
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def SERVICE_UNAVAILABLE(message: str, *args, **kwargs):
        """Service Unavailable"""
        s = Status(
            StatusType.SERVICE_UNAVAILABLE,
            message,
            StatusType.SERVICE_UNAVAILABLE.value,
            *args,
            **kwargs,
        )
        app_logger.warning(s.__message)
        return s

    @staticmethod
    def TROUBLESHOOT(message: str, *args, **kwargs):
        """Troubleshoot"""
        s = Status(
            StatusType.TROUBLESHOOT,
            message,
            StatusType.TROUBLESHOOT.value,
            *args,
            **kwargs,
        )
        app_logger.exception(s.__message)
        return s

    def task_status(self):
        """Returns the status in a dictionary format"""
        return self.to_dict()

    def _serialize(self, value):  # pylint: disable=too-many-return-statements
        """Serialize the value"""
        if value is None:
            return None
        if isinstance(value, Enum):
            return str(value)
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(v) for v in value]
        if isinstance(value, datetime):
            return value.strftime("%A, %d %B %Y %I:%M %p %Z")
        if isinstance(value, date):
            return value.strftime("%A, %d %B %Y")
        if issubclass(type(value), BaseModel):
            return self._serialize(value.model_dump())

        return str(value)

    def to_dict(self):
        """Returns the status in a dictionary format"""
        kwargs = self._serialize(self.kwargs)

        return {
            "status": self.status.value,
            "message": self.message,
            "status_code": self.code,
            **kwargs,
        }

    def __str__(self):
        # convert dictionary to string
        return str(self.to_dict())
