# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the APIResponse class"""
from datetime import date
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Optional
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from src.utils.status import Status
from src.utils.status import StatusType


# Add the APIResponse model
class APIResponse(BaseModel):
    """APIResponse is a utility class for standardizing API responses. It extends the BaseModel
    and provides attributes and methods to handle response data, status, and serialization.
    Attributes:
        status (StatusType): Represents the status of the response.
        status_code (int): HTTP status code associated with the response.
        message (str): A message providing additional context about the response.
        error (Optional[str]): An optional error message if the response indicates a failure.
        data (Optional[Any]): The data payload of the response.
        model_config (ConfigDict): Configuration for the model, including JSON encoders for
            datetime and date objects.
    Methods:
        assign_status(status: Status, data: Any = None, error: str = None, **kwargs) -> Self:
        _serialize(value: Any):
            Serializes the given value into a format suitable for JSON encoding.
                value (Any): The value to be serialized.
                Any: The serialized value. Handles various types including None, Enum, dict,
                list, tuple, set, datetime, date, and BaseModel."""

    status: StatusType = None
    status_code: int = None
    message: str = None
    error: Optional[str] = None
    data: Optional[Any] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else None,
            date: lambda v: v.isoformat() if isinstance(v, date) else None,
        },
        use_enum_values=True,
    )

    def assign_status(
        self, status: Status, data: Any = None, error: str = None, **kwargs
    ) -> Self:
        """
        Assigns status, data, error, and additional attributes to an instance of the class.
        Args:
            status (Status): An instance of the `Status` class containing status information.
            data (Any, optional): The data to be assigned to the instance. Defaults to None.
            error (str, optional): An error message to be assigned to the instance. Defaults to None.
            **kwargs: Additional attributes to be set on the instance.
        Returns:
            Self: An instance of the class with the assigned attributes.
        """
        self.status = status.status
        self.status_code = status.code
        self.message = status.message
        self.error = error
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def _serialize(self, value: Any):  # pylint: disable=too-many-return-statements
        """Serialize the value"""
        if value is None:
            return None
        if isinstance(value, Enum):
            return str(value)
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize(v) for v in value]
        if isinstance(value, (datetime, date)):  # Combined check
            return value.isoformat()
        if isinstance(value, BaseModel):
            return self._serialize(value.model_dump())

        return str(value)
