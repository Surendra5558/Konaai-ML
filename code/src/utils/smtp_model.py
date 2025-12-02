# # Copyright (C) KonaAI - All Rights Reserved
"""This module defines the NotificationModel for managing SMTP settings and recipient emails."""
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr


class NotificationModel(BaseModel):
    """
    NotificationModel defines notification settings for the application.
    Attributes:
    ----------
        SMTPServer (Optional[str]): SMTP server address.
        SMTPPort (Optional[int]): SMTP port, defaults to 587.
        SMTPUsername (Optional[str]): SMTP username.
        SMTPPassword (Optional[SecretStr]): SMTP password, stored securely.
        RecipientEmails (List[str]): List of recipient email addresses.
        FromEmail (Optional[str]): Email address used as the sender.
    Config:
        model_config (ConfigDict): Custom JSON encoder for SecretStr to ensure
            secure serialization of sensitive data.
    """

    SMTPServer: Optional[str] = Field(default=None, description="SMTP server address")
    SMTPPort: Optional[int] = Field(default=587, description="SMTP port")
    SMTPUsername: Optional[str] = Field(None, description="SMTP username")
    SMTPPassword: Optional[SecretStr] = Field(default=None, description="SMTP password")
    RecipientEmails: List[str] = Field(
        default_factory=list, description="List of recipient emails"
    )
    FromEmail: Optional[str] = Field(None, description="From email address")

    model_config = ConfigDict(
        json_encoders={
            SecretStr: lambda v: (
                v.get_secret_value() if isinstance(v, SecretStr) else v
            ),
        },
    )
