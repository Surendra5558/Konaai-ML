# # Copyright (C) KonaAI - All Rights Reserved
"""This module defines the Instance class, which represents an instance with various attributes and methods to retrieve instances."""
import json
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import joblib
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from src.utils.api_config import APIClient
from src.utils.api_config import EndPoint
from src.utils.conf import Setup
from src.utils.database_config import SQLDatabaseManager
from src.utils.jwt_model import JWTModel
from src.utils.llm_config import BaseLLMConfig
from src.utils.smtp_model import NotificationModel
from src.utils.status import Status


class InstanceSettings(BaseModel):
    """
    Represents the settings for an instance, encapsulating configuration for authentication, notifications, databases, API client, and callback endpoints.
    Attributes:
        jwt (Optional[JWTModel]): JWT configuration model for authentication.
        notification (Optional[NotificationModel]): Notification settings for the instance.
        masterdb (SQLDatabaseManager): Master database configuration.
        projectdb (SQLDatabaseManager): Project database configuration.
        application_api_client (APIClient): API client configuration.
        anomaly_callback_endpoint (Optional[EndPoint]): Anomaly callback endpoint configuration for the API. This endpoint is used to provide updates on the anomaly detection status back to the application.
        automl_callback_endpoint (Optional[EndPoint]): Automl callback endpoint configuration for the API. This endpoint is used to provide updates on the automl prediction status back to the application.
    """

    jwt: Optional[JWTModel] = Field(
        default=JWTModel(), description="JWT configuration model"
    )
    notification: Optional[NotificationModel] = Field(
        default=NotificationModel(), description="Notification settings"
    )
    masterdb: SQLDatabaseManager = Field(
        default=SQLDatabaseManager(), description="Master database configuration"
    )
    projectdb: SQLDatabaseManager = Field(
        default=SQLDatabaseManager(), description="Project database configuration"
    )
    application_api_client: APIClient = Field(
        default=APIClient(), description="API client configuration"
    )
    # Anomaly Callback Endpoint
    anomaly_callback_endpoint: Optional[EndPoint] = Field(
        default_factory=lambda: EndPoint(
            path="/api/Anomaly/AnomalyStatus", method="POST"
        ),
        description="Anomaly callback endpoint configuration for the API. This endpoint is used to provide updates on the anomaly detection status back to the application.",
    )
    # Automl Callback Endpoint
    automl_callback_endpoint: Optional[EndPoint] = Field(
        default_factory=lambda: EndPoint(
            path="/api/AutoML/AutoMLStatus", method="POST"
        ),
        description="Automl callback endpoint configuration for the API. This endpoint is used to provide updates on the automl prediction status back to the application.",
    )
    llm_config: Optional[BaseLLMConfig] = Field(
        default=None, description="LLM configuration for the instance"
    )


class Instance(BaseModel):
    """Represents an instance of a client project with associated settings and metadata.
    The `Instance` class encapsulates information about a specific client project instance,
    including unique identifiers, client and project names, creation date, and configurable
    settings. It provides methods to load, save instance settings from/to disk,
    ensuring persistence and backward compatibility with legacy settings formats."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if isinstance(v, datetime) else v,
        },
    )

    instance_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the instance",
    )
    client_name: str = Field(default=None, description="Name of the client")
    ClientUID: str = Field(default=None, description="Unique identifier for the client")
    project_name: str = Field(default=None, description="Name of the project")
    ProjectUID: str = Field(
        default=None, description="Unique identifier for the project"
    )
    created_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation date of the instance in UTC",
    )
    settings: InstanceSettings = Field(
        default=InstanceSettings(), description="Instance settings"
    )

    def __init__(self, **data):
        """
        Initialize an Instance object with the provided data.

        Args:
            **data: Arbitrary keyword arguments representing instance attributes.
        """
        super().__init__(**data)
        if not self.settings:
            self.settings = InstanceSettings()

        # Load settings from file if it exists
        self._load_settings()

    @property
    def _settings_file_path(self) -> Path:
        return Path(Setup().db_path, self.instance_id, "settings").absolute()

    def _load_settings(self) -> bool:
        """
        Load the instance settings from the settings file.

        This method reads the settings from the JSON file and updates the instance's
        settings accordingly.

        Returns:
            bool: True if the settings were loaded successfully, False otherwise.
        """
        try:
            # Migrate old settings structure if necessary
            self._migrate_instance_settings_structure()

            file_path = self._settings_file_path
            if not file_path.exists():
                return False

            with file_path.open("rb") as f:
                settings_data = joblib.load(f)
                data = json.loads(Setup().decrypt(settings_data))
                self.settings = InstanceSettings(**data)
            return True
        except Exception as e:
            Status.FAILED("Instance settings load failed", error=str(e))
            return False

    def save_settings(self) -> bool:
        """
        Save the current instance settings to the global settings.

        This method updates the global settings with the current instance's settings.

        Returns:
            bool: True if the settings were saved successfully, False otherwise.

        Raises:
            Exception: If an error occurs during the saving process, it is caught and logged.
        """
        try:
            file_path = self._settings_file_path

            file_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists
            with file_path.open("wb") as f:
                data = Setup().encrypt(self.settings.model_dump(mode="json"))
                joblib.dump(data, f)

            Status.SUCCESS("Instance settings saved successfully", self)
            return True
        except Exception as e:
            Status.FAILED("Instance settings save failed", error=str(e))
            return False

    def __repr__(self):
        return f"Instance ID: {self.instance_id}, Client: {self.client_name}, Project: {self.project_name}, Created: {self.created_date.isoformat()}"

    def __str__(self):
        return f"Instance ID: {self.instance_id}, Client: {self.client_name}, Project: {self.project_name}, Created: {self.created_date.isoformat()}"

    def _migrate_instance_settings_structure(self):
        """
        Migrates old-format settings.json to a simplified new format:
        If no legacy keys are found, returns the existing settings as-is.
        """

        try:
            file_path = Path(Setup().db_path, self.instance_id, "settings.json")
            if not file_path.exists():
                return

            content = {}
            with open(file_path, encoding="utf-8") as f:
                # Check if the file is empty or has no legacy keys
                content = f.read().strip()
                if content:
                    content = Setup().decrypt(content)

            settings: dict = json.loads(content)
            db = SQLDatabaseManager()
            db.Server = settings.get("pipeline", {}).get("pipeline_database_server", "")
            db.Database = settings.get("pipeline", {}).get("pipeline_database_name", "")
            db.Username = settings.get("pipeline", {}).get(
                "pipeline_database_username", ""
            )
            db.Password = settings.get("pipeline", {}).get(
                "pipeline_database_password", ""
            )

            self.settings.projectdb = db

            if self.save_settings():
                # Remove the old settings file after migration
                file_path.unlink(missing_ok=True)
                Status.SUCCESS("Instance settings migrated successfully")
        except Exception as e:
            Status.FAILED(f"Failed to migrate instance settings: {str(e)}")
