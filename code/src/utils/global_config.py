# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides global configuration management for the application"""
import json
import os
import shutil
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from src.utils.conf import Setup
from src.utils.database_config import SQLDatabaseManager
from src.utils.instance import Instance
from src.utils.status import Status


class BrokerModel(BaseModel):
    """
    BrokerModel represents broker configuration settings.
    Attributes:
        UserName (Optional[str]): Broker username.
        Password (Optional[SecretStr]): Broker password (stored securely).
        HostName (Optional[str]): Broker host name.
        VirtualHost (Optional[str]): Broker virtual host.
        Port (int): Broker port (default: 5672, range: 1-65535).
        ConsumerTimeoutHrs (int): Consumer timeout in hours (default: 36, range: 0-240).
        HeartbeatSeconds (int): Heartbeat interval in seconds (default: 0, range: 0-86400).
        SSL (bool): Enable SSL (default: False).
    Config:
        model_config (ConfigDict): Custom JSON encoder for SecretStr fields.
    """

    UserName: Optional[str] = Field(default="", description="Broker username")
    Password: Optional[SecretStr] = Field(None, description="Broker password")
    HostName: Optional[str] = Field(default="", description="Broker host name")
    VirtualHost: Optional[str] = Field(default="", description="Broker virtual host")
    Port: int = Field(default=5672, ge=1, le=65535, description="Broker port")
    ConsumerTimeoutHrs: int = Field(
        default=36, ge=0, le=240, description="Consumer timeout in hours"
    )
    HeartbeatSeconds: int = Field(
        default=0, ge=0, le=86400, description="Heartbeat interval in seconds"
    )
    SSL: bool = Field(False, description="Enable SSL")

    model_config = ConfigDict(
        json_encoders={
            SecretStr: lambda v: (
                v.get_secret_value() if isinstance(v, SecretStr) else v
            ),
        },
    )


class GlobalSettings(BaseModel):
    """
    A configuration management class.
    This class provides methods for loading, saving, migrating, and manipulating global settings, as well as managing project instances.
    Attributes:
        instances (List[Instance]): List of all configured instances.
        active_instance_id (Optional[str]): The ID of the currently active instance.
        jwt (Optional[JWTModel]): JWT authentication configuration.
        broker (Optional[BrokerModel]): Message broker configuration.
        notification (Optional[NotificationModel]): Notification/email settings.
        masterdb (Optional[SQLDatabaseManager]): Master database connection manager.

    Notes:
        - Handles encryption and decryption of settings for security.
        - Supports migration from legacy settings files.
        - Provides both instance and class methods for flexibility in usage.
        - Uses the Status class for logging operation results.
    """

    instances: List[Instance] = Field(
        default_factory=list, description="List of instances in the global settings"
    )
    active_instance_id: Optional[str] = Field(
        default=None, description="Active instance ID"
    )

    broker: Optional[BrokerModel] = Field(
        default=BrokerModel(), description="Broker URL for message queue"
    )
    workerdb: Optional[SQLDatabaseManager] = Field(
        default=SQLDatabaseManager(), description="Worker database connection manager"
    )

    model_config = ConfigDict(
        json_encoders={
            SecretStr: lambda v: (
                v.get_secret_value() if isinstance(v, SecretStr) else v
            ),
        },
        arbitrary_types_allowed=True,
    )

    def __init__(self, **kwargs):
        data = self._load_settings()
        if data:
            # merge args and kwargs with loaded data
            data.update(kwargs)
        else:
            data = kwargs

        # Initialize the model with the merged data
        super().__init__(**data)

        self.instances = [
            Instance(**instance) for instance in data.get("instances", [])
        ]
        self.active_instance_id = data.get("active_instance_id", None)
        self.broker = BrokerModel(**data.get("broker", {}))
        self.workerdb = SQLDatabaseManager(**data.get("workerdb", {}))

    @classmethod
    def _get_settings_file_path(cls) -> str:
        """
        Returns the full path to the encrypted global settings file.
        Uses Setup for path logic.
        """
        return os.path.join(
            Setup().db_path,
            Setup().global_constants.get("ASSETS", {}).get("GLOBAL_SETTINGS_FILE", ""),
        )

    @classmethod
    def _load_settings(cls) -> Dict:

        file_path = cls._get_settings_file_path()
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, encoding="utf-8") as f:
                encrypted = f.read()
            decrypted = Setup().decrypt(encrypted)
            return json.loads(decrypted) if decrypted else {}
        except Exception as e:
            Status.FAILED("Error while loading global settings", error=str(e))
            return {}

    def save_settings(self) -> bool:
        """
        Serializes and encrypts the global settings, then saves them to an encrypted JSON file.

        Returns:
            bool: True if the settings were successfully saved; False otherwise.

        Raises:
            Exception: If an error occurs during the save process, logs the error and returns False.
        """
        file_path = self._get_settings_file_path()
        try:
            return self._save_settings(file_path)
        except Exception as e:
            Status.FAILED("Error while saving global settings", error=str(e))
            return False

    def _save_settings(self, file_path):
        # exclude instance settings from serialization
        # This is done to avoid saving instance settings in a single file for security reasons
        # as instances may contain sensitive database connection details
        # and we want to keep them separate.
        new_instances = []
        for instance in self.instances:
            # create new Instance model without settings
            instance_copy = instance.model_copy(deep=True)
            del instance_copy.settings  # remove settings field
            new_instances.append(instance_copy)

        # Create a new model with the updated fields
        self.instances = new_instances

        data = self.model_dump_json(indent=2)
        encrypted = Setup().encrypt(data)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(encrypted)
        Status.SUCCESS("Global settings saved successfully.")
        return True

    def create_instance(
        self, client_name: str, client_uid: str, project_name: str, project_uid: str
    ) -> Optional["Instance"]:
        """
        Creates a new Instance object with the specified client and project details.
        This method checks for duplicate instances based on client and project UIDs.
        If a duplicate exists, it returns None and logs a failure status.
        Otherwise, it creates a new instance, sets up the corresponding folder,
        adds the instance to the global settings, saves the settings, and returns the instance.
        Args:
            client_name (str): The name of the client.
            client_uid (str): The unique identifier for the client.
            project_name (str): The name of the project.
            project_uid (str): The unique identifier for the project.
        Returns:
            Optional["Instance"]: The created Instance object if successful, otherwise None.
        """
        try:
            instance = Instance()
            instance.client_name = client_name
            instance.ClientUID = client_uid
            instance.project_name = project_name
            instance.ProjectUID = project_uid

            # check if duplicate instance exists
            if self.instance_by_client_project(client_uid, project_uid):
                Status.FAILED("Instance already exists for this client and project")
                return None

            # create instance folder
            instance_path = os.path.join(Setup().db_path, instance.instance_id)
            os.makedirs(instance_path, exist_ok=True)

            # add instance to global settings
            self.instances.append(instance)
            self.save_settings()

            Status.SUCCESS("Instance created successfully", instance)
            return instance
        except Exception as e:
            Status.FAILED("Instance creation failed", error=str(e))
            return None

    def set_active_instance(self, instance_id: str) -> bool:
        """
        Sets the specified instance as the active instance.

        Args:
            instance_id (str): The unique identifier of the instance to set as active.

        Returns:
            bool: True if the instance was set as active successfully, False otherwise.

        Side Effects:
            - Updates the `active_instance_id` attribute.
            - Persists the updated settings by calling `save_settings()`.
            - Logs the status of the operation using the `Status` class.
        """
        try:
            self.active_instance_id = instance_id
            self.save_settings()
            Status.SUCCESS(
                "Instance set as active successfully", instance_id=instance_id
            )
            return True
        except Exception as e:
            Status.FAILED("Instance activation failed", error=str(e))
            return False

    def delete_instance(self, instance_id: str) -> bool:
        """
        Deletes an instance with the specified instance_id from the instances list and removes its associated folder from the file system.
        Args:
            instance_id (str): The unique identifier of the instance to be deleted.
        Returns:
            bool: True if the instance was deleted successfully, False otherwise.
        Side Effects:
            - Updates the self.instances list by removing the specified instance.
            - Deletes the instance's folder from the file system if it exists.
            - Saves the updated settings.
            - Logs the status of the operation using the Status class.
        """
        try:
            self.instances = [
                instance
                for instance in self.instances
                if instance.instance_id != instance_id
            ]

            # delete instance folder
            instance_path = os.path.join(Setup().db_path, instance_id)
            if os.path.exists(instance_path):
                shutil.rmtree(instance_path, ignore_errors=True)

            self.save_settings()
            Status.SUCCESS("Instance deleted successfully", instance_id=instance_id)
            return True
        except Exception as e:
            Status.FAILED("Instance deletion failed", error=str(e))
            return False

    @classmethod
    def instance_by_id(cls, instance_id: str) -> Optional["Instance"]:
        """
        Retrieve an instance from the list of instances by its unique instance ID.

        Args:
            instance_id (str): The unique identifier of the instance to retrieve.

        Returns:
            Optional["Instance"]: The instance object with the matching instance_id if found, otherwise None.

        Raises:
            Exception: If an error occurs during the search, logs the error and returns None.
        """
        try:
            obj = cls()
            if not obj.instances:
                Status.WARNING("No instances found in global settings.")
                return None

            return next(
                (
                    instance
                    for instance in obj.instances
                    if instance.instance_id.lower() == instance_id.lower()
                ),
                None,
            )
        except Exception as e:
            Status.FAILED("Instance fetch failed", error=str(e))
            return None

    @classmethod
    def instance_by_client_project(
        cls, client_uid: str, project_uid: str
    ) -> Optional["Instance"]:
        """
        Retrieve an instance matching the specified client and project UIDs.

        Args:
            client_uid (str): The unique identifier for the client.
            project_uid (str): The unique identifier for the project.

        Returns:
            Optional["Instance"]: The first matching instance if found, otherwise None.

        Raises:
            Exception: If an error occurs during instance retrieval, logs the failure and returns None.
        """
        try:
            obj = cls()
            if not obj.instances:
                Status.WARNING("No instances found in global settings.")
                return None

            return next(
                (
                    instance
                    for instance in obj.instances
                    if instance.ClientUID.lower() == client_uid.lower()
                    and instance.ProjectUID.lower() == project_uid.lower()
                ),
                None,
            )
        except Exception as e:
            Status.FAILED("Instance fetch failed", error=str(e))
            return None
