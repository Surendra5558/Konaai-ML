# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides all settings"""
import base64
import configparser
import json
import os
import pathlib
import secrets
from dataclasses import dataclass
from typing import Dict
from typing import Union

from cryptography.fernet import Fernet
from platformdirs import PlatformDirs
from src.utils.status import Status


config_file_path = os.path.join(os.path.dirname(__file__), "config.ini")
config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file_path, encoding="utf-8")

ENVIRONMENT_VAR_NAME = "INTELLIGENCE_PATH"


@dataclass
class Setup:
    """
    Setup
    This class handles the configuration and setup of directory paths, environment variables, and encryption utilities for the application.
    Attributes:
        db_path (str): Path to the database directory.
        log_path (str): Path to the log directory.
        temp_path (str): Path to the temporary files directory.
        assets_path (str): Path to the assets directory.
        root_path (str): Root path of the project.
        db_folder (str): Name of the database folder, loaded from config.
        log_folder (str): Name of the log folder, loaded from config.
        temp_folder (str): Name of the temporary folder, loaded from config.
        assets_folder (str): Name of the assets folder, loaded from config.
    """

    db_path: str = None
    log_path: str = None
    temp_path: str = None
    assets_path: str = None

    def __init__(
        self,
        db_folder=config.get("ASSETS", "DB_FOLDER"),
        log_folder=config.get("ASSETS", "LOG_FOLDER"),
        temp_folder=config.get("ASSETS", "TEMP_FOLDER"),
        assets_folder=config.get("ASSETS", "ASSETS_FOLDER"),
    ) -> None:
        """Constructor"""

        self.root_path: str = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..")
        )

        # check if environment variable is set
        env_var = os.getenv(ENVIRONMENT_VAR_NAME) or os.getenv(
            str(ENVIRONMENT_VAR_NAME).lower()
        )
        env_var = env_var.replace('"', "").replace("'", "").strip()
        if not env_var:
            Status.FAILED(f"Environment variable {ENVIRONMENT_VAR_NAME} is not set. ")
            return

        self.db_path: str = os.path.join(env_var, db_folder)
        self.log_path: str = os.path.join(env_var, log_folder)
        self.temp_path: str = os.path.join(env_var, temp_folder)
        self.assets_path: str = os.path.join(self.root_path, assets_folder)

        # set dask temp directory
        os.environ["DASK_TEMPORARY_DIRECTORY"] = self.temp_path

        # create folders if not exists
        self._create_directory(self.db_path)
        self._create_directory(self.log_path)
        self._create_directory(self.temp_path)

    def _create_directory(self, folder: str) -> None:
        """This function creates directory if not exists

        Args:
            folder (str): Folder path
        """
        folder = folder.replace('"', "").replace("'", "").strip()
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            # ensure the folder is writable by the app user/group
            try:
                # give owner read/write/execute and group read/write/execute
                os.chmod(folder, 0o770)  # nosec
            except Exception as _e:
                Status.FAILED(
                    "Unable to set permissions on folder",
                    folder=folder,
                    error=str(_e),
                )

    @property
    def global_constants(self) -> Dict:
        """This function provides the global config

        Returns:
            dict: global config
        """
        try:
            if config:
                # Convert the configuration to a dictionary
                config_dict = {}
                for section in config.sections():
                    config_dict[section] = {}
                    for option in config.options(section):
                        config_dict[section][option] = config.get(section, option)
                return config_dict

        except BaseException as _e:
            Status.FAILED("Error while loading global config", error=str(_e))
        return {}

    def _encryption_key(self):
        encryption_key_file = os.path.join(
            self.db_path,
            config.get("SETTINGS_ENCRYPTION", "FILE_NAME"),
        )

        key = None
        if os.path.exists(encryption_key_file):
            with open(encryption_key_file, "rb") as _f:
                return _f.read()
        else:
            key = base64.urlsafe_b64encode(secrets.token_bytes(32))
            with open(encryption_key_file, "wb") as _f:
                _f.write(key)
        return key

    def encrypt(self, data: Union[str, Dict]) -> str:
        """
        Encrypts the provided data using Fernet symmetric encryption.
        Args:
            data (Union[str, Dict]): The data to encrypt. Can be a string or a dictionary.
                If a dictionary is provided, it will be converted to a JSON string before encryption.
        Returns:
            str: The encrypted data as a UTF-8 encoded string.
        Raises:
            Exception: If encryption fails due to invalid key or data format.
        """
        # convert data to string
        if isinstance(data, Dict):
            data = json.dumps(data)

        _key = self._encryption_key()
        fernet = Fernet(_key)
        return fernet.encrypt(data.encode("utf-8")).decode("utf-8")

    def decrypt(self, data: str) -> str:
        """
        Decrypts the provided encrypted data string using the configured encryption key.

        Args:
            data (str): The encrypted data as a string.

        Returns:
            str: The decrypted data as a UTF-8 string. Returns an empty string if decryption fails.

        Raises:
            None: Any exceptions are caught internally and logged via Status.FAILED.
        """
        try:
            _key = self._encryption_key()
            fernet = Fernet(_key)
            return fernet.decrypt(data).decode("utf-8")
        except Exception as _e:
            Status.FAILED("Error while decrypting settings", error=str(_e))
            return ""

    @staticmethod
    def user_data_dir() -> pathlib.Path:
        """This function provides the user data directory"""
        dirs = PlatformDirs("KonaAIML")
        user_data_dir = pathlib.Path(dirs.user_data_dir).resolve()
        if not user_data_dir.exists():
            user_data_dir.mkdir(parents=True, exist_ok=True)

        return user_data_dir
