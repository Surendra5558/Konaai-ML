# # Copyright (C) KonaAI - All Rights Reserved
"""Provides common file handling activities"""
import os
import ssl
import urllib
import uuid
from pathlib import Path
from typing import Optional
from typing import Tuple
from urllib.parse import quote
from urllib.parse import unquote

from src.utils.conf import Setup
from src.utils.status import Status


class FileHandler:
    """FileHandler Class"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Creates and returns a single instance of the class (singleton pattern).

        This method ensures that only one instance of the class exists throughout the application's lifecycle.
        If an instance does not already exist, it creates one; otherwise, it returns the existing instance.

        Args:
        -----
            *args: Variable length argument list passed to the superclass __new__ method.
            **kwargs: Arbitrary keyword arguments passed to the superclass __new__ method.

        Returns:
            The singleton instance of the class.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def _is_http_url(self, url: str) -> bool:
        """Determines if the given URL is a web URL.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a web URL, False otherwise.
        """

        return bool(url.startswith("http://") or url.startswith("https://"))

    def download_file_from_web(self, url: str, file_path: str) -> str:
        """
        Downloads a file from the specified URL and saves it to the given file path.
        Args:
            url (str): The URL of the file to be downloaded.
            file_path (str): The local file path where the downloaded file will be saved.
        Returns:
            str: The file path where the file was saved, or None if the download failed.
        Raises:
            Exception: Propagates any exception encountered during the download process.
        """
        try:
            self._download_file_from_web(url, file_path)
        except BaseException as _e:
            file_path = None
            Status.FAILED("Can not download file from url", error=_e)
        return file_path

    def _download_file_from_web(self, uri, file_path):
        """Downloads the file from web URL and saves it in local system directory."""
        Status.INFO("Downloading file from URL", url=uri)

        # Create a context with SSL verification disabled
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        # Open the URL and download the file
        req = urllib.request.Request(uri)
        with urllib.request.urlopen(req, context=context) as response:
            file_data = response.read()

        # Save the file to disk
        if response.getcode() == 200:
            with open(file_path, "wb") as _f:
                _f.write(file_data)

    def download_file(self, uri: str, extension: str) -> str:
        """
        Downloads a file from the specified URI and saves it to the local system directory.
        Args:
        -----
            uri (str): The URL of the file to download.
            extension (str): The file extension to use when saving the file. If None, the extension is inferred from the URI.

        Returns:
            str: The local file path where the downloaded file is saved.

        Raises:
            ValueError: If the file extension is not provided or the URI is not a supported web URL.
            FileNotFoundError: If the file download fails.

        Logs:
            Various status messages are logged during the download process, including success and failure states.
        """
        file_path = None
        try:
            Status.INFO("Downloading file from URL", url=uri)

            if extension is None:
                extension = Path(uri).suffix

            # ensure that the URL is properly encoded
            uri = quote(unquote(uri), safe=r"\:/?&=")

            if not extension:
                raise ValueError("File extension is not provided")

            _, file_path = self.get_new_file_name(extension)
            Status.INFO("Downloading file to local path", path=file_path)

            if self._is_http_url(uri):
                Status.NOT_FOUND("File path is web url. Trying default web download")
                file_path = self.download_file_from_web(uri, file_path)
            else:
                raise ValueError("File path is not a web url or not supported")

            if file_path:
                Status.SUCCESS("File downloaded successfully", path=file_path)
            else:
                raise FileNotFoundError("File download failed")
        except BaseException as _e:
            Status.FAILED("Can not download file", error=_e)
        return file_path

    def get_new_file_name(
        self, file_extension: Optional[str] = None, file_name: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generates a new file name and file path.
        Args:
        -----
            file_extension (Optional[str]): The extension of the file. Defaults to None.
            file_name (Optional[str]): The name of the file. Defaults to None.
        Returns:
            Tuple[str, str]: A tuple containing the new file name and file path.
        """
        file_path = None
        try:
            # generate new file name
            if not file_extension.startswith("."):
                file_extension = f".{file_extension}"

            if file_name:
                if not file_name.endswith(file_extension):
                    file_name = file_name + file_extension
            else:
                file_name = "".join([uuid.uuid4().hex, file_extension])

            # generate new file path
            file_path = os.path.join(Setup().temp_path, file_name)

        except BaseException as _e:
            Status.FAILED("Can not generate new file name", error=_e)
        return file_name, file_path

    # Below function creates a new directory
    def get_new_directory(self) -> Tuple[str, str]:
        """
        Generates a new directory with a unique name.
        This method creates a new directory with a unique name generated using UUID.
        The directory is created in the temporary path specified in the Setup().

        Returns:
            tuple: A tuple containing the directory name and the directory path.

        Raises:
            BaseException: If the directory cannot be created, an error message is logged.
        """

        directory_name = None
        directory_path = None

        try:
            directory_name = uuid.uuid4().hex
            directory_path = os.path.join(Setup().temp_path, directory_name)
            os.mkdir(directory_path)
        except BaseException as _e:
            Status.FAILED("Can not generate new directory", error=_e)
        return directory_name, directory_path


file_handler = FileHandler()
