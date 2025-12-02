# # Copyright (C) KonaAI - All Rights Reserved
"""This script builds the server executable using PyInstaller."""
import os
import platform
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

cwd = Path(__file__).parent.resolve()
code_dir = Path(cwd, "code").resolve()
print(f"Code Directory: {code_dir}")

if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))


from code.src.utils.conf import config as project_config
from code.src.utils.conf import config_file_path
from PyInstaller.__main__ import run as pyinstaller_run


def main():
    """
    Main entry point for building the executable using PyInstaller.

    This function performs the following steps:
    1. Resolves the path to the 'server.spec' file located in the same directory as this script.
    2. Checks if the spec file exists; if not, prints an error message and exits.
    3. Attempts to download or locate the UPX binary required for executable compression.
    4. If UPX is found, runs PyInstaller with the '--clean' and '--upx-dir' options using the spec file.
    5. Handles and reports any exceptions that occur during the process, exiting with an error code if necessary.
    """
    # Update the project version in the config file
    if project_version := get_version():
        print(f"Building Intelligence Server version: {project_version}")
        if not set_version(project_version):
            print("Failed to set version in config file.")
            sys.exit(1)
    else:
        print("Building Intelligence Server, version information not found.")
        sys.exit(1)

    spec_file = Path(Path(__file__).parent, "server.spec").resolve()
    if not spec_file.exists():
        print(f"Spec file not found at: {spec_file}")
        sys.exit(1)

    # Check if the spec file is a valid PyInstaller spec file
    try:
        if upx_path := download_upx():
            print(f"UPX binary found at: {upx_path}")
            pyinstaller_run(["--clean", f"--upx-dir={upx_path}", str(spec_file)])
        else:
            raise FileNotFoundError("UPX binary not found")
    except Exception as e:
        print(f"Error running PyInstaller: {e}")
        sys.exit(1)


def get_version() -> Optional[str]:
    """
    Retrieves the version string from the 'pyproject.toml' file located in the same directory as this script.

    Returns:
        Optional[str]: The version string if found, "unknown" if the version is not specified, or None if an error occurs.

    Raises:
        FileNotFoundError: If 'pyproject.toml' does not exist in the expected location.
        ValueError: If there is an error decoding the TOML file.
    """
    script_path = Path(__file__).parent
    toml_file = Path(script_path, "pyproject.toml")
    if not toml_file.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {toml_file}")

    with open(toml_file, "rb") as f:
        try:
            pyproject = tomllib.load(f)
            return pyproject.get("project", {}).get("version", "unknown")
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Error decoding TOML file: {e}") from e

    return None


def set_version(version: str) -> bool:
    """
    Sets the project version in the configuration file.

    This function updates the specified configuration file by setting the version
    under the "PROJECT_INFO" section. If the section does not exist, it is created.
    The function writes the updated configuration back to the file.

    Args:
        version (str): The version string to set in the configuration file.

    Returns:
        bool: True if the version was set successfully, False otherwise.

    Raises:
        Exception: Prints an error message if any exception occurs during the process.
    """
    try:
        SECTION_NAME = "PROJECT_INFO"
        VERSION_KEY = "VERSION"

        # create a section in the config file if it doesn't exist
        if SECTION_NAME not in project_config.sections():
            project_config.add_section(SECTION_NAME)

        # set the version in the config file
        project_config.set(SECTION_NAME, VERSION_KEY, version)

        # write the changes back to the config file
        with open(config_file_path, "w", encoding="utf-8") as configfile:
            project_config.write(configfile)

        return True
    except Exception as e:
        print(f"Error setting version in config file: {e}")
        return False


def download_upx():
    """
    Downloads and extracts the UPX binary for Windows platforms.

    This function detects the current operating system, downloads the appropriate UPX release archive,
    extracts it to a temporary directory, and locates the UPX executable. The path to the UPX binary
    is returned if found.

    Raises:
        OSError: If the current platform is not supported.
        ValueError: If the downloaded archive format is unsupported.
        FileNotFoundError: If the UPX binary cannot be found after extraction.

    Returns:
        str: The file path to the extracted UPX binary.
    """
    system = platform.system().lower()

    if "windows" in system:
        url = "https://github.com/upx/upx/releases/download/v5.0.1/upx-5.0.1-win64.zip"
    else:
        raise OSError(f"Unsupported platform: {system}")

    # Download
    tmp_dir = Path(tempfile.gettempdir()) / "upx_temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / url.rsplit("/", maxsplit=1)[-1]

    print(f"Downloading UPX from {url}...")
    urllib.request.urlretrieve(url, archive_path)

    # Extract
    print("Extracting UPX...")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)  # nosec
    elif archive_path.suffix == ".xz":

        with tarfile.open(archive_path, "r:xz") as tar_ref:
            tar_ref.extractall(tmp_dir)  # nosec
    else:
        raise ValueError("Unsupported archive format")

    # Find UPX binary
    for file in tmp_dir.rglob("upx.exe" if "windows" in system else "upx"):
        if file.is_file():
            os.chmod(file, 0o755)  # nosec
            print(f"UPX binary found: {file}")
            return str(file)

    raise FileNotFoundError("UPX binary not found in extracted files")


if __name__ == "__main__":
    main()
