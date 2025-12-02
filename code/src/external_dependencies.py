# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains functions to setup external dependencies."""
import logging
import pathlib
import warnings

from src.utils.conf import Setup

# ignore warnings
warnings.filterwarnings("ignore")


def set_logging_level():
    """
    Sets the logging level of all active loggers to ERROR, except for the 'KonaAIML' logger.

    This function iterates through all loggers currently managed by the logging module.
    It sets the logging level to ERROR for each logger, except for the logger named 'KonaAIML',
    which is left unchanged. This is useful for suppressing less severe log messages from
    external dependencies while retaining detailed logs from the main application logger.
    """
    # List all active loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name != "KonaAIML":
            # set all loggers to error level except KonaAIML
            logging.getLogger(logger_name).setLevel(logging.ERROR)


def setup_dependencies():
    """
    Sets up all required external dependencies for the application.
    This function performs the following tasks:
    1. Checks if the temporary folder specified by `Setup().temp_path` exists; if not, it creates the folder (including any necessary parent directories).
    2. Sets the logging level to error by calling `set_logging_level()`.
    Note:
    - Assumes that the `Setup` class and `set_logging_level` function are defined elsewhere in the codebase.
    - Uses the `pathlib` module for filesystem operations.
    """
    # if temp folder is not created, create it
    if not pathlib.Path(Setup().temp_path).exists():
        pathlib.Path(Setup().temp_path).mkdir(parents=True, exist_ok=True)

    # set logging level to error
    set_logging_level()


if __name__ == "__main__":
    setup_dependencies()
