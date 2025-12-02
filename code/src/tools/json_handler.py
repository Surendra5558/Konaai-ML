# # Copyright (C) KonaAI - All Rights Reserved
"""provide functions to handle json files"""
import json
import pathlib
from typing import Dict
from typing import Union

from src.utils.status import Status


def is_valid_json(value: Union[str, Dict]) -> bool:
    """
    Check if the provided value is a valid JSON string or a dictionary.

    Args:
    ----
        value (Union[str, Dict]): The value to check. Can be a JSON-formatted string or a dictionary.

    Returns:
        bool: True if the value is a valid JSON string or a dictionary, False otherwise.
    """
    if isinstance(value, str):
        try:
            json.loads(value)
            return True
        except json.JSONDecodeError:
            return False
    elif isinstance(value, dict):
        return True
    return False


# When we are dumping a dictionary to a json file, some of the keys, or values may have double quotes in them.
# Downstream angular code is not able to parse the json file properly and throws an error.
# This function will remove double quotes from the json file and replace them with single quotes.
def remove_double_quotes_from_json_file(file_path: str) -> None:
    """
    Removes double quotes from all string values within a JSON file and replaces them with single quotes.
    This function reads a JSON file, recursively replaces all double quotes (") in string values with single quotes ('),
    and writes the modified data back to the same file.
    Args:
    -----
        file_path (str): The path to the JSON file to be processed.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        Any exception encountered during file reading, writing, or JSON parsing is caught and reported via Status.FAILED.
    """
    try:
        file_path = pathlib.Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        data = {}
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)

        def replace_double_quotes(value):
            # go to each value and replace double quotes with single quotes
            if isinstance(value, str):
                return value.replace('"', "'")

            if isinstance(value, dict):
                return {k: replace_double_quotes(v) for k, v in value.items()}

            if isinstance(value, list):
                return [replace_double_quotes(v) for v in value]

            return value

        # go to each value and replace double quotes with single quotes
        for key in data:
            data[key] = replace_double_quotes(data[key])

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except BaseException as _e:
        Status.FAILED(
            "Error while removing double quotes from {file_path}", error=str(_e)
        )
