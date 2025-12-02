# # Copyright (C) KonaAI - All Rights Reserved
"""provide general text processing functions"""
import os


# Application front end forbids certain characters due to security and other reasons..
# This function will remove forbidden characters from the text file and replace them with given character.
def remove_forbidden_characters(file_path: str):
    """
    Removes forbidden characters from the specified file by replacing them with allowed alternatives.
    This function reads the contents of a file, replaces all occurrences of the forbidden characters
    ('<' with '(', and '>' with ')'), and writes the modified content back to the same file.
    Args:
    ----
        file_path (str): The path to the file to be processed.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    forbidden_characters = {
        "<": "(",
        ">": ")",
    }

    data = None
    with open(file_path, encoding="utf-8") as file:
        data = file.read()

    for key, value in forbidden_characters.items():
        data = data.replace(key, value)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)
