# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the common tools used in the application"""
import uuid


def is_uuid4(input_string):
    """This function is used to check if the input is a valid UUID4"""
    try:
        # Attempt to create a UUID object from the input
        uuid.UUID(input_string, version=4)
        return True
    except ValueError:
        # If conversion fails, the input is not a valid UUID4
        return False


def is_request_json(request):
    """This function is used to check if the flask request is a json"""
    if not request.is_json:
        return False
    try:
        return request.get_json()
    except Exception:
        return False
