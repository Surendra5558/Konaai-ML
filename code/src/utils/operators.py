# # Copyright (C) KonaAI - All Rights Reserved
"""Input Handlers for Chatbot"""
from enum import Enum


class ValueOperators(str, Enum):
    """Value Operators Enum"""

    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT EQUALS"
    GREATER_THAN = "GREATER THAN"
    LESS_THAN = "LESS THAN"
    GREATER_THAN_OR_EQUAL = "GREATER THAN OR EQUAL"
    LESS_THAN_OR_EQUAL = "LESS THAN OR EQUAL"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_MISSING = "IS MISSING"
    IS_NOT_MISSING = "IS NOT MISSING"
    CONTAINS = "CONTAINS"
    NOT_CONTAINS = "NOT CONTAINS"
    STARTS_WITH = "STARTS WITH"
    ENDS_WITH = "ENDS WITH"
