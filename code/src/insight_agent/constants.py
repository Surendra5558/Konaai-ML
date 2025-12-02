# # Copyright (C) KonaAI - All Rights Reserved
"""Audit Agent Constants Module"""
# --- LLM Model Name Constants ---
LLM_MODELS = {
    "AZURE": "gpt-4o-mini",
    "OPENAI": "gpt-4o-mini",
    "COHERE": "command-r-plus",
    "GEMINI": "gemini-1.5-pro-latest",
    "CLAUDE": "claude-3-5-sonnet-20240620",
}

# --- Default Values ---
DEFAULT_LLM_NAME = "COHERE"
DEFAULT_TEMPERATURE = 0.7

TRANSACTION_DATA_QUERY = """
            SELECT * FROM {table_name}
            WHERE SPT_RowID = '{transaction_id}'
            """

# --- Thresholds ---
CONTRIBUTION_PERCENT_THRESHOLD = 0  # Percentage threshold for feature contributions
