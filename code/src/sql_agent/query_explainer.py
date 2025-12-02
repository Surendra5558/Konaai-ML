# # Copyright (C) KonaAI - All Rights Reserved
"""Agent to explain SQL queries in business terms"""
from typing import Dict
from typing import Optional

import pandas as pd
from src.sql_agent.data_dictionary import SQLDataDictionary
from src.sql_agent.query_parser import SQLParser
from src.utils.status import Status


class SQLExplainerAgent:
    """Agent to explain SQL queries in business terms"""

    _response_key: str = "<|assistant|>"
    _sql_query: str = ""
    _data_dictionary: pd.DataFrame
    _llm = None

    def execute(  # pylint: disable=arguments-differ
        self, sql_query: str, data_dictionary: pd.DataFrame, llm
    ) -> Optional[str]:
        """Generates a business-friendly explanation for the given SQL query."""
        if not sql_query or not sql_query.strip():
            Status.WARNING("Cannot explain empty SQL query.")
            return None

        self._sql_query = sql_query
        self._data_dictionary = data_dictionary
        self._llm = llm

        if explanation := self._explain_query():
            Status.SUCCESS("Generated explanation for SQL query.")
            return explanation

        Status.WARNING("Failed to generate explanation for SQL query.")
        return None

    def _explain_query(self) -> Optional[str]:
        Status.INFO("Generating business-friendly explanation for SQL query.")

        if not self._sql_query:
            Status.WARNING("Cannot explain empty SQL query.")
            return None

        try:
            return self._process()
        except Exception as e:
            Status.FAILED(f"Error generating SQL explanation: {str(e)}")
            return None

    def _process(self):
        # Get table and column context for better explanations
        context_info = self._get_query_context(self._sql_query)

        # Try primary explanation method first
        explanation = self._generate_explanation_with_context(context_info)

        if not explanation:
            # Fallback to simpler explanation method
            Status.INFO("Primary explanation failed, trying fallback method.")
            explanation = self._generate_simple_explanation(self._sql_query)

        if explanation:
            # Clean and validate the explanation
            explanation = self._clean_explanation(explanation)
            Status.INFO("SQL query explanation generated successfully.")
            return explanation

        Status.WARNING("Failed to generate explanation for SQL query.")
        return None

    def _get_query_context(self, sql_query: str) -> Optional[Dict]:
        # sourcery skip: extract-method
        """
        Extracts context information from the SQL query and data dictionary.

        Args:
            sql_query (str): The SQL query to analyze

        Returns:
            dict: Context information including table name, columns, and their descriptions
        """
        context = {
            "table_name": "",
            "columns": [],
            "business_context": "",
        }

        try:
            # Parse SQL to get referenced columns
            parser = SQLParser(sql_query)
            if not parser.is_valid:
                Status.WARNING("SQL parsing failed, cannot extract query context.")
                return None

            if self._data_dictionary is None or self._data_dictionary.empty:
                Status.WARNING(
                    "Data dictionary is empty, cannot extract column context."
                )
                return None

            query_columns = parser.get_columns()

            # Get schema to provide column descriptions
            column_name_col = SQLDataDictionary.DDMetadata.COLUMN_NAME.value.strip()
            desc_col = SQLDataDictionary.DDMetadata.DESCRIPTION.value.strip()

            for col in query_columns:
                col_info = self._data_dictionary[
                    self._data_dictionary[column_name_col] == col
                ]
                if not col_info.empty:
                    description = col_info.iloc[0].get(desc_col, "No description")
                    context["columns"].append(
                        {
                            "name": col,
                            "description": (
                                str(description).strip()
                                if description
                                else "No description"
                            ),
                        }
                    )

            return context

        except Exception as e:
            Status.WARNING(f"Failed to extract query context: {str(e)}")
            return None

    def _generate_explanation_with_context(self, context: dict) -> Optional[str]:
        # Build column context string
        column_context = ""
        if context["columns"]:
            column_descriptions = []
            column_descriptions.extend(
                f"- {col['name']}: {col['description']}"
                for col in context["columns"]
                if col["description"] and col["description"] != "No description"
            )
            if column_descriptions:
                column_context = "\nColumn meanings:\n" + "\n".join(column_descriptions)

        system_prompt = f"""<|system|>You are a business analyst. Explain this SQL query in simple business terms.
                Context:
                {column_context}

                Rules:
                - Write for business users without technical SQL knowledge
                - Explain what information the query provides from database perspective without going into benefits or drawbacks
                - Use business terminology, not technical SQL terms
                - Be concise but complete (2-4 sentences)
                - Don't mention SQL syntax, tables, or technical details
                - Start with "This analysis..." or "This shows..." or "This finds..."

                Example:
                Query: SELECT vendor_name, SUM(amount) as total_spend FROM expenses WHERE date >= '2024-01-01' GROUP BY vendor_name ORDER BY total_spend DESC
                Explanation: This analysis shows the total spending with each vendor since the beginning of 2024, ranked from highest to lowest spend.

                <|end|>
                <|user|>Explain this query: {self._sql_query}<|end|>
                <|assistant|>"""

        try:
            return self._call_llm(system_prompt)
        except Exception as e:
            Status.WARNING(f"Context-based explanation failed: {str(e)}")
            return None

    def _generate_simple_explanation(self) -> Optional[str]:
        """Generate a concise, non-technical business explanation for the SQL query stored on the instance.
        Builds a system prompt that instructs an LLM to produce a 1–3 sentence explanation aimed at business users
        (without SQL terminology), starting with "This shows...", "This finds...", or "This analyzes...".
        The SQL text used is taken from self._sql_query and the prompt is sent to self._call_llm().
        Returns:
            Optional[str]: The generated explanation on success, or None if the LLM call fails (a warning is logged).
        """
        system_prompt = f"""<|system|>You are a business analyst. Explain this SQL query in simple terms.

                        Rules:
                        - Write for business users without SQL knowledge
                        - Explain what the query accomplishes in business terms from database perspective without going into benefits or drawbacks
                        - Be concise (1-3 sentences)
                        - Focus on the business purpose and results
                        - Avoid technical SQL terminology
                        - Start with "This shows..." or "This finds..." or "This analyzes..."

                        <|end|>
                        <|user|>{self._sql_query}<|end|>
                        <|assistant|>"""

        try:
            return self._call_llm(system_prompt)
        except Exception as e:
            Status.WARNING(f"Simple explanation failed: {str(e)}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Calls the LLM with the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM

        Returns:
            Optional[str]: The LLM response or None if failed
        """
        try:
            response = (
                self._llm.invoke(prompt)
                if hasattr(self._llm, "invoke")
                else self._llm(prompt)
            )
            if not response:
                raise ValueError("LLM did not return a response.")

            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            Status.WARNING(f"LLM call failed: {str(e)}")
            return None

    def _clean_explanation(self, explanation: str) -> str:
        """
        Cleans and validates the generated explanation.

        Args:
            explanation (str): The raw explanation from the LLM

        Returns:
            str: The cleaned explanation
        """
        if not explanation:
            return ""

        # Remove any SQL tags that might have been included
        explanation = explanation.replace(self._response_key, "").strip()

        # Remove common unwanted prefixes/suffixes
        unwanted_prefixes = [
            "explanation:",
            "answer:",
            "response:",
            "result:",
            "the query",
            "this query",
            "the sql",
        ]

        explanation_lower = explanation.lower()
        for prefix in unwanted_prefixes:
            if explanation_lower.startswith(prefix):
                explanation = explanation[len(prefix) :].strip()
                explanation_lower = explanation.lower()

        # Ensure proper capitalization
        if explanation and not explanation[0].isupper():
            explanation = explanation[0].upper() + explanation[1:]

        # Remove extra whitespace and newlines
        explanation = " ".join(explanation.split())

        # Ensure it ends with proper punctuation
        if explanation and explanation[-1] not in ".!?":
            explanation += "."

        return explanation
