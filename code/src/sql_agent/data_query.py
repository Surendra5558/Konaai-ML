# # Copyright (C) KonaAI - All Rights Reserved
"""Data Query Response Handler"""
import re
from typing import List
from typing import Optional

import pandas as pd
from src.sql_agent.data_dictionary import SQLDataDictionary
from src.sql_agent.errors import ConfigurationError
from src.sql_agent.errors import HallucinationError
from src.sql_agent.errors import NotAllowedError
from src.sql_agent.filter_conditions import FilterGroup
from src.sql_agent.prompt_catalogue import PromptCatalogue
from src.sql_agent.query_explainer import SQLExplainerAgent
from src.sql_agent.query_parser import SQLParser
from src.sql_agent.retriever import get_data_dictionary
from src.sql_agent.retriever import semantic_search
from src.sql_agent.retriever import SQLColumn
from src.utils.agent_models import AgentResponseModel
from src.utils.agent_models import FollowUpQuestion
from src.utils.agent_models import Message
from src.utils.database_config import DataType
from src.utils.database_config import SQLDatabaseManager
from src.utils.global_config import GlobalSettings
from src.utils.instance import Instance
from src.utils.llm_factory import get_llm
from src.utils.operators import ValueOperators
from src.utils.status import Status


class SQLQueryAgent:
    """Data Query Response Handler"""

    id: str = "sql_query_agent"
    llm = None

    conversation_model: AgentResponseModel
    _data_dictionary: Optional[pd.DataFrame] = None
    _table_name: Optional[str] = None
    _database: Optional[SQLDatabaseManager] = None

    def execute(self, model: AgentResponseModel) -> AgentResponseModel:
        # sourcery skip: extract-duplicate-method
        """
        Execute the data-query agent pipeline for a single AgentResponseModel.
        This method orchestrates the preparation and generation of a SQL/data query response
        for the provided AgentResponseModel. It performs validation, manages follow-up
        question flow, and delegates final query creation. Any errors encountered are
        handled by recording an explanatory message on the model and marking the status
        as failed; exceptions are not propagated.
        Behavior:
        - Ensures the model.agent_id is set; if not, assigns self.id.
        - Stores the provided model on self.conversation_model for use by internal helpers.
        - Runs pre-validation logic via self._pre_validation().
        - If the model has pending follow-up questions (model.pending_follow_up_questions()
            returns True), logs an informational status and returns the model unchanged,
            awaiting user responses.
        - If the model has answered follow-up questions present (model.follow_up_questions),
            delegates handling to self.handle_follow_up_questions() and returns its result.
        - Otherwise, delegates to self._create_query() to produce and return the final
            AgentResponseModel containing the generated query/result.
        - On any exception, appends a user-facing error Message to the model (explaining
            that a valid SQL query could not be generated), logs a failed status with the
            exception information, and returns the (modified) model.
        Args:
                model (AgentResponseModel): The conversation/response model to process. This
                        object may be mutated (agent_id may be set, messages may be added, and
                        follow-up/question state may be updated).
        Returns:
                AgentResponseModel: The (possibly mutated) model after processing. This will
                either be the original model (if awaiting follow-ups or on error), the result
                of handle_follow_up_questions(), or the result of _create_query().
        Side effects:
        - Mutates model (sets agent_id if missing, may add Message instances).
        - Sets self.conversation_model to the provided model.
        - Calls Status.INFO or Status.FAILED for logging/telemetry.
        - Suppresses exceptions and converts them into a user-facing message on the model.
        Notes:
        - This method catches all exceptions to ensure the agent returns a usable model
            rather than raising; callers should not rely on exceptions for control flow.
        - The exact behaviors of model.pending_follow_up_questions(),
            model.follow_up_questions, handle_follow_up_questions(), and _create_query()
            depend on their respective implementations.
        """
        retry_count: int = 0
        try:
            return self._execute(model)
        except HallucinationError as error:
            msg = Message(
                role="agent",
                content="I'm sorry, but it seems there was an issue with the SQL query generation due to LLM hallucination. Please try rephrasing your question or providing more specific details. Ask for help if the issue persists.",
            )
            model.add_message(msg)
            Status.FAILED(
                "Data Query response generation failed due to hallucinated tables/columns.",
                error=str(error),
                traceback=False,
            )

            # remove all follow-up questions as they may no longer be valid
            model.follow_up_questions = []
            retry_count += 1
            return self._create_query(retry_count=retry_count)
        except Exception as error:
            msg = Message(
                role="agent",
                content="I'm sorry, I couldn't generate a valid SQL query based on your request or no data exists for the query. Please try rephrasing your question or providing more specific details. Ask for help if the issue persists.",
            )
            model.add_message(msg)
            Status.FAILED(
                "Data Query response generation failed.",
                error=str(error),
                traceback=True,
            )
            return model

    def _execute(self, model: AgentResponseModel) -> AgentResponseModel:
        # set agent id if not already set
        if not model.agent_id:
            model.agent_id = self.id

        self.conversation_model = model

        # PERFORM PRE-VALIDATION CHECKS
        self._pre_validation()

        # check if there are pending followup questions
        if model.pending_follow_up_questions():
            Status.INFO("Pending follow-up questions exist. Awaiting user responses.")
            return model

        # handle answered follow-up questions if any
        if model.follow_up_questions:
            Status.INFO("Handling answered follow-up questions.")
            return self.handle_follow_up_questions()

        return self._create_query()

    def _create_query(
        self, retry_count: int = 0, max_retries: int = 2
    ) -> AgentResponseModel:
        if retry_count > max_retries:
            raise ValueError("Maximum retries exceeded for SQL query generation.")

        if retry_count > 0:
            Status.INFO(
                f"Retrying SQL query generation. Attempt {retry_count} of {max_retries}."
            )

        # Get all user messages as per manager's requirement
        all_user_messages = [
            m.content for m in self.conversation_model.messages if m.role == "user"
        ]
        if not all_user_messages:
            raise ValueError("No user messages found.")

        # Get the last user message (current question)
        last_user_message: Message = (
            self.conversation_model.last_unsolicited_user_message()
        )
        if not last_user_message:
            raise ValueError("Last unsolicited message not found.")

        current_question = last_user_message.content

        # Step 1: Check if there is conversation history
        if len(all_user_messages) > 1:
            # Get the last agent explanation to provide context about what was previously queried
            last_agent_explanation = None
            for msg in reversed(self.conversation_model.messages):
                if msg.role == "agent" and msg.content_type == "text" and msg.content:
                    # Skip follow-up questions, get the actual explanation
                    if (
                        "?" not in msg.content or len(msg.content) > 50
                    ):  # Explanations are usually longer
                        last_agent_explanation = msg.content
                        break

            # Build conversation history with context
            conversation_history = ". ".join(all_user_messages[:-1])
            if last_agent_explanation:
                # Add the previous query context to help intent detection
                conversation_history = f"Previous query context: {last_agent_explanation}. User questions: {conversation_history}"

            # Step 2: Detect intent - is this a follow-up or new topic?
            intent = self._detect_intent(conversation_history, current_question)

            Status.INFO(f"Intent detected: {intent} for current question")

            # Step 3: Branch based on intent
            if intent == "NEW_TOPIC":
                # New topic - ignore history, use only current question
                user_prompt_for_sql = current_question
            else:  # FOLLOW_UP
                # Follow-up - include context
                user_prompt_for_sql = f"Previous context: {conversation_history}. Current question: {current_question}"
        else:
            # First question - no history
            user_prompt_for_sql = current_question

        # retrieve the most relevant columns from the data dictionary based on the current question
        # Use the latest question for semantic search to get relevant columns
        relevant_columns: Optional[List[SQLColumn]] = semantic_search(
            user_prompt=current_question,
            data_dictionary=self._data_dictionary,
            k=20,
        )

        # Generate SQL query with the appropriately formatted prompt
        prompt = self._query_prompt(
            user_prompt_for_sql,
            relevant_columns,
        )

        sql_query: Optional[str] = self._llm_response(prompt)
        if not sql_query:
            raise ValueError("No SQL query found in the response.")

        # validate the SQL query
        sql_query = self._validate_query(sql_query, retry_count=retry_count)
        if not sql_query:
            raise AssertionError("SQL query validation failed.")

        # improve column aliases and formatting
        parser = SQLParser(sql_query)
        if parser.is_valid and parser.is_select_query:
            sql_query = parser.update_column_aliases() or sql_query

        Status.SUCCESS("Data Query response generated successfully.")

        # check for filter conditions
        if parser.get_filter_columns():
            # Store message count before calling update_filter_questions
            message_count_before = len(self.conversation_model.messages)

            self.conversation_model = self.update_filter_questions(sql_query)

            # Check if update_filter_questions already handled the query
            # It handled it if: (1) follow-up questions were created, OR
            # (2) new messages were added (meaning query was executed)
            if (
                self.conversation_model.follow_up_questions
                or len(self.conversation_model.messages) > message_count_before
            ):
                return self.conversation_model

        if json_data := self._execute_and_format_query(sql_query):
            # If we got data, send it with the new content_type "data_table"
            self.conversation_model.add_message(
                Message(role="agent", content=json_data, content_type="data_table")
            )
        else:
            # If no data was found, send a simple text message
            self.conversation_model.add_message(
                Message(
                    role="agent",
                    content="I ran the query, but no data was found.",
                    content_type="text",
                )
            )

        # add explanation if needed
        return self._add_explanation(sql_query)

    def _execute_and_format_query(self, sql_query: str) -> Optional[str]:
        """
        Executes a SQL query and formats the result into a JSON string.
        Returns a JSON string of a list of dictionaries, or None on failure.
        """
        try:
            results_df = self._database.download_table_or_query(query=sql_query)
            # data = results_df.compute().to_dict(orient='records')
            # return json.dumps(data)

            if results_df is None or len(results_df) == 0:
                return None

            # Convert the Dask DataFrame to a Pandas DataFrame
            pandas_df = results_df.compute()

            return pandas_df.to_json(orient="records")
        except Exception as e:
            Status.FAILED("Failed to execute query and fetch data", error=str(e))
            return None

    def _get_plain_llm_response(self, prompt: str) -> Optional[str]:
        """
        Gets a plain text response from the LLM without SQL extraction.
        Used for intent detection and other non-SQL LLM calls.
        """
        try:
            response = (
                self.llm.invoke(prompt)
                if hasattr(self.llm, "invoke")
                else self.llm(prompt)
            )
            if not response:
                return None

            response: str = (
                response.content if hasattr(response, "content") else str(response)
            )

            return None if not isinstance(response, str) else response.strip()
        except Exception as e:
            Status.WARNING("Failed to get LLM response", error=str(e))
            return None

    def _detect_intent(self, conversation_history: str, current_question: str) -> str:
        """
        Detects if the current question is a follow-up to previous conversation or a new topic.
        Returns: "FOLLOW_UP" or "NEW_TOPIC"
        """
        if not conversation_history:
            return "NEW_TOPIC"

        intent_prompt = f"""You are a conversation analyst. Your job is to determine if a new user question is a follow-up to the previous conversation or if it is a completely new topic.

        Here is the conversation history:
        \"\"\"
        {conversation_history}
        \"\"\"

        Here is the new user question:
        \"\"\"
        {current_question}
        \"\"\"

        Analyze the new question in the context of the history.

        A FOLLOW_UP question is one that:
        - Asks about the same topic/entity but with different filters or conditions (e.g., "what about X country?" after asking about vendors)
        - Refines or modifies the previous query (e.g., changing a filter value, adding a condition)
        - Asks for more details about the same subject matter

        A NEW_TOPIC question is one that:
        - Completely changes the subject (e.g., from vendors to payments, from risk scores to transaction amounts)
        - Asks about a different entity or metric entirely
        - Has no relation to the previous query context

        Respond with ONLY one of these two words:
        FOLLOW_UP
        NEW_TOPIC"""

        try:
            response = self._get_plain_llm_response(intent_prompt)
            if not response:
                Status.WARNING(
                    "Intent detection: LLM returned empty response, defaulting to NEW_TOPIC"
                )
                return "NEW_TOPIC"

            Status.INFO(f"Intent detection - LLM raw response: {response}")

            response_upper = response.strip().upper()
            # Check for FOLLOW_UP first (more specific)
            if "FOLLOW_UP" in response_upper or "FOLLOWUP" in response_upper:
                Status.INFO("Intent detection: Classified as FOLLOW_UP")
                return "FOLLOW_UP"
            if "NEW_TOPIC" in response_upper or "NEW TOPIC" in response_upper:
                Status.INFO("Intent detection: Classified as NEW_TOPIC")
                return "NEW_TOPIC"
            # If neither is found, default to NEW_TOPIC but log a warning
            Status.WARNING(
                f"Intent detection: Unexpected response format, defaulting to NEW_TOPIC. Response: {response}"
            )
            return "NEW_TOPIC"
        except Exception as e:
            Status.WARNING(
                "Intent detection failed, defaulting to NEW_TOPIC", error=str(e)
            )
            return "NEW_TOPIC"

    def _add_explanation(self, sql_query: str) -> AgentResponseModel:
        explainer = SQLExplainerAgent()
        if explanation := explainer.execute(
            sql_query=sql_query,
            data_dictionary=self._data_dictionary,
            llm=self.llm,
        ):
            self.conversation_model.add_message(
                Message(role="agent", content=explanation)
            )
        return self.conversation_model

    def handle_follow_up_questions(self) -> AgentResponseModel:
        """
        Process answered follow-up questions from the conversation model to produce and finalize
        an updated SQL query, then attach that SQL to the conversation and return an agent response.
        Behavior:
        - If there are no follow-up questions, returns the existing conversation model unchanged.
        - Iterates over self.conversation_model.follow_up_questions and only processes questions
            where the question is marked as answered (q.is_answered is truthy).
        - For each answered question, validates that an operator, a placeholder SQL query, and a
            column name exist; logs warnings and skips the question if any are missing.
        - Uses SQLParser to validate and update the placeholder SQL by applying a filter update
            with the follow-up question's selected operator and selected values:
                updated_query = parser.update_filter(filter_column=column_name,
                                                                                         values=selected_values)
            If the parser is invalid or update_filter fails, the question is skipped and a warning
            is emitted.
        - When an update succeeds, removes that follow-up question from the conversation model,
            and continues applying further follow-up updates to the most recent SQL query.
        - After all follow-up questions have been processed, if no valid SQL query was produced,
            raises ValueError("No valid SQL query generated after handling follow-up questions.").
        - On success, logs completion, adds the finalized SQL query to the conversation model as
            an agent message with content_type="sql", and returns the result of self._add_explanation(sql_query).
        Returns:
                AgentResponseModel: Either the unchanged conversation model when there are no follow-up
                questions, or the conversation model enriched with the finalized SQL and any generated
                explanation (the return value of self._add_explanation).
        Side effects:
        - Mutates self.conversation_model by removing processed follow-up questions and adding
            a new agent message containing the finalized SQL.
        - Emits status logs for info, warnings, and success during processing.
        Assumptions:
        - Each follow-up question object provides at least these attributes:
            - is_answered (bool), answers (list or single value), answer_operator (str),
                sql_query (str placeholder), column_name (str), and question_id (identifier).
        - SQLParser is available and exposes is_valid and update_filter(filter_column, operator, values).
        """
        if not self.conversation_model.follow_up_questions:
            return self.conversation_model

        sql_query: str = None
        for q in self.conversation_model.follow_up_questions:
            q: FollowUpQuestion
            if not q.is_answered:
                continue

            Status.INFO("Processing follow-up question.", question_id=q.question_id)
            selected_values = q.answers

            selected_operator = q.answer_operator
            if not selected_operator:
                Status.WARNING(
                    "No operator selected for follow-up question.",
                    question_id=q.question_id,
                )
                continue

            # get the placeholder query
            placeholder_query: str = q.sql_query
            if not placeholder_query:
                Status.WARNING(
                    "No SQL query placeholder found for follow-up question.",
                    question_id=q.question_id,
                )
                continue

            if not sql_query:
                sql_query = placeholder_query
            parser = SQLParser(sql_query)
            if not parser.is_valid:
                Status.WARNING(
                    "Invalid SQL query in placeholder for follow-up question.",
                    question_id=q.question_id,
                    sql_query=placeholder_query,
                )
                continue

            # get the placeholder column name
            if not q.column_name:
                Status.WARNING(
                    "No column name found for follow-up question.",
                    question_id=q.question_id,
                )
                continue
            # get the placeholder column name
            column_name = q.column_name

            # update filter values in the query
            updated_query = parser.update_filter(
                filter_column=column_name,
                operator=selected_operator,
                values=selected_values,
            )
            if not updated_query:
                Status.WARNING(
                    "Failed to update filter values in SQL query.",
                    question_id=q.question_id,
                )
                continue

            # remove the follow-up question from the model
            self.conversation_model.remove_follow_up_questions(q)
            sql_query = updated_query

        if not sql_query:
            raise ValueError(
                "No valid SQL query generated after handling follow-up questions."
            )

        Status.SUCCESS("All follow-up questions processed. Finalized SQL query.")

        if json_data := self._execute_and_format_query(sql_query):
            # If we got data, send it with content_type "data_table"
            self.conversation_model.add_message(
                Message(role="agent", content=json_data, content_type="data_table")
            )
        else:
            # If no data was found, send a simple text message
            self.conversation_model.add_message(
                Message(
                    role="agent",
                    content="I ran the query, but no data was found.",
                    content_type="text",
                )
            )

        # add explanation if needed
        return self._add_explanation(sql_query)

    def update_filter_questions(self, sql_query: str) -> AgentResponseModel:
        """Update the conversation model with follow-up questions for filter conditions."""
        filter_group: FilterGroup = FilterGroup(
            query=sql_query, data_dictionary=self._data_dictionary
        )
        filter_group.create(db=self._database)
        if not filter_group.filters:
            if json_data := self._execute_and_format_query(sql_query):
                self.conversation_model.add_message(
                    Message(role="agent", content=json_data, content_type="data_table")
                )
            else:
                self.conversation_model.add_message(
                    Message(
                        role="agent",
                        content="I ran the query, but no data was found.",
                        content_type="text",
                    )
                )
            return self._add_explanation(sql_query)

        # add follow-up questions for each filter condition
        for filter_cond in filter_group.filters:
            if filter_cond.dtype in [DataType.DATE, DataType.DATETIME, DataType.NUMBER]:
                Status.INFO(
                    "Skipping non-categorical filter condition.",
                    column=filter_cond.column_name,
                    dtype=filter_cond.dtype,
                )
                continue  # skip non-categorical filters for now

            # ensure we have a selected operator
            if filter_cond.selected_operator is None:
                Status.INFO(
                    "Skipping filter condition with no selected operator.",
                    column=filter_cond.column_name,
                )
                continue

            fq = FollowUpQuestion(
                question=f"What values would you like to filter by for '{filter_cond.column_name}'?",
                answer_options=filter_cond.options,
                operator_options=list(ValueOperators),
                answer_operator=filter_cond.selected_operator,
            )

            # extra info for processing later
            fq.sql_query = sql_query
            fq.column_name = filter_cond.column_name

            self.conversation_model.add_follow_up_question(fq)

        return self.conversation_model

    @property
    def _response_key(self) -> str:
        """
        Returns the key used to identify the SQL query in the LLM response.
        This key is used to extract the SQL query from the LLM response.
        """
        return "<|SQL|>"

    def _llm_response(  # pylint: disable=too-many-return-statements
        self, prompt: str
    ) -> Optional[str]:
        response = (
            self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else self.llm(prompt)
        )
        if not response:
            raise ValueError("LLM did not return a response for user prompt.")

        response: str = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Ensure response is a string before membership test
        if not isinstance(response, str):
            raise TypeError("LLM response is not a string.")

        if (
            self._response_key.lower()
            not in response.lower()  # pylint: disable=unsupported-membership-test
        ):
            response_preview = (
                response[:200] + "..." if len(response) > 200 else response
            )
            raise ValueError(
                f"Response does not contain the expected SQL key. Response preview: {response_preview}"
            )

        response = response.strip().lower()
        Status.INFO("LLM response received.", response=response)

        # Find all occurrences of the response key (opening tags)
        opening_tags = []
        start_pos = 0
        while True:
            pos = response.find(self._response_key.lower(), start_pos)
            if pos == -1:
                break
            opening_tags.append(pos)
            start_pos = pos + 1

        if not opening_tags:
            Status.FAILED("No opening response key found")
            return None

        # Extract the first complete SQL query
        first_opening = opening_tags[0]
        sql_start = first_opening + len(self._response_key.lower())

        # Look for the next occurrence of response key as closing tag
        if len(opening_tags) > 1:
            # Multiple tags found, assume the second one is the closing tag
            sql_end = opening_tags[1]
            sql_content = response[sql_start:sql_end].strip()
        else:
            # Only one tag found, look for closing tag after the opening
            closing_pos = response.find(self._response_key.lower(), sql_start)
            if closing_pos != -1:
                sql_content = response[sql_start:closing_pos].strip()
            else:
                # No closing tag, take everything after opening tag
                sql_content = response[sql_start:].strip()

        sql_query = self._clean_text_for_query(sql_content)

        # Additional validation for the extracted SQL query
        if not sql_query:
            Status.FAILED("No SQL query found in the response.")
            return None

        if len(sql_query.strip()) < 5:
            Status.FAILED("Extracted SQL query is too short to be valid.")
            return None

        # Basic SQL validation - should at least contain SELECT
        if "select" not in sql_query.lower():
            Status.FAILED(
                "Extracted query does not appear to be a valid SELECT statement.",
                extracted_query=sql_query,
            )
            return None

        return sql_query

    def _clean_text_for_query(self, text: str) -> str:
        """
        Cleans and sanitizes a text string intended to be used as a SQL query.
        This method removes common separator patterns, extraneous alternatives, and explanatory text
        that may be introduced by language models or users. It attempts to extract the main SQL query,
        starting from the first 'SELECT' statement, and trims away any trailing comments or explanations.
        If a single semicolon is present, it truncates the query at the semicolon.
        Args:
            text (str): The input text potentially containing a SQL query and additional formatting or explanations.
        Returns:
            str: The cleaned SQL query string, with extraneous content removed.
        """
        if not text:
            return text

        # Remove common separator patterns that LLMs might introduce

        # Define patterns that commonly separate multiple queries or alternatives
        separator_patterns = [
            r"===+",  # Multiple equals signs
            r"---+",  # Multiple dashes
            r"\*\*\*+",  # Multiple asterisks
            r"OR:?\s*$",  # OR: at end of line
            r"Alternative:?\s*$",  # Alternative: at end of line
            r"Option\s+\d+:?\s*$",  # Option 1:, Option 2:, etc.
            r"Query\s+\d+:?\s*$",  # Query 1:, Query 2:, etc.
            r"Version\s+\d+:?\s*$",  # Version 1:, Version 2:, etc.
        ]

        # Split by any of these patterns and take the first part
        for pattern in separator_patterns:
            parts = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if len(parts) > 1:
                text = parts[0].strip()
                break

        # Find the SELECT statement
        _, match, after = text.lower().partition("select")
        if match:
            text = match + after

        # Clean up the query - if it has a single semicolon, take everything before it
        if text.count(";") == 1:
            text = text.split(";", 1)[0].strip()

        # Remove any trailing explanatory text (common patterns)
        # Look for common explanation starters and truncate there
        explanation_patterns = [
            r"\s+--\s+explanation:",
            r"\s+--\s+this query",
            r"\s+\/\*.*?\*\/",  # SQL comments
            r"\s+explanation:",
            r"\s+note:",
        ]

        for pattern in explanation_patterns:
            text = re.split(pattern, text, flags=re.IGNORECASE)[0].strip()

        return text.strip()

    def _validate_query(self, sql_query: str, retry_count: int = 0) -> Optional[str]:
        if not sql_query:
            raise ValueError("SQL query is empty.")

        Status.INFO("Validating the SQL query.")
        schema: pd.DataFrame = self._data_dictionary
        if schema is None or schema.empty:
            return None

        parser = SQLParser(sql_query)
        if not parser.is_valid:
            # retry generating the query
            self._create_query(retry_count=retry_count + 1)

        # check if the query contains valid tables and columns
        query_tables = parser.get_tables()
        reference_tables = [self._table_name]
        actual_tables = parser.get_tables(reference_tables=reference_tables)
        if actual_tables is None or not actual_tables:
            raise HallucinationError(
                f"Invalid table names in the SQL query. {sql_query}",
                tables=query_tables,
            )

        query_columns = parser.get_columns()
        reference_cols = schema[
            SQLDataDictionary.DDMetadata.COLUMN_NAME.value.strip()
        ].tolist()
        actual_columns = parser.get_columns(reference_columns=reference_cols)
        if actual_columns is None or not actual_columns:
            raise HallucinationError(
                f"Invalid column names in the SQL query. {sql_query}",
                columns=query_columns,
            )

        if (query_tables != actual_tables) or (query_columns != actual_columns):
            # find imagined tables and columns
            imagined_tables = list(set(query_tables) - set(actual_tables))
            imagined_columns = list(set(query_columns) - set(actual_columns))

            raise HallucinationError(
                f"Hallucinated tables or columns found in the SQL query. Imagined Tables: {imagined_tables}, Imagined Columns: {imagined_columns}",
            )

        Status.INFO("SQL query is valid and matches the expected tables and columns.")

        if not sql_query:
            raise ValueError("SQL query is empty after validation.")

        parser = SQLParser(sql_query)

        if not parser.is_select_query:
            raise NotAllowedError("Only data retrieval operations are allowed.")

        return parser.parse() if parser.is_valid else None

    def _query_prompt(
        self,
        user_prompt: str,
        data_dictionary: List[SQLColumn],
    ) -> str:
        """
        Generates a system prompt for an AI assistant to create a valid T-SQL query based on a user's natural language request.
        Args:
            user_prompt (str): The user's question or request in natural language.
            data_dictionary (List[DataQueryAgent.SQLColumn]): List of SQLColumn objects containing metadata for the table columns.
            table_name (str): The name of the SQL table to query.
        Returns:
            str: A formatted prompt string containing instructions, table metadata, and the user's prompt, ready for use with an AI language model.
        The generated prompt enforces rules for SQL generation, provides table metadata, and includes example usage to guide the AI assistant.
        """
        Status.INFO("Creating prompt for data query response generation.")

        system_prompt = """
        <|system|>You are a SQL expert. For any user question, return exactly ONE complete SQL statement.
        Rules:
        - ALWAYS preserve the full table name including schema if provided (e.g., analytics.payments, not just payments).
        - Never return all columns using "*".
        - Never create new columns.
        - Generate only valid T-SQL queries for Microsoft SQL Server.
        - Generate only select statements.
        - End SQL query with ;
        - Limit results to 10 rows unless specified.
        - Equality: use "=".
        - For math, wrap numbers in CAST([col] AS FLOAT).
        - Employ SUM, AVG, MAX, COUNT, GROUP BY, ORDER BY, LIMIT as needed.
        - Superlatives → SELECT TOP 10 [col] FROM ... ORDER BY [col] DESC
        - Counts → SELECT COUNT([col]) AS count FROM …
        - Totals → SELECT SUM(CAST([col] AS FLOAT)) AS total FROM …
        - Lists → select only the requested columns.
        - Include OVER clause for window functions.
        - Avoid placing filters in HAVING clause unless absolutely necessary.
        - Output only the SQL—no explanations or extra text.
        - Return ONLY ONE SQL query wrapped in {sql_key} tags.
        - Do not include multiple queries or alternatives.
        - Ensure proper tag formatting: opening tag, SQL query, closing tag.
        Table Metadata:
        Column Name (meta): Description
        {data_dictionary}
        Important Instructions:
        {prompt_data}
        Example:
        User: What is the total sales amount for each product in the last month?
        Assistant:{sql_key}SELECT [product_id], SUM([sales_amount]) AS total_sales FROM analytics.sales WHERE [sale_date] >= DATEADD(month, -1, GETDATE()) GROUP BY [product_id];{sql_key}
        <|end|>
        <|user|>{user_prompt}<|end|>
        <|assistant|>
        """

        # Format the data dictionary into a string representation with only descriptions
        data_dictionary_str = "\n".join(f"{col.meta}" for col in data_dictionary)

        # access prompt catalog
        catalog = PromptCatalogue(
            module=self.conversation_model.context.module,
            submodule=self.conversation_model.context.submodule,
            instance_id=self.conversation_model.context.instance_id,
        )
        prompt_data = catalog.get_prompt_data()

        return system_prompt.format(
            table_name=self._table_name,
            data_dictionary=data_dictionary_str,
            prompt_data=prompt_data or "No additional instructions provided.",
            user_prompt=user_prompt,
            sql_key=self._response_key,
        )

    def _pre_validation(self):
        # Check if we have the necessary context
        if not self.conversation_model.context:
            raise ValueError("Conversation context is missing.")

        # access llm
        if not self.llm:
            instance: Instance = GlobalSettings.instance_by_id(
                self.conversation_model.context.instance_id
            )
            cfg = instance.settings.llm_config
            self.llm = get_llm(llm_config=cfg, max_tokens=400)

        # Get database connection
        db = self.conversation_model.context.get_database()
        if not db:
            raise ValueError(
                "Database connection is missing.", self.conversation_model.context
            )

        # Check database connectivity
        if not db.is_db_connected:
            raise ConfigurationError("Database is not connected.", database=db)
        self._database = db

        # Check if we have a data table name
        table_name = self.conversation_model.context.get_data_table_name()
        if not table_name:
            raise ValueError(
                "Data table name is missing for the context.",
                self.conversation_model.context,
            )
        self._table_name = table_name

        # Basic table existence check
        if not db.does_table_exist(self._table_name):
            raise ValueError(
                "Table does not exist in the database.",
                self.conversation_model.context,
                table_name=self._table_name,
            )

        # Retrieve data dictionary for the table
        data_dict: Optional[pd.DataFrame] = get_data_dictionary(
            sql_table_name=self._table_name, db=db
        )
        if data_dict is None or len(data_dict) == 0:
            raise ValueError(
                "Data dictionary is missing or empty for the table.",
                self.conversation_model.context,
                table_name=self._table_name,
            )

        self._data_dictionary = data_dict
        Status.INFO(
            "Data dictionary retrieved successfully for the table.",
            table_name=self._table_name,
        )
