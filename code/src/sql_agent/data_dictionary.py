# # Copyright (C) KonaAI - All Rights Reserved
"""SQL Data Dictionary Module"""
import re
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from sqlalchemy.sql import quoted_name
from src.utils.database_config import SQLDatabaseManager
from src.utils.llm_config import BaseLLMConfig
from src.utils.llm_factory import get_llm
from src.utils.status import Status
from tqdm import tqdm


class SQLDataDictionary(BaseModel):
    """SQL Data Dictionary class for retrieving table schema information."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    table_schema: str = Field(..., description="Schema of the database table")
    table_name: str = Field(..., description="Name of the database table")
    columns: Optional[pd.DataFrame] = Field(
        None, description="List of columns in the table"
    )
    db: SQLDatabaseManager = Field(
        ..., exclude=True, description="Database manager instance"
    )

    class DDMetadata(Enum):
        """Metadata fields for the data dictionary."""

        COLUMN_NAME = "COLUMN_NAME"
        DATA_TYPE = "DATA_TYPE"
        IS_NULLABLE = "IS_NULLABLE"
        IS_PRIMARY_KEY = "IS_PRIMARY_KEY"
        IS_UNIQUE = "IS_UNIQUE"
        DESCRIPTION = "DESCRIPTION"
        EXCLUDE = "EXCLUDE"

    def __init__(self, table_schema: str, table_name: str, db: SQLDatabaseManager):
        """
        Initializes the SQLDataDictionary instance with the provided schema, table name, and database handler.

        Args:
            table_schema (str): The schema of the database table.
            table_name (str): The name of the database table.
        """
        super().__init__(table_schema=table_schema, table_name=table_name, db=db)
        self.columns = None  # Initialize columns attribute

    def _load_columns(self) -> Optional[pd.DataFrame]:
        """Loads metadata about columns for a specific table from the database.
        This method constructs and executes a SQL query to retrieve column information
        such as column name, data type, nullability, primary key status, and uniqueness
        for the specified table and schema. The results are returned as a pandas DataFrame.
        Returns:
            Optional[pd.DataFrame]: A DataFrame containing column metadata if data is found,
            otherwise None."""

        query_template = """
            SELECT
                cols.column_name AS {column_name},
                cols.data_type AS {data_type},
                cols.is_nullable AS {is_nullable},
                CASE WHEN pk.column_name IS NOT NULL THEN 'YES' ELSE 'NO' END AS {is_primary_key},
                CASE WHEN uq.column_name IS NOT NULL THEN 'YES' ELSE 'NO' END AS {is_unique},
                -- empty description field to be filled later
                NULL AS {description},
                0 AS {exclude}
            FROM
                information_schema.columns cols
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_name = '{table}'
                AND tc.table_schema = '{schema}'
            ) pk ON cols.column_name = pk.column_name
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'UNIQUE'
                AND tc.table_name = '{table}'
                AND tc.table_schema = '{schema}'
            ) uq ON cols.column_name = uq.column_name
            WHERE
                cols.table_name = '{table}'
                AND cols.table_schema = '{schema}'
            ORDER BY
                cols.ordinal_position;
            """

        # Format the query
        query = query_template.format(
            schema=self.table_schema,
            table=self.table_name,
            column_name=self.DDMetadata.COLUMN_NAME.value,
            data_type=self.DDMetadata.DATA_TYPE.value,
            is_nullable=self.DDMetadata.IS_NULLABLE.value,
            is_primary_key=self.DDMetadata.IS_PRIMARY_KEY.value,
            is_unique=self.DDMetadata.IS_UNIQUE.value,
            description=self.DDMetadata.DESCRIPTION.value,
            exclude=self.DDMetadata.EXCLUDE.value,
        )
        ddf = self.db.download_table_or_query(query=query)
        if ddf is None or len(ddf) == 0:
            Status.WARNING(
                f"No data found for table {self.table_name} in schema {self.table_schema}."
            )
            return None

        return ddf.compute()

    def get_schema(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the schema of the data dictionary table as a pandas DataFrame.
        Automatically syncs with actual database schema to include new columns and remove deleted ones.
        Uses current database schema as source of truth and merges descriptions from data dictionary.
        Returns:
            Optional[pd.DataFrame]: The schema as a DataFrame if available, otherwise None.
        """
        target_table_name = self.data_dictionary_table_name
        if not target_table_name:
            Status.FAILED("Failed to retrieve data dictionary table name", self)
            return self._load_columns()

        # STEP 1: Get actual schema from database (source of truth)
        actual_schema: pd.DataFrame = self._load_columns()
        if actual_schema is None or actual_schema.empty:
            Status.WARNING(f"No actual schema found for table {self.table_name}")
            return None

        # STEP 2: Get data dictionary from table (has descriptions)
        dff = self.db.download_table_or_query(table_name=target_table_name)
        if dff is None or len(dff) == 0:
            Status.WARNING(
                f"No data found in the data dictionary table {target_table_name}. "
                f"Using actual schema without descriptions."
            )
            return actual_schema  # Return actual schema even without descriptions

        described_schema: pd.DataFrame = dff.compute()

        # STEP 3: Get column name key for merging
        col_name_key = self.DDMetadata.COLUMN_NAME.value
        desc_col = self.DDMetadata.DESCRIPTION.value
        exclude_col = self.DDMetadata.EXCLUDE.value

        # STEP 4: Merge schemas - Use actual_schema as base (LEFT JOIN)
        # Prepare columns to merge from data dictionary
        dict_columns_to_merge = [col_name_key]
        if desc_col in described_schema.columns:
            dict_columns_to_merge.append(desc_col)
        if exclude_col in described_schema.columns:
            dict_columns_to_merge.append(exclude_col)

        merged_schema = actual_schema.merge(
            described_schema[dict_columns_to_merge],
            on=col_name_key,
            how="left",  # LEFT JOIN: keeps all columns from actual_schema
            suffixes=("", "_dict"),
        )

        # STEP 5: Clean up duplicate columns
        # Use description from data dictionary if available, else keep original (NULL)
        if f"{desc_col}_dict" in merged_schema.columns:
            merged_schema[desc_col] = merged_schema[f"{desc_col}_dict"].fillna(
                merged_schema[desc_col]
            )
            merged_schema = merged_schema.drop(columns=[f"{desc_col}_dict"])

        # Handle exclude column
        if f"{exclude_col}_dict" in merged_schema.columns:
            merged_schema[exclude_col] = merged_schema[f"{exclude_col}_dict"].fillna(
                merged_schema[exclude_col]
                if exclude_col in merged_schema.columns
                else False
            )
            merged_schema = merged_schema.drop(columns=[f"{exclude_col}_dict"])

        # STEP 6: Detect and log schema changes
        actual_columns = set(actual_schema[col_name_key].values)
        dict_columns = (
            set(described_schema[col_name_key].values)
            if not described_schema.empty
            else set()
        )

        new_columns = actual_columns - dict_columns
        deleted_columns = dict_columns - actual_columns

        if new_columns:
            Status.INFO(
                f"New columns detected in database (not in data dictionary): {', '.join(new_columns)}. "
                f"These columns will be included but may need descriptions.",
                self,
            )

        if deleted_columns:
            Status.WARNING(
                f"Columns in data dictionary no longer exist in database: {', '.join(deleted_columns)}. "
                f"These columns have been removed from the schema.",
                self,
            )

        return merged_schema

    def _is_description_updated(self) -> bool:
        """
        Checks if all columns in the schema have descriptions.
        Uses get_schema() which already handles merging database schema with data dictionary,
        detects new/deleted columns, and logs discrepancies.
        Returns:
            bool: True if all columns have non-null descriptions, False otherwise.
        """
        # Get the merged schema (get_schema() already handles:
        # - Loading actual schema from database
        # - Loading data dictionary from table
        # - LEFT JOIN merge (database as source of truth)
        # - Detecting and logging new/deleted columns
        # - Fallback when data dictionary is missing/empty)
        merged_schema: pd.DataFrame = self.get_schema()

        if merged_schema is None or merged_schema.empty:
            Status.WARNING(
                f"No schema found for table {self.table_name} in schema {self.table_schema}. Description not updated."
            )
            self.columns = None
            return False

        # Set self.columns to the merged schema
        self.columns = merged_schema

        # Check if description column exists
        desc_col = self.DDMetadata.DESCRIPTION.value
        if desc_col not in self.columns.columns:
            Status.WARNING(
                f"No description column found in schema for table {self.table_name} in schema {self.table_schema}."
            )
            return False

        # Strip description values safely (only if it's a string/object column)
        if self.columns[desc_col].dtype == "object":
            self.columns[desc_col] = self.columns[desc_col].str.strip()

        # Check if all columns have non-null, non-empty descriptions
        has_descriptions = self.columns[desc_col].notnull() & (
            self.columns[desc_col] != ""
        )
        return all(has_descriptions)

    def update_schema_description(
        self,
        llm_config: BaseLLMConfig,
        ready_descriptions: Dict[str, str] = None,
        exclude_patterns: List[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Updates the descriptions of all columns in the schema using an LLM (Large Language Model).
        For each column in the schema, this method generates a detailed, plain-language description suitable for documentation,
        avoiding technical jargon such as data types or constraints. The descriptions are generated by prompting an LLM with
        relevant column metadata and updating each column's `description` attribute with the LLM's response.
        Raises:
            ValueError: If no columns are present in the schema.
        Returns:
            pd.DataFrame: The updated DataFrame of columns with their descriptions.
        """

        # Check if the description is already updated
        if self._is_description_updated():
            Status.INFO("Schema description already updated. Skipping update.", self)
            return self.columns

        Status.INFO("Updating schema description", self)

        prompt = """
            <|system|>You are a professional business process analyst. Your task is to write a description for the following database column.
            Rules:
            1. Write a plain-text description of the column in simple, natural language.
            2. Do not use technical jargon or mention data types, primary keys, or unique constraints.
            2. Do not use bullet points, questions, or tables.
            3. Avoid using private, personal, indecent, offensive, discriminatory, explicit, or harmful language.
            4. Avoid speculative assumptions or sensitive personal data.
            Table Name: {table_name}
            Column Name: {column_name}
            Column Type: {column_type}
            Is it a Primary Key: {column_primary_key}
            Is it Unique: {column_unique}
            <|end|>
            <|user|>Write a plain-text description of the column<|end|>
            <|assistant|>
        """

        if self.columns is None or len(self.columns) == 0:
            # If no columns are found, log a warning and return None
            Status.WARNING("No columns found in the schema. Description not updated.")
            return None

        # If ready_descriptions are provided, use them to update the descriptions
        if ready_descriptions:
            Status.INFO("Using provided descriptions to update schema", self)
            for column_name, description in ready_descriptions.items():
                if (
                    column_name
                    in self.columns[self.DDMetadata.COLUMN_NAME.value].values
                ):
                    self.columns.loc[
                        self.columns[self.DDMetadata.COLUMN_NAME.value] == column_name,
                        self.DDMetadata.DESCRIPTION.value,
                    ] = description
            Status.INFO("Schema description updated with provided descriptions", self)
        if exclude_patterns:
            # Pre-compile the regex pattern for efficiency
            pattern = re.compile("|".join(exclude_patterns), flags=re.IGNORECASE)
            # Filter rows using apply with the pre-compiled regex
            self.columns = self.columns[
                ~self.columns[self.DDMetadata.COLUMN_NAME.value].apply(
                    lambda x: bool(pattern.search(x)) if pd.notnull(x) else False
                )
            ]

        # find columns that do not have a description
        missing_description_columns = self.columns[
            self.columns[self.DDMetadata.DESCRIPTION.value].isnull()
        ][self.DDMetadata.COLUMN_NAME.value].tolist()

        new_descriptions: int = 0
        for column_name in tqdm(
            missing_description_columns,
            desc="Updating column descriptions",
            unit="column",
        ):
            # Get the column metadata
            column: pd.Series = self.columns[
                self.columns[self.DDMetadata.COLUMN_NAME.value] == column_name
            ].iloc[0]

            # Format the prompt with the column metadata
            updated_prompt = prompt.format(
                table_name=self.table_name,
                column_name=column_name,
                column_type=column[self.DDMetadata.DATA_TYPE.value],
                column_primary_key=column[self.DDMetadata.IS_PRIMARY_KEY.value]
                or False,
                column_unique=column[self.DDMetadata.IS_UNIQUE.value] or False,
            )

            try:
                llm = get_llm(llm_config=llm_config, max_tokens=300)
                response: str = (
                    llm.invoke(updated_prompt)
                    if hasattr(llm, "invoke")
                    else llm(updated_prompt)
                )
            except Exception as e:
                Status.FAILED(
                    f"Error while generating description for column {column_name}: {e}"
                )
                continue

            if not response:
                Status.NOT_FOUND(f"No response received for column {column_name}")
                continue

            # Update the column description
            self.columns.loc[
                self.columns[self.DDMetadata.COLUMN_NAME.value] == column_name,
                self.DDMetadata.DESCRIPTION.value,
            ] = response
            new_descriptions += 1

        if new_descriptions == 0:
            Status.WARNING(
                "No new descriptions generated. Schema description not updated."
            )
            return None

        Status.SUCCESS(
            f"Schema description updated successfully. {new_descriptions} new descriptions generated.",
            self,
        )
        return self.columns

    @property
    def data_dictionary_table_name(self) -> Optional[str]:
        """Generates the name of the data dictionary table based on the schema and table name."""
        try:
            schema_name = quoted_name(value=self.table_schema, quote=True)
            table_name = quoted_name(
                value=self.table_name.replace(" ", "_"), quote=True
            )
            return f"{schema_name}.{table_name}_DataDictionary"
        except Exception as e:
            Status.FAILED(f"Error generating data dictionary table name: {e}")
            return None

    @property
    def exists(self) -> bool:
        """Checks if the data dictionary exists and is populated for the specified table."""
        table_name = self.data_dictionary_table_name
        if not table_name:
            return False

        if not self.db.does_table_exist(table_name=table_name):
            return False

        dff = self.db.download_table_or_query(table_name=table_name)
        if dff is None or len(dff) == 0:
            return False

        # Check if all columns have non-null descriptions
        ddf: pd.DataFrame = dff.compute()
        # return false only if all descriptions are null else true
        return not ddf[self.DDMetadata.DESCRIPTION.value].isnull().all()

    def __str__(self):
        columns_summary = (
            f"{len(self.columns)} columns loaded"
            if self.columns is not None
            else "No columns loaded"
        )
        return f"(Schema: {self.table_schema}, Table Name: {self.table_name}, {columns_summary})"
