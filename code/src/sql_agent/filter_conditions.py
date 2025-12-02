# # Copyright (C) KonaAI - All Rights Reserved
"""Filter Conditions for SQL Queries"""
from typing import Any
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import dask.dataframe as dd
import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from src.sql_agent.data_dictionary import SQLDataDictionary
from src.sql_agent.query_parser import SQLParser
from src.sql_agent.query_parser import WhereCondition
from src.utils.database_config import DataType
from src.utils.database_config import SQLDatabaseManager
from src.utils.operators import ValueOperators
from src.utils.status import Status
from src.utils.status import StatusType


class Filter(BaseModel):
    """Represents a filter condition for a SQL query."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # Allow complex types like SQLDatabaseManager

    db: SQLDatabaseManager = Field(
        ..., description="Database manager instance", exclude=True
    )
    status: StatusType = Field(
        StatusType.PENDING, description="Status of the condition"
    )
    table_name: str = Field(..., description="Name of the table", exclude=True)
    column_name: str = Field(..., description="Name of the column")
    dtype: Optional[DataType] = Field(None, description="Data type of the column")
    operators: Optional[List[ValueOperators]] = Field(
        None, description="Operator for the condition"
    )
    options: Optional[Union[str, List[str]]] = Field(
        None, description="List of options for the condition"
    )
    selected_operator: Optional[ValueOperators] = Field(
        None, description="Operator for the condition"
    )
    selected_options: Optional[Union[str, List[str]]] = Field(
        None, description="Selected option for the condition"
    )

    def __init__(
        self, db: SQLDatabaseManager, table_name: str, column_name: str, **kwargs
    ):
        # Get the data type before calling super().__init__
        dtype = db.get_column_type(table_name, column_name)

        # Call Pydantic's __init__ with all required fields
        super().__init__(
            db=db, table_name=table_name, column_name=column_name, dtype=dtype, **kwargs
        )

        self.operators = list(ValueOperators)

        # Initialize options after the model is created
        self.get_options()

    def get_options(self):
        """
        Fetches and sets possible filter options for the specified column based on its data type.
        - For string and boolean columns, retrieves up to 100 unique values from the database and sets them as selectable options.
        - For numeric and date columns, retrieves the minimum and maximum values and sets them as range options.
        - If no options are found (all values are null), marks the filter as complete.
        Logs informational and warning messages during the process.
        Returns:
            None
        """
        Status.INFO(
            f"Fetching value options for column '{self.column_name}' of type '{self.dtype}'"
        )
        # Dont get more than 100 unique values
        query = f"SELECT DISTINCT TOP (100) {self.column_name} FROM {self.table_name} ORDER BY {self.column_name} ASC"
        results: dd.DataFrame = self.db.download_table_or_query(query=query)
        if results is not None and len(results) > 0:
            # self.options = results[self.column_name].tolist()
            self.options = results[self.column_name].astype(str).compute().tolist()

        # When options are all null, mark as complete
        if self.options is None:
            Status.WARNING(
                f"No options found for column '{self.column_name}' in table '{self.table_name}'. Marking filter as complete.",
                self,
            )
            self.mark_complete()

    def set_choices(self, operator: ValueOperators, selected_options: Any):
        """
        Sets the operator and selected options for the filter condition.
        Parameters:
            operator (ValueOperators): The operator to be used for filtering (e.g., equals, greater than, contains).
            selected_options (Any): The value(s) to be used with the operator for filtering.
        Raises:
            ValueError: If the data type (`dtype`) of the column is not set, indicating the column was not found in the specified table.
        Side Effects:
            - Updates the status to NOT_FOUND and raises an exception if the column is missing.
            - Sets the selected operator and options.
            - Marks the filter condition as complete.
        """
        if self.dtype is None:
            self.status = StatusType.NOT_FOUND
            raise ValueError(
                f"Column '{self.column_name}' not found in table '{self.table_name}'."
            )

        self.selected_operator = operator
        self.selected_options = selected_options
        self.mark_complete()

    def mark_complete(self):
        """
        Mark the current instance as completed by setting its status to StatusType.COMPLETED.
        """
        self.status = StatusType.COMPLETED

    @model_validator(mode="after")
    def validate(self) -> "Filter":
        """
        Validates the filter by ensuring the data type (`dtype`) is set.
        Raises:
            ValueError: If `dtype` is None, indicating the column was not found in the specified table.
        Returns:
            Filter: The validated Filter instance.
        """
        # validate dtype
        if self.dtype is None:
            self.status = StatusType.NOT_FOUND
            raise ValueError(
                f"Column '{self.column_name}' not found in table '{self.table_name}'."
            )

        return self

    def __str__(self):
        return f"Filter(table={self.table_name}, column={self.column_name}, dtype={self.dtype}, status={self.status})"


class FilterGroup(BaseModel):
    """Represents a group of filter conditions for a SQL query."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # Allow complex types like data dictionary

    query: str = Field(..., description="Original SQL query")
    data_dictionary: pd.DataFrame = Field(
        ..., description="Schema of the query result", exclude=True
    )
    filters: Optional[List[Filter]] = Field(
        None, description="List of filter conditions"
    )
    logical_operator: Literal["AND", "OR"] = Field(
        "AND", description="Logical operator to combine conditions (AND/OR)"
    )

    def add_filter(self, filter_condition: Filter):
        """
        Adds a filter condition to the list of filters.

        If the filters list is not initialized, it creates an empty list before appending the new filter condition.

        Args:
            filter_condition (Filter): The filter condition to be added to the filters list.
        """
        if self.filters is None:
            self.filters = []
        self.filters.append(filter_condition)

    def remove_filter(self, index: int):
        """
        Removes a filter from the filters list at the specified index.

        Args:
            index (int): The position of the filter to remove.

        Raises:
            IndexError: If the provided index is out of range.
        """
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
        else:
            raise IndexError("Filter index out of range")

    def clear_filters(self):
        """
        Clears all filter conditions by resetting the filters list to empty.

        This method removes any previously applied filters, allowing for a fresh start
        when building new filter conditions.
        """
        self.filters = []

    def _next_pending_filter(self) -> Optional[Filter]:
        Status.INFO("Checking for next pending filter condition")
        return next(
            (
                filter
                for filter in self.filters
                if filter.status != StatusType.COMPLETED
            ),
            None,
        )

    def create(self, db: SQLDatabaseManager):
        """
        Creates filter conditions for the SQL query by extracting filterable columns and adding them as Filter objects.
        Also normalizes column names in the data dictionary to lowercase for easier matching.
        Args:
            db (SQLDatabaseManager): The database manager instance used for filter creation.
        Side Effects:
            - Adds Filter objects to the current instance for each filterable column found in the query.
            - Updates the 'COLUMN_NAME' field in the data dictionary to lowercase if the data dictionary is present.
        Warnings:
            - Issues a warning if no filterable columns are found in the query.
        """
        parser = SQLParser(self.query)

        filter_conditions: Optional[List[WhereCondition]] = parser.get_filter_columns()
        if not filter_conditions:
            Status.WARNING("No filterable columns found in the query.")
            return

        for where_cond in filter_conditions:
            if filter_condition := Filter(
                db=db,
                table_name=where_cond.table,
                column_name=where_cond.column,
                selected_operator=where_cond.mapped_operator,
            ):
                self.add_filter(filter_condition)

        # lower all all column names in data dictionary for easier matching
        if self.data_dictionary is not None and len(self.data_dictionary) > 0:
            name_column = SQLDataDictionary.DDMetadata.COLUMN_NAME.value
            self.data_dictionary[name_column] = self.data_dictionary[
                name_column
            ].str.lower()
