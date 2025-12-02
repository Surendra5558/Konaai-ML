# # Copyright (C) KonaAI - All Rights Reserved
"""SQLParser class for parsing SQL queries and helper functions."""
import re
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from sqlglot import Dialects
from sqlglot import exp
from sqlglot import parse
from sqlglot import parse_one
from sqlglot.errors import ParseError
from src.utils.operators import ValueOperators
from src.utils.status import Status

operator_map = {
    "eq": ValueOperators.EQUALS,
    "neq": ValueOperators.NOT_EQUALS,
    "gt": ValueOperators.GREATER_THAN,
    "lt": ValueOperators.LESS_THAN,
    "gte": ValueOperators.GREATER_THAN_OR_EQUAL,
    "lte": ValueOperators.LESS_THAN_OR_EQUAL,
    "in": ValueOperators.IN,
    "not in": ValueOperators.NOT_IN,
    "like": ValueOperators.CONTAINS,
    "not like": ValueOperators.NOT_CONTAINS,
}


@dataclass
class WhereCondition:
    """Data class to hold information about a WHERE condition."""

    condition: str = ""
    table: str = ""
    column: Optional[str] = None
    operator_key: str = ""
    mapped_operator: Optional[ValueOperators] = None


class SQLParser:
    """A class to parse SQL queries and helper functions."""

    def __init__(self, sql_query: str, dialect: Dialects = Dialects.TSQL):
        self.sql_query = sql_query
        # set tsql dialect as default for Microsoft SQL Server
        self.dialect = dialect

    def parse(self) -> Optional[str]:
        """
        Parses the SQL query using the specified SQL dialect and returns a formatted SQL string.

        Returns:
            Optional[str]: The formatted SQL query string if parsing is successful; otherwise, None.

        Raises:
            None: Any parsing errors are caught and handled internally.

        Notes:
            - If parsing fails, an error message is logged via Status.NOT_FOUND and None is returned.
            - The returned SQL string is formatted for readability.
        """
        try:
            parsed_query = parse_one(sql=self.sql_query, dialect=self.dialect)
            # convert to strings for easier handling later
            return parsed_query.sql(dialect=self.dialect, pretty=True)
        except ParseError as e:
            Status.NOT_FOUND(f"Error parsing SQL query: {e}")
            return None

    @property
    def is_valid(self) -> bool:
        """
        Check if the SQL query is valid.
        """
        try:
            parse(self.sql_query, dialect=self.dialect)
            return True
        except ParseError:
            Status.NOT_FOUND("Invalid SQL query.", query=self.sql_query)
            return False

    @property
    def is_select_query(self) -> bool:
        """
        Check if the SQL query is a SELECT query.

        Returns:
            bool: True if the query is a SELECT query, False otherwise.
        """
        try:
            stmts = parse(self.sql_query, dialect=self.dialect)
            return all(isinstance(stmt, exp.Select) for stmt in stmts)
        except ParseError:
            return False

    def get_tables(self, reference_tables: List[str] = None) -> Optional[set[str]]:
        """
        Extracts and returns a set of table names referenced in the SQL query.
        Args:
            reference_tables (List[str], optional): A list of reference table names to filter the extracted tables.
                Table names can be in the format 'schema.table' or just 'table'. If provided, only tables matching
                these references (case-insensitive, with or without schema) will be included in the result.
        Returns:
            Optional[set[str]]: A set of table names (as SQL strings, quoted and schema-qualified if applicable)
                found in the SQL query, filtered by reference_tables if provided. Returns None if the SQL query
                is invalid or if an error occurs during parsing.
        Raises:
            None. Any exceptions encountered during parsing are caught and logged, and None is returned instead.
        """
        try:
            if not self.is_valid:
                return None

            tables: set[str] = set()
            for stmt in parse(self.sql_query, dialect=self.dialect):
                for table in stmt.find_all(exp.Table):
                    # Handle tables with or without schema
                    schema = table.db or None
                    full_table_name = exp.table_(table.name, quoted=True, db=schema)
                    tables.add(full_table_name.sql(dialect=self.dialect))

            if reference_tables:
                # Process reference tables to handle both schema.table and table formats
                processed_ref_tables = set()
                for ref_table in reference_tables:
                    if "." in ref_table:
                        # Has schema: split into schema and table
                        parts = ref_table.split(".", 1)
                        schema, table_name = parts[0], parts[1]
                        full_ref_table = exp.table_(
                            table_name.lower(), quoted=True, db=schema.lower()
                        )
                    else:
                        # No schema: just table name
                        full_ref_table = exp.table_(
                            ref_table.lower(), quoted=True, db=None
                        )
                    processed_ref_tables.add(full_ref_table.sql(dialect=self.dialect))

                # Filter tables based on reference tables
                tables = {
                    table for table in tables if table.lower() in processed_ref_tables
                }

            # sort the tables for consistency
            if tables:
                tables = set(sorted(tables, key=lambda x: x.lower()))

            return tables
        except Exception as e:
            Status.FAILED(
                "Error parsing tables from SQL query", error=e, traceback=False
            )
            return None

    def get_columns(self, reference_columns: List[str] = None) -> Optional[set[str]]:
        """
        Extracts and returns a set of column names used in the SQL query, excluding aliases.
        Args:
            reference_columns (List[str], optional): A list of reference column names to filter the extracted columns.
                Only columns present in this list will be included in the result. If None, all columns found in the query are returned.
        Returns:
            Optional[set[str]]: A sorted set of actual column names used in the SQL query (excluding aliases),
                filtered by reference_columns if provided. Returns None if the query is invalid or an error occurs during parsing.
        Raises:
            None: Any exceptions are caught internally and logged via Status.FAILED.
        """
        try:
            if not self.is_valid:
                return None

            columns: set[str] = set()
            alias_names: set[str] = set()

            for stmt in parse(self.sql_query, dialect=self.dialect):
                # First, collect all alias names from SELECT expressions
                for alias_expr in stmt.find_all(exp.Alias):
                    if alias_expr.alias:
                        alias_names.add(alias_expr.alias.lower())

                # Then collect all column references
                for col in stmt.find_all(exp.Column):
                    # Skip columns that are actually aliases
                    if col.name.lower() not in alias_names:
                        full_col_name = exp.column(col.name, quoted=True)
                        columns.add(full_col_name.sql(dialect=self.dialect))

            if reference_columns:
                # make sure reference_columns is a set for faster lookup
                # also they are full column names
                reference_columns = {
                    exp.column(name.lower(), quoted=True).sql(dialect=self.dialect)
                    for name in reference_columns
                }
                # Filter columns based on actual columns if provided
                columns = {col for col in columns if col.lower() in reference_columns}

            # sort the columns for consistency
            if columns:
                columns = set(sorted(columns, key=lambda x: x.lower()))

            return columns
        except Exception as e:
            Status.FAILED(
                "Error parsing columns from SQL query", error=e, traceback=False
            )
            return None

    def update_column_aliases(self) -> Optional[str]:
        # sourcery skip: extract-method, use-named-expression
        """
        Updates column aliases in the SQL query to improve readability.

        This function:
        1. Improves existing aliases by cleaning up spaces, special characters
        2. Adds aliases for columns that don't have them using improved naming logic
        3. Ensures all aliases are unique and follow naming conventions

        Returns:
            Optional[str]: The updated SQL query with improved aliases, or None if parsing fails
        """
        try:
            Status.INFO("Updating column aliases for better readability.")
            tree = parse_one(self.sql_query, dialect=self.dialect)
            alias_map = {}
            used_aliases = set()

            # First pass: collect existing aliases and improve them
            for alias in tree.find_all(exp.Alias):
                if alias.alias:
                    original_alias = alias.alias
                    new_alias = self._create_readable_alias(
                        original_alias, used_aliases
                    )
                    alias_map[original_alias] = new_alias
                    used_aliases.add(new_alias.lower())
                    alias.set("alias", new_alias)

            # Second pass: find SELECT expressions without aliases and add them
            if hasattr(tree, "expressions") and tree.expressions:
                for expr in tree.expressions:
                    # Check if this expression doesn't have an alias
                    if not isinstance(expr, exp.Alias):
                        alias_name = self._generate_alias_for_expression(
                            expr, used_aliases
                        )
                        if alias_name:
                            # Wrap the expression in an Alias node
                            aliased_expr = exp.alias_(expr, alias_name)
                            # Replace the original expression with the aliased version
                            expr.replace(aliased_expr)
                            used_aliases.add(alias_name.lower())

            # Third pass: replace occurrences of original aliases in Column expressions
            for col in tree.find_all(exp.Column):
                if col.name in alias_map:
                    col.set("this", exp.to_identifier(alias_map[col.name]))

            return tree.sql(dialect=self.dialect, pretty=True)
        except ParseError as e:
            Status.NOT_FOUND(f"Error parsing SQL query: {e}")
            return None
        except Exception as e:
            Status.FAILED(f"Error updating column aliases: {e}")
            return None

    def _create_readable_alias(self, original_alias: str, used_aliases: set) -> str:
        """
        Creates a readable alias from an original alias by applying naming conventions.

        Args:
            original_alias (str): The original alias name
            used_aliases (set): Set of already used aliases to ensure uniqueness

        Returns:
            str: A cleaned, readable alias name
        """
        # Clean the alias
        clean_alias = original_alias.strip()

        # Replace spaces and special characters with underscores
        clean_alias = clean_alias.replace(" ", "_").replace("-", "_").replace(".", "_")

        # Remove multiple consecutive underscores
        while "__" in clean_alias:
            clean_alias = clean_alias.replace("__", "_")

        # Remove leading/trailing underscores
        clean_alias = clean_alias.strip("_")

        # Convert to title case for readability
        clean_alias = clean_alias.title()

        # Ensure it's not empty
        clean_alias = clean_alias or "Alias"

        self._update_alias_counter(clean_alias, used_aliases)
        return clean_alias

    def _generate_alias_for_expression(self, expr, used_aliases: set) -> Optional[str]:
        """
        Generates a meaningful alias for a SQL expression.

        Args:
            expr: The SQL expression to create an alias for
            used_aliases (set): Set of already used aliases to ensure uniqueness

        Returns:
            Optional[str]: A generated alias name, or None if unable to generate
        """
        alias_name = None

        # Handle different types of expressions
        if isinstance(expr, exp.Column):
            # Simple column reference
            column_name = expr.name
            alias_name = self._create_readable_column_alias(column_name)

        elif isinstance(expr, exp.Func):
            # Function calls (SUM, COUNT, AVG, etc.)
            func_name = (
                expr.this if hasattr(expr, "this") else str(expr.__class__.__name__)
            )

            # Get the column being operated on
            if hasattr(expr, "expressions") and expr.expressions:
                # Find the first column in the function arguments
                for arg in expr.expressions:
                    if isinstance(arg, exp.Column):
                        column_name = arg.name
                        alias_name = f"{func_name}_{self._create_readable_column_alias(column_name)}"
                        break

                # If no column found, use generic naming
                if not alias_name:
                    alias_name = f"{func_name}_Result"
            else:
                alias_name = f"{func_name}_Result"

        elif isinstance(expr, exp.Binary):
            # Binary operations (arithmetic, comparisons)
            alias_name = "Calculated_Value"

        elif isinstance(expr, exp.Case):
            # CASE expressions
            alias_name = "Case_Result"

        elif isinstance(expr, exp.Cast):
            # CAST expressions
            if hasattr(expr, "this") and isinstance(expr.this, exp.Column):
                column_name = expr.this.name
                alias_name = (
                    f"{self._create_readable_column_alias(column_name)}_Converted"
                )
            else:
                alias_name = "Converted_Value"

        elif isinstance(expr, exp.Literal):
            # Literal values
            alias_name = "Literal_Value"

        else:
            # Fallback for other expression types
            alias_name = "Expression_Result"

        # Ensure uniqueness if alias was generated
        if alias_name:
            self._update_alias_counter(alias_name, used_aliases)
        return alias_name

    def _update_alias_counter(self, arg0, used_aliases):
        base_alias = arg0
        counter = 1
        while arg0.lower() in used_aliases:
            arg0 = f"{base_alias}_{counter}"
            counter += 1

    def _create_readable_column_alias(self, column_name: str) -> str:
        """
        Creates a readable alias from a column name.

        Args:
            column_name (str): The original column name

        Returns:
            str: A readable alias version of the column name
        """
        # Remove brackets and quotes if present
        clean_name = column_name.strip("[]\"'`")

        # Split on common separators and join with underscores
        parts = []
        parts = next(
            (
                clean_name.split(separator)
                for separator in ["_", "-", ".", " "]
                if separator in clean_name
            ),
            [clean_name],
        )
        # Clean each part and convert to title case
        cleaned_parts = []
        for part in parts:
            if part := part.strip():
                # Handle camelCase by inserting underscores before capitals
                part = re.sub("([a-z])([A-Z])", r"\1_\2", part)
                cleaned_parts.append(part.title())

        # Join parts with underscores
        result = "_".join(cleaned_parts) if cleaned_parts else "Column"

        # Ensure it doesn't start with a number
        if result and result[0].isdigit():
            result = f"Col_{result}"

        return result

    def _get_operator_and_column(
        self, condition: exp.Expression
    ) -> Tuple[str, exp.Expression]:
        """Extract operator and column from condition, handling NOT operators properly"""
        if not isinstance(condition, exp.Not):
            # Handle regular operators
            return condition.key, condition.this
        # Handle NOT operators (NOT LIKE, NOT IN, etc.)
        inner_condition = condition.this
        if isinstance(inner_condition, exp.Like):
            return "not like", inner_condition.this

        if isinstance(inner_condition, exp.In):
            return "not in", inner_condition.this

        return f"not {inner_condition.key}", inner_condition.this

    def _extract_all_conditions(
        self,
        expression: exp.Expression,
        conditions_list: Optional[List[exp.Expression]] = None,
    ):
        """
        Recursively extract all leaf conditions from a complex WHERE expression.
        Handles AND, OR, NOT, parentheses, and nested structures.
        """
        if conditions_list is None:
            conditions_list = []

        if isinstance(expression, (exp.And, exp.Or)):
            # Recursively process left and right sides of logical operators
            self._extract_all_conditions(expression.left, conditions_list)
            self._extract_all_conditions(expression.right, conditions_list)
        elif isinstance(expression, exp.Paren):
            # Handle parentheses - extract what's inside
            self._extract_all_conditions(expression.this, conditions_list)
        elif isinstance(expression, exp.Not):
            # NOT wraps a complex expression - treat as one condition
            conditions_list.append(expression)
        else:
            # This is a leaf condition (comparison, etc.)
            conditions_list.append(expression)

        return conditions_list

    def _parse_where_conditions(self, where_expr: exp.Where) -> List[WhereCondition]:
        # sourcery skip: extract-duplicate-method
        """Parse WHERE clause and extract all individual conditions"""
        where_condition = where_expr.this
        all_conditions = self._extract_all_conditions(where_condition)

        parsed_conditions: List[WhereCondition] = []
        for cond in all_conditions:
            try:
                operator_key, column_expr = self._get_operator_and_column(cond)

                if isinstance(column_expr, exp.Column):
                    c = WhereCondition()
                    c.condition = str(cond)
                    c.column = column_expr.name
                    c.operator_key = operator_key
                    c.mapped_operator = operator_map.get(operator_key)
                    parsed_conditions.append(c)
                else:
                    # Handle complex conditions that don't fit the simple pattern
                    c = WhereCondition()
                    c.condition = str(cond)
                    c.column = None
                    c.operator_key = "complex"
                    c.mapped_operator = None

                parsed_conditions.append(c)
            except Exception as e:
                Status.FAILED(f"Error parsing condition '{cond}': {e}")

                # Handle any parsing errors gracefully
                c = WhereCondition()
                c.condition = str(cond)
                c.column = None
                c.operator_key = "error"
                c.mapped_operator = None

                parsed_conditions.append(c)

        return parsed_conditions

    def get_filter_columns(self) -> List[WhereCondition]:
        """
        Extract and return the list of unique filter columns inferred from the SQL query's WHERE clause.
        The method performs these steps:
        - Validates the parser state; returns None if the parser is not valid.
        - Parses the SQL statement (using the configured dialect) and collects all WHERE expressions.
        - If no WHERE clause is present, returns None.
        - Retrieves the set of tables referenced by the query via self.get_tables(). If zero or multiple
            tables are found, a warning is emitted and the method returns None because the table context
            for filters cannot be determined unambiguously.
        - For the single table found, iterates over all WHERE expressions and delegates parsing of each
            expression to self._parse_where_conditions(), which yields objects describing individual
            conditions (expected to be of type WhereCondition or similar with at least `column` and
            `condition` attributes).
        - Conditions that cannot be mapped to a simple column (i.e., cond_info.column is falsy) are
            skipped with a warning. For all recognized conditions, the method sets cond_info.table to the
            resolved table name and collects them.
        - Deduplicates collected conditions by column name in a case-insensitive manner, preserving
            the first occurrence for each column.
        - Emits an informational message listing the found filter columns.
        Returns
        -------
        List[WhereCondition] | None
                A list of unique WhereCondition objects (with their `table` attribute set to the
                resolved table name) representing the filter columns found in the WHERE clause, or
                None when the parser is invalid, no WHERE clause exists, or the table context is
                ambiguous (zero or multiple tables).
        Side effects
        ------------
        - Emits warnings and informational messages via the Status logging helpers (Status.WARNING,
            Status.INFO).
        - Calls parse_one(...) to build a parse tree and self._parse_where_conditions(...) to
            interpret individual WHERE subexpressions.
        Notes
        -----
        - Deduplication is case-insensitive on column names.
        - The returned order preserves the first-seen occurrence of each column from the WHERE
            expressions.
        """
        if not self.is_valid:
            return None

        tree = parse_one(self.sql_query, dialect=self.dialect)
        where_conditions = list(tree.find_all(exp.Where))

        if not where_conditions:
            return None

        # get the table name
        table_set = self.get_tables()
        if not table_set or len(table_set) > 1:
            Status.WARNING(
                "Multiple or no tables found in the query. Cannot determine table for filter columns."
            )
            return None

        # unquote the table name
        table = next(iter(table_set))

        filters: List[WhereCondition] = []
        for where_expr in where_conditions:
            conditions = self._parse_where_conditions(where_expr)
            for _, cond_info in enumerate(conditions, 1):
                if not cond_info.column:
                    Status.WARNING(
                        f"Complex or unrecognized condition in WHERE clause: {cond_info.condition}"
                    )
                    continue
                cond_info.table = table
                filters.append(cond_info)

        # find unique filters based on column
        unique_filters: List[WhereCondition] = []
        seen_columns = set()
        for filter_cond in filters:
            if filter_cond.column.lower() not in seen_columns:
                seen_columns.add(filter_cond.column.lower())
                unique_filters.append(filter_cond)
        Status.INFO(
            f"Found {len(unique_filters)} filter columns: {','.join([f.column for f in unique_filters])}"
        )
        return unique_filters

    def update_filter(
        self,
        filter_column: str,
        values: Union[str, List[str]],
        operator: ValueOperators,
    ) -> Optional[str]:
        """
        Update an existing filter in the current SQL query.
        This method checks whether the provided filter_column exists among the query's
        current filters (case-insensitive). If the column is not present, it reports
        the condition via Status.NOT_FOUND and returns None. If the column exists,
        it delegates the actual update work to self._update_filter(filter_column, values, operator).
        Any exception raised during the update is caught, reported via Status.FAILED,
        and the method returns None.
        Parameters
        ----------
        filter_column : str
            The name of the filter column to update. Matching is performed
            case-insensitively against existing query filter columns.
        values : Union[List[str], RangeValue]
            The new value(s) for the filter. For string-based filters this is
            typically a list of string values (e.g., for an IN-style update).
            For numeric filters this is a RangeValue (e.g., int/float or a
            domain-specific numeric type) appropriate for the chosen numeric operator.
        operator : ValueOperators
            The operator to apply when updating the filter. Use a member of
            ValueOperators for both numeric and string-based operations.
        Returns
        -------
        Optional[str]
            The updated filter expression or representation returned by
            self._update_filter on success. Returns None if the filter column
            was not found or if an error occurred during the update (errors are
            reported via Status.NOT_FOUND or Status.FAILED, respectively).
        Side effects
        ------------
        - Calls self.get_filter_columns() to determine existing filters.
        - May call Status.NOT_FOUND(...) or Status.FAILED(...) to report issues.
        - Delegates update logic to self._update_filter(...).
        """
        filters: List[WhereCondition] = self.get_filter_columns()
        if not filters:
            Status.NOT_FOUND("No filters found in the SQL query.")
            return None

        if filter_column.lower() not in [f.column.lower() for f in filters]:
            Status.NOT_FOUND(
                f"Filter column '{filter_column}' not found in query filters: {filters}"
            )
            return None

        try:
            return self._update_filter(filter_column, values, operator)
        except Exception as e:
            Status.FAILED(f"Error updating filters in SQL query: {e}")
            return None

    def _update_filter(  # pylint: disable=too-many-return-statements
        self,
        filter_column: str,
        values: Union[str, List[str]],
        operator: ValueOperators,
    ) -> Optional[str]:
        tree = parse_one(self.sql_query, dialect=self.dialect)
        where_conditions = list(tree.find_all(exp.Where))

        if not where_conditions:
            Status.NOT_FOUND(
                "No WHERE conditions found in the query to update filters."
            )
            return None

        # Find the specific condition expression that contains the target column
        target_expression = None
        for where_condition in where_conditions:
            target_expression = self._find_column_condition(
                where_condition.this, filter_column
            )
            if target_expression:
                break

        if not target_expression:
            Status.NOT_FOUND(
                f"Filter column '{filter_column}' not found in WHERE conditions."
            )
            return None

        Status.INFO(
            f"Updating filter on column '{filter_column}' with operator '{operator}'"
        )

        # Create the new filter expression
        column_expr = exp.column(filter_column, quoted=True)
        filter_expr = self._create_filter_expression(column_expr, values, operator)

        if not filter_expr:
            Status.NOT_FOUND("Failed to create filter expression.")
            return None

        # Replace only the specific condition expression with the new filter
        target_expression.replace(filter_expr)

        return tree.sql(dialect=self.dialect, pretty=True)

    def _find_column_condition(
        self, expression, column_name: str
    ):  # pylint: disable=too-many-return-statements
        """
        Recursively find the condition expression that contains the specified column.

        Args:
            expression: The expression to search in
            column_name: The name of the column to find

        Returns:
            The expression that directly contains the column condition, or None if not found
        """
        # For direct comparison expressions, check if they contain the target column
        if (
            isinstance(  # pylint: disable=too-many-boolean-expressions
                expression,
                (
                    exp.EQ,
                    exp.NEQ,
                    exp.GT,
                    exp.LT,
                    exp.GTE,
                    exp.LTE,
                    exp.In,
                    exp.Like,
                    exp.Between,
                    exp.Is,
                ),
            )
            and (
                hasattr(expression, "this") and isinstance(expression.this, exp.Column)
            )
            and (
                hasattr(expression, "this")
                and isinstance(expression.this, exp.Column)
                and expression.this.name.lower() == column_name.lower()
            )
        ):
            return expression
        # For NOT expressions, check the inner expression
        if isinstance(expression, exp.Not) and hasattr(expression, "this"):
            if (
                isinstance(expression.this, exp.Column)
                and expression.this.name.lower() == column_name.lower()
            ):
                return expression
            # For NOT(IN(...)) or NOT(other conditions)

            if (
                hasattr(expression.this, "this")
                and isinstance(expression.this.this, exp.Column)
                and expression.this.this.name.lower() == column_name.lower()
            ):
                return expression

        # For Paren expressions, unwrap and search inside
        if isinstance(expression, exp.Paren) and hasattr(expression, "this"):
            return self._find_column_condition(expression.this, column_name)

        # For complex expressions (AND, OR), search recursively
        if isinstance(expression, (exp.And, exp.Or)):
            # Check left side first
            if hasattr(expression, "left") and expression.left:
                if result := self._find_column_condition(expression.left, column_name):
                    return result

            # Then check right side
            if hasattr(expression, "right") and expression.right:
                if result := self._find_column_condition(expression.right, column_name):
                    return result

        return None

    def _get_singular_value(
        self, values: Union[int, float, str, List[Union[int, float, str]]]
    ) -> Optional[Union[int, float, str]]:
        if isinstance(values, list):
            return values[0] if values else None
        return values

    def _get_stack_values(
        self, values: Union[int, float, str, List[Union[int, float, str]]]
    ) -> List[Union[int, float, str]]:
        if isinstance(values, list):
            return values
        return [values] if values is not None else []

    def _create_filter_expression(
        self, column_expr, values, operator
    ):  # pylint: disable=too-many-return-statements
        # sourcery skip: low-code-quality
        """
        Create a filter expression based on the column, values, and operator.

        Args:
            column_expr: The column expression
            values: The filter values
            operator: The filter operator

        Returns:
            The created filter expression or None if invalid
        """
        if operator == ValueOperators.EQUALS:
            value = self._get_singular_value(values)
            if value is not None:
                return exp.EQ(
                    this=column_expr, expression=exp.Literal.string(str(value))
                )
            Status.WARNING("EQUALS operator with NULL value converted to IS MISSING.")
            return exp.Is(this=column_expr, expression=exp.Null())

        if operator == ValueOperators.NOT_EQUALS:
            value = self._get_singular_value(values)
            if value is not None:
                return exp.NEQ(
                    this=column_expr, expression=exp.Literal.string(str(value))
                )
            Status.WARNING(
                "NOT EQUALS operator with NULL value converted to IS NOT MISSING."
            )
            return exp.Not(this=column_expr, expression=exp.Null())

        if operator == ValueOperators.GREATER_THAN:
            value = self._get_singular_value(values)
            if value is not None:
                return exp.GT(this=column_expr, expression=exp.Literal.string(value))
            Status.WARNING(
                "GREATER THAN operator with NULL value converted to IS MISSING."
            )
            return exp.Is(this=column_expr, expression=exp.Null())

        if operator == ValueOperators.LESS_THAN:
            value = self._get_singular_value(values)
            if value is not None:
                return exp.LT(this=column_expr, expression=exp.Literal.string(value))
            Status.WARNING(
                "LESS THAN operator with NULL value converted to IS MISSING."
            )
            return exp.Is(this=column_expr, expression=exp.Null())

        if operator == ValueOperators.GREATER_THAN_OR_EQUAL:
            value = self._get_singular_value(values)
            if value is not None:
                return exp.GTE(this=column_expr, expression=exp.Literal.string(value))
            Status.WARNING(
                "GREATER THAN OR EQUAL operator with NULL value converted to IS MISSING."
            )
            return exp.Is(this=column_expr, expression=exp.Null())

        if operator == ValueOperators.LESS_THAN_OR_EQUAL:
            value = self._get_singular_value(values)
            if value is not None:
                return exp.LTE(this=column_expr, expression=exp.Literal.string(value))
            Status.WARNING(
                "LESS THAN OR EQUAL operator with NULL value converted to IS MISSING."
            )
            return exp.Is(this=column_expr, expression=exp.Null())

        if operator == ValueOperators.IN:
            values_list = self._get_stack_values(values)
            if len(values_list) > 0:
                return exp.In(
                    this=column_expr,
                    expressions=[exp.Literal.string(str(v)) for v in values_list],
                )
            Status.NOT_FOUND("IN operator requires a non-empty list of values.")
            return None

        if operator == ValueOperators.NOT_IN:
            values_list = self._get_stack_values(values)
            if len(values_list) > 0:
                return exp.In(
                    this=column_expr,
                    expressions=[exp.Literal.string(str(v)) for v in values_list],
                ).not_()

            Status.NOT_FOUND("NOT IN operator requires a non-empty list of values.")
            return None

        if operator == ValueOperators.IS_MISSING:
            return exp.Is(this=column_expr, expression=exp.Null())

        if operator == ValueOperators.IS_NOT_MISSING:
            return exp.Is(this=column_expr, expression=exp.NotNullColumnConstraint())

        if operator == ValueOperators.CONTAINS:
            value = self._get_singular_value(values)
            return exp.Like(
                this=column_expr, expression=exp.Literal.string(f"%{value}%")
            )

        if operator == ValueOperators.STARTS_WITH:
            value = self._get_singular_value(values)
            return exp.Like(
                this=column_expr, expression=exp.Literal.string(f"{value}%")
            )

        if operator == ValueOperators.ENDS_WITH:
            value = self._get_singular_value(values)
            return exp.Like(
                this=column_expr, expression=exp.Literal.string(f"%{value}")
            )

        Status.NOT_FOUND(f"Unsupported operator: {operator}")
        return None
