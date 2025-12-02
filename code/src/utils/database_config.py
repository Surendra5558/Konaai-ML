# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides a Pydantic model for managing SQL Server database connections."""
import time
import urllib
from datetime import timedelta
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from urllib.parse import quote_plus

import dask.dataframe as dd
import humanize
import pandas as pd
import pyodbc
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from src.tools.dask_tools import DASK_KEY
from src.tools.dask_tools import infer_dtypes
from src.tools.dask_tools import repartition_data
from src.utils.file_mgmt import FileHandler
from src.utils.status import Status


class DataType(str, Enum):
    """DataType Enum for SQL column types"""

    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


class SQLDatabaseManager(BaseModel):
    """SQLDatabaseManager is a Pydantic model for managing SQL Server database connections.
    Attributes:
    -----------
        Server (Optional[str]): The name or network address of the SQL Server instance.
        Database (Optional[str]): The name of the database to connect to.
        Username (Optional[str]): The username for database authentication.
        Password (Optional[SecretStr]): The password for database authentication, stored securely.
        Trusted (bool): Indicates whether to use Windows Authentication (single sign-on).
        _connection_string (str): The constructed connection string for the database (internal use).

    Class Attributes:
        model_config (ConfigDict): Pydantic configuration for custom JSON encoding, especially for SecretStr.
    """

    Server: Optional[str] = Field(None, description="Server Name")
    Database: Optional[str] = Field(None, description="Database Name")
    Username: Optional[str] = Field(None, description="Username")
    Password: Optional[SecretStr] = Field(None, description="Password")
    Trusted: bool = Field(False, description="Use when using single sign-on")
    _connection_string: str = None

    model_config = ConfigDict(
        json_encoders={
            SecretStr: lambda v: (
                v.get_secret_value() if isinstance(v, SecretStr) else v
            ),
        },
    )

    def __str__(self):
        if self.Trusted:
            return f"Server={self.Server}, Database={self.Database}, Trusted={self.Trusted}"
        return (
            f"Server={self.Server}, Database={self.Database}, Username={self.Username}"
        )

    @property
    def is_db_connected(self) -> bool:
        """This function checks if the database is connected

        Returns:
            bool: True if connected, False otherwise
        """
        is_connected = False
        # Set connection string if not set
        self._set_connection_string()

        if self._connection_string is None:
            Status.NOT_FOUND("Connection string is not set", self)
            return is_connected

        # Check if connection to DB is working with pyodbc
        try:
            with pyodbc.connect(self._connection_string) as conn:
                is_connected = True
            conn.close()

        except pyodbc.Error as _e:
            Status.FAILED(
                "Can not connect to database", self, error=_e, traceback=False
            )

        return is_connected

    def _get_driver(self) -> str:
        """This function returns the driver

        Returns:
            str: driver
        """
        drivers = list(pyodbc.drivers())
        sql_server_drivers = [driver for driver in drivers if "SQL Server" in driver]
        return sorted(
            sql_server_drivers,
            key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0,
        )[-1]

    def _set_connection_string(
        self,
    ) -> str:  # sourcery skip: avoid-builtin-shadow

        driver = self._get_driver()
        if not driver:
            Status.NOT_FOUND("No SQL Server driver found")
            return None

        if not self.Server or not self.Database:
            Status.NOT_FOUND("Server or Database is not set")
            return None

        conn = f"DRIVER={{{driver}}};SERVER={self.Server};DATABASE={self.Database};"

        if self.Trusted:
            conn += ";Trusted_Connection=yes"
        else:
            conn += f"UID={self.Username};PWD={self.Password.get_secret_value()};TrustServerCertificate=Yes"

        self._connection_string = conn
        return conn

    def connstring_to_uri(self) -> str:
        """Convert the connection string to a URI format.

        Returns:
            str: The connection string in URI format.
        """
        self._set_connection_string()
        if not self._connection_string:
            Status.NOT_FOUND("Connection string is not set", self)
            return ""

        if not self.Trusted:
            uri = f"mssql+pyodbc://{quote_plus(self.Username)}:{quote_plus(self.Password.get_secret_value())}@{quote_plus(self.Server)}/{quote_plus(self.Database)}?driver={quote_plus(self._get_driver())}&TrustServerCertificate=Yes"
        else:
            uri = f"mssql+pyodbc://{quote_plus(self.Server)}/{quote_plus(self.Database)}?driver={quote_plus(self._get_driver())}&Trusted_Connection=yes"
        return uri

    def engine(self, timeout_hrs: int = 1):
        """This function returns the engine

        Returns:
            engine: engine
        """
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return None

        timeout = int(timedelta(hours=timeout_hrs).total_seconds())
        conn_str = self.connstring_to_uri()
        return create_engine(
            conn_str,
            pool_size=5,  # Max 5 persistent connections
            max_overflow=5,  # Allow up to 5 additional connections temporarily
            pool_timeout=15,  # Wait up to 30 seconds for a connection
            pool_recycle=300,  # Recycle connections after timeout
            connect_args={
                "timeout": timeout,
                "fast_executemany": True,
                "autocommit": True,
            },  # For MSSQL optimization
            pool_pre_ping=True,
            pool_reset_on_return="rollback",  # Reset state to avoid stale connections
        )

    def download_table_or_query(
        self, table_name: str = None, query: str = None, demo=False, chunksize=150000
    ) -> Optional[dd.DataFrame]:
        """
        Downloads data from a database table or executes a SQL query and returns the result as a Dask DataFrame.
        Args:
        ----
            table_name (str, optional): Name of the database table to download. Defaults to None.
            query (str, optional): SQL query to execute. Defaults to None.
            demo (bool, optional): If True, uses demo mode for downloading data. Defaults to False.
            chunksize (int, optional): Number of rows per chunk to download. Defaults to 150000.

        Returns:
            Optional[dd.DataFrame]: A Dask DataFrame containing the downloaded data, or None if an error occurs or neither table_name nor query is provided.

        Raises:
            Status.FAILED: If neither table_name nor query is provided.
            Status.WARNING: If a database error occurs during download.
        """
        # either table name or query should be provided
        if table_name is None and query is None:
            Status.FAILED("Please provide either table name or query")
            return None

        try:
            return self._download_table_or_query(demo, table_name, query, chunksize)
        except pyodbc.ProgrammingError as _e:
            Status.WARNING("Can not download data", self, error=_e)
        except BaseException as _e:
            Status.WARNING("Can not download data", self, error=_e)
        return None

    def _download_table_or_query(
        self, demo, table_name, query, chunksize
    ) -> Optional[dd.DataFrame]:
        # Check if connection to DB is working
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return None

        # if demo is True, download only 1000 rows
        if demo:
            sample_size = 1000
            query = (
                f"SELECT TOP {sample_size} * FROM {table_name}" if table_name else query
            )

        # Download data from database in chunks
        start_time = time.time()
        data = None
        chunksize = int(chunksize)
        if query:
            Status.INFO(
                "Downloading data from query",
                self,
                query=query,
                chunksize=chunksize,
            )
            data = pd.read_sql(query, self.engine(), chunksize=chunksize)
        elif table_name:
            # check if table exists
            if not self.does_table_exist(table_name):
                Status.FAILED(f"Table {table_name} does not exist", self)
                return None

            Status.INFO(
                "Downloading data from table",
                self,
                table_name=table_name,
                chunksize=chunksize,
            )
            data = pd.read_sql(
                f"SELECT * FROM {table_name}",
                self.engine(),
                chunksize=chunksize,
            )

        # check if there is data
        if data is None:
            Status.FAILED("No data retrieved from the database", self)
            return None

        # get new directory
        directory_name, directory_path = FileHandler().get_new_directory()

        if directory_name:
            total_rows = 0
            total_cols = 0
            total_size = 0
            # Write data to parquet file in chunks
            for i, chunk in enumerate(data):
                time_elapsed = time.time() - start_time
                # check if time elapsed is more than 5 minutes
                # and chunk size is more than 1000
                if i == 0 and time_elapsed > 300 and chunksize > 1000:
                    Status.WARNING(
                        "Data download took more than 5 minutes. trying with smaller chunk size",
                        self,
                    )
                    chunksize = chunksize // 1.25
                    # download data again with smaller chunk size
                    return self.download_table_or_query(
                        table_name=table_name,
                        query=query,
                        demo=demo,
                        chunksize=chunksize,
                    )

                start_index = total_rows + 1
                stop_index = total_rows + len(chunk.index) + 1
                chunk.index = range(start_index, stop_index)
                total_size += chunk.memory_usage(deep=True).sum()
                chunk.to_parquet(
                    f"{directory_path}/{directory_name}_{i}.parquet",
                    engine="pyarrow",
                    index=False,
                )
                total_rows += len(chunk.index)
                total_cols = max(total_cols, len(chunk.columns))
                Status.INFO("Downloading chunk", chunk=i + 1)

        if not total_rows:
            Status.WARNING("No data downloaded", self)
            return None

        Status.INFO(
            "Data downloaded",
            self,
            rows=total_rows,
            columns=total_cols,
            size=humanize.naturalsize(total_size),
            path=directory_path,
        )

        # Repartition data
        df = dd.read_parquet(directory_path)
        df = infer_dtypes(repartition_data(df, verbose=True))

        # Store the path in DASK_KEY for later use
        setattr(df, DASK_KEY, directory_path)
        return df

    def upload_table(self, data_file_path: str, table_name: str) -> bool:
        """
        Uploads data from a Parquet file to a specified table in the database.
        Parameters:
            data_file_path (str): The file path to the Parquet data file to be uploaded.
            table_name (str): The full name of the target table in the format 'schema.table'.
        Returns:
            bool: True if the upload is successful, False otherwise.
        Raises:
            ValueError: If either `data_file_path` or `table_name` is not provided.
        Process:
            - Validates input parameters.
            - Reads the Parquet file into a Dask DataFrame.
            - Checks if the database connection is active.
            - Constructs the database URI and connection parameters.
            - Uploads the data to the specified table, replacing it if it exists.
            - Logs status messages for success or failure.
        """

        # either table name or data_file_path should be provided
        if table_name is None or data_file_path is None:
            Status.FAILED("Please provide table name and data file path to write to DB")
            raise ValueError

        chunksize = 1000
        # Read Parquet file into a DataFrame
        odf = dd.read_parquet(data_file_path)

        try:
            # Check if connection to DB is working
            if self.is_db_connected:
                params = urllib.parse.quote_plus(
                    urllib.parse.quote_plus(self._connection_string)
                )
                uri = f"mssql+pyodbc:///?odbc_connect={params}"
                schema = table_name.split(".")[0]
                table_name = table_name.split(".")[1]
                # Inserting data into table
                Status.INFO(
                    "Uploading data to table", table_name=f"{schema}.{table_name}"
                )

                timeout = int(timedelta(hours=3).total_seconds())
                engine_kwargs = {
                    "pool_size": 5,  # Max 5 persistent connections
                    "max_overflow": 5,  # Allow up to 5 additional connections temporarily
                    "pool_timeout": 30,  # Wait up to 30 seconds for a connection
                    "pool_recycle": timeout,  # Recycle connections after timeout
                    "connect_args": {"timeout": timeout},
                }
                odf.to_sql(
                    table_name,
                    uri=uri,
                    schema=schema,
                    if_exists="replace",
                    index=True,
                    compute=True,
                    chunksize=chunksize,
                    engine_kwargs=engine_kwargs,
                )

            Status.INFO(
                "Data uploaded to table", self, table_name=f"{schema}.{table_name}"
            )
            return True

        except BaseException as _e:
            Status.FAILED(
                "Can not upload data to table",
                self,
                table_name=f"{schema}.{table_name}",
                error=_e,
                traceback=False,
            )
        return False

    def execute_query(self, query: str, timeout_hrs: int = 1) -> bool:
        """
        Executes a given SQL query on the database.
        Parameters:
        ---------
        query (str): The SQL query to be executed.
        timeout_hrs (int, optional): The timeout for the query execution in hours. Defaults to 1 hour.

        Returns:
        bool: True if the query executed successfully, False otherwise.

        Raises:
        SQLAlchemyError: If there is an error during SQL execution.
        Exception: If there is an unexpected error during query execution.

        Notes:
        - Checks if the database connection is active before executing the query.
        - Uses SQLAlchemy to manage the database connection and execute the query.
        - Logs the status of the query execution using the Status class.
        """
        # Check if connection to DB is working
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return False

        Status.INFO("Executing SQL query")
        with self.engine(timeout_hrs=timeout_hrs).connect() as conn:
            try:
                conn = conn.execution_options(stream_results=True)
                with conn.begin():
                    result = conn.execute(text(query))
                    row_count = result.rowcount
                    if row_count == -1:
                        Status.INFO("Query executed successfully.", self)
                    else:
                        Status.INFO(
                            f"Query executed successfully. {row_count} rows affected.",
                            self,
                        )
                    return True
            except SQLAlchemyError as e:
                # check if error has an orig attribute
                if hasattr(e, "orig"):
                    Status.FAILED(
                        "SQL execution failed.",
                        self,
                        error=str(e.orig),
                        traceback=False,
                    )
                else:
                    Status.FAILED(
                        "SQL execution failed.", self, error=str(e), traceback=False
                    )
            except Exception as e:
                Status.FAILED(
                    f"Query execution failed. Unexpected error. {e}",
                    self,
                    traceback=False,
                )
        return False

    def execute_procedure(self, procedure_name: str, timeout_hrs: int = 1) -> bool:
        """
        Executes a stored procedure in the database.

        Args:
        ----
            procedure_name (str): The name of the stored procedure to execute.
            timeout_hrs (int, optional): The timeout for the procedure execution in hours. Defaults to 1 hour.

        Returns:
            bool: True if the procedure executed successfully, False otherwise.
        """
        query = f"EXEC {procedure_name}"
        return self.execute_query(query, timeout_hrs=timeout_hrs)

    def does_table_exist(self, table_name: str) -> bool:
        """
        Checks if a table exists in the database.
        Args:
        ----
            table_name (str): The name of the table to check. Can include schema name
                              in the format "schema_name.table_name".
        Returns:
            bool: True if the table exists, False otherwise.

        Raises:
            SQLAlchemyError: If there is an error during SQL execution.
            Exception: If an unexpected error occurs during query execution.

        Notes:
            - If the database connection is not established, the method logs a failure
              status and returns False.
            - If the table name includes a schema (e.g., "schema_name.table_name"),
              the schema name is extracted and used in the query.
            - Logs informational messages indicating whether the table exists or not.
            - Logs failure messages in case of SQL execution or unexpected errors.
        """
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return False

        # its quite possible that we pass table name with schema name. validate such cases and extract schema name and table name.
        # Also handle quoted names (with brackets like [Schema].[Table])
        schema_name = None
        if "." in table_name:
            schema_name, table_name = table_name.split(".", 1)

        # Remove brackets from schema and table names if present (SQL Server quoted identifiers)
        # INFORMATION_SCHEMA stores names without brackets
        if schema_name:
            schema_name = schema_name.strip()
            if schema_name.startswith("[") and schema_name.endswith("]"):
                schema_name = schema_name[1:-1]

        table_name = table_name.strip()
        if table_name.startswith("[") and table_name.endswith("]"):
            table_name = table_name[1:-1]

        if schema_name:
            query = f"SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = '{schema_name}'"
        else:
            query = f"SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}'"

        with self.engine().connect() as conn:
            try:
                result = conn.execute(text(query))
                if result.fetchone():
                    Status.INFO(f"Table {table_name} exists.", self)
                    return True

                Status.INFO(f"Table {table_name} does not exist.", self)
                return False
            except SQLAlchemyError as e:
                Status.FAILED(
                    "SQL execution failed.", self, error=str(e), traceback=False
                )
            except Exception as e:
                Status.FAILED(
                    f"Query execution failed. Unexpected error. {e}",
                    self,
                    traceback=False,
                )

        return False

    def get_column_type(self, table_name: str, column_name: str) -> Optional[DataType]:
        """
        Retrieves the data type of a specified column in a given table from the database.
        Args:
            table_name (str): The name of the table, optionally including the schema (e.g., 'schema.table').
            column_name (str): The name of the column whose data type is to be fetched.
        Returns:
            Optional[DataType]: The mapped data type of the specified column if found, otherwise None.
        Logs:
            - INFO: When fetching the data type and executing the query.
            - WARNING: If the specified column is not found in the table.
            - FAILED: If the database is not connected or if query execution fails.
        Raises:
            None: All exceptions are handled internally and logged."""
        Status.INFO(
            f"Fetching data type for column '{column_name}' in table '{table_name}'",
            self,
        )
        # Check if connection to DB is working
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return None

        # remove quotes if they exist
        table_name = (
            table_name.replace("[", "")
            .replace("]", "")
            .replace('"', "")
            .replace("'", "")
        )

        # check if schema name is passed with table name
        if "." in table_name:
            schema_name, table_name = table_name.split(".", 1)
            query = f"""
            SELECT DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{schema_name}' AND TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column_name}'
            """
        else:
            query = f"""
            SELECT DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column_name}'
            """
        Status.INFO(f"Executing query to fetch column type: {query}", self)

        with self.engine().connect() as conn:
            try:
                result = conn.execute(text(query))
                if row := result.fetchone():
                    return self._sql_dtype_mapping(row[0])

                Status.WARNING(
                    f"Column {column_name} not found in table {table_name}.", self
                )
                return None
            except SQLAlchemyError as e:
                Status.FAILED(
                    "SQL execution failed.", self, error=str(e), traceback=False
                )
            except Exception as e:
                Status.FAILED(
                    f"Query execution failed. Unexpected error. {e}",
                    self,
                    traceback=False,
                )

        return None

    def _sql_dtype_mapping(self, sql_dtype: str) -> Optional[DataType]:
        """
        Maps SQL data types to DataType enum.

        Args:
            sql_dtype (str): The SQL data type as a string.

        Returns:
            Optional[DataType]: The corresponding DataType enum value if mapped, None otherwise.
        """
        mapping = {
            "char": DataType.STRING,
            "varchar": DataType.STRING,
            "text": DataType.STRING,
            "nchar": DataType.STRING,
            "nvarchar": DataType.STRING,
            "ntext": DataType.STRING,
            "int": DataType.NUMBER,
            "bigint": DataType.NUMBER,
            "smallint": DataType.NUMBER,
            "tinyint": DataType.NUMBER,
            "decimal": DataType.NUMBER,
            "numeric": DataType.NUMBER,
            "float": DataType.NUMBER,
            "real": DataType.NUMBER,
            "date": DataType.DATE,
            "datetime": DataType.DATETIME,
            "datetime2": DataType.DATETIME,
            "smalldatetime": DataType.DATETIME,
            "time": DataType.DATETIME,
            "bit": DataType.BOOLEAN,
        }
        return mapping.get(sql_dtype.lower())

    def get_table_row_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table using an efficient COUNT query.

        Args:
            table_name (str): Name of the table to count rows for

        Returns:
            int: Number of rows in the table, 0 if error or empty
        """
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return 0

        try:
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"

            with self.engine().connect() as conn:
                result = conn.execute(text(count_query))
                row = result.fetchone()
                count = row[0] if row else 0
                Status.INFO(f"Table '{table_name}' has {count:,} rows")
                return count

        except SQLAlchemyError as e:
            Status.FAILED(f"Failed to get row count for table '{table_name}': {str(e)}")
            return 0
        except Exception as e:
            Status.FAILED(
                f"Unexpected error getting row count for '{table_name}': {str(e)}"
            )
            return 0

    def check_column_exists(self, table_name: str, column_name: str) -> bool:
        """
        Check if a column exists in a specific table.

        Args:
            table_name (str): Name of the table to check
            column_name (str): Name of the column to check for

        Returns:
            bool: True if column exists, False otherwise
        """
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return False

        try:
            # Parse schema if present
            schema_name = None
            if "." in table_name:
                schema_name, table_name = table_name.split(".", 1)

            if schema_name:
                query = """
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND COLUMN_NAME = ?
                """
                params = (schema_name, table_name, column_name)
            else:
                query = """
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ? AND COLUMN_NAME = ?
                """
                params = (table_name, column_name)

            with self.engine().connect() as conn:
                result = conn.execute(text(query), params)
                exists = result.fetchone() is not None
                Status.INFO(
                    f"Column '{column_name}' {'exists' if exists else 'does not exist'} in table '{table_name}'"
                )
                return exists

        except Exception as e:
            Status.FAILED(f"Error checking column existence: {str(e)}")
            return False

    def validate_query_syntax(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query syntax without executing it fully.
        Uses SET PARSEONLY ON to check syntax in SQL Server.

        Args:
            query (str): SQL query to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not self.is_db_connected:
            return False, "Database not connected"

        try:
            # Use SET PARSEONLY ON to validate syntax without execution
            validation_query = f"SET PARSEONLY ON; {query}; SET PARSEONLY OFF;"

            with self.engine().connect() as conn:
                conn.execute(text(validation_query))
                return True, None

        except SQLAlchemyError as e:
            error_msg = str(e)
            Status.WARNING(f"Query syntax validation failed: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during syntax validation: {str(e)}"
            Status.WARNING(error_msg)
            return False, error_msg

    def get_table_columns(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get all columns for a table with their data types.

        Args:
            table_name (str): Name of the table

        Returns:
            List[Dict[str, str]]: List of column info dicts with 'name' and 'type' keys
        """
        if not self.is_db_connected:
            Status.FAILED("DB not connected", self)
            return []

        try:
            # Parse schema if present
            schema_name = None
            if "." in table_name:
                schema_name, table_name = table_name.split(".", 1)

            if schema_name:
                query = """
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
                """
                params = (schema_name, table_name)
            else:
                query = """
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
                """
                params = (table_name,)

            columns = []
            with self.engine().connect() as conn:
                result = conn.execute(text(query), params)
                columns.extend(
                    {"name": row[0], "type": row[1]} for row in result.fetchall()
                )
            Status.INFO(f"Retrieved {len(columns)} columns for table '{table_name}'")
            return columns

        except Exception as e:
            Status.FAILED(
                f"Error retrieving columns for table '{table_name}': {str(e)}"
            )
            return []

    def test_query_performance(self, query: str) -> Tuple[bool, float, Optional[str]]:
        """
        Test query performance by running it with a short timeout.

        Args:
            query (str): SQL query to test

        Returns:
            Tuple[bool, float, Optional[str]]: (success, execution_time, error_message)
        """
        if not self.is_db_connected:
            return False, 0.0, "Database not connected"

        try:
            start_time = time.time()

            # Add a lightweight modification to limit results for testing
            test_query = f"SELECT TOP 1 * FROM ({query}) AS test_query"

            with self.engine().connect() as conn:
                # Set query timeout
                conn = conn.execution_options(autocommit=True)
                result = conn.execute(text(test_query))
                result.fetchone()  # Just fetch one row to test

            execution_time = time.time() - start_time
            Status.SUCCESS(f"Query performance test completed in {execution_time:.2f}s")
            return True, execution_time, None

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = (
                f"Query performance test failed after {execution_time:.2f}s: {str(e)}"
            )
            Status.WARNING(error_msg)
            return False, execution_time, error_msg
