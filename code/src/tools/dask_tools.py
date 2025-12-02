# # Copyright (C) KonaAI - All Rights Reserved
"""This module contains the dask tools"""
import operator
import re
import shutil
from enum import Enum
from functools import reduce
from itertools import islice
from typing import Any
from typing import Callable
from typing import List
from typing import Type
from typing import TypeVar
from typing import Union

import dask.dataframe as dd
import humanize
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from dask.array import Array
from pandas import Series
from src.utils.file_mgmt import file_handler
from src.utils.status import Status

ArrayLike = TypeVar("ArrayLike", Array, np.ndarray)
SeriesType = Union[dd.Series, Series]


def repartition_data(_df: dd.DataFrame, verbose=False) -> dd.DataFrame:
    """
    Repartitions a Dask DataFrame based on available memory to optimize processing.
    This function calculates the optimal number of partitions for a given Dask DataFrame
    based on the available system memory. It ensures that the data can be processed efficiently
    without exceeding memory limits.
    Parameters:
    -----------
    _df (dd.DataFrame): The Dask DataFrame to be repartitioned.
    Returns:
    dd.DataFrame: The repartitioned Dask DataFrame.

    Notes:
    - The function assumes that each row requires 1.5 times its actual memory usage for processing.
    - It uses 80% of the available system memory for calculations.
    - If the calculated number of partitions is less than or equal to the original number of partitions,
      the original DataFrame is returned without changes.
    """

    if verbose:
        Status.INFO("Starting data repartitioning")

    original_partitions = _df.npartitions

    if original_partitions == 1:
        if verbose:
            Status.INFO("Data partitions already optimized for processing")
        return _df

    # find the total memory required by a single row assuming 1.5 times of single row
    # find the row index with minimum number of missing values
    min_null_row_index = _df.isnull().sum(axis=1).idxmin().compute()
    one_row_size = np.ceil(
        _df.loc[min_null_row_index].memory_usage(deep=True).sum().compute() * 2
    )  # 2 times of single row
    total_memory_required = one_row_size * len(_df)

    # We can assume that we need 1.5 times of single row to house the data in memory during processing.
    # So, we can calculate the number of rows that can be processed in memory at a time.
    # We can use this number to calculate the number of partitions required to process the data.
    memory_available = psutil.virtual_memory().available
    memory_available = memory_available * 0.8  # use 80% of available memory
    total_partitions = int(np.ceil(total_memory_required / memory_available))

    # check if the total partitions is less than the original partitions
    if total_partitions <= original_partitions:
        if verbose:
            Status.INFO("Data partitions already optimized for processing")
        return _df

    if verbose:
        Status.INFO(
            "Optimizing data partitions for processing",
            original_partitions=original_partitions,
            memory_available=humanize.naturalsize(memory_available),
            total_records=humanize.intcomma(len(_df)),
            estimated_memory_required=humanize.naturalsize(total_memory_required),
            new_partitions=total_partitions,
        )

    return _df.repartition(npartitions=total_partitions)


def discretize(s: dd.Series):  # pylint: disable=unused-argument
    """
    Bins a continuous numeric Dask Series using the Freedman-Diaconis rule for bin width selection.
    This function discretizes a Dask Series by:
    - Converting boolean columns to integers.
    - Returning 0 for columns with only one unique value.
    - Returning integer values for binary columns (0 and 1).
    - For continuous numeric columns, calculates the optimal number of bins using the Freedman-Diaconis rule,
        then bins the data using equal-width binning.
    Returns
    -------
    dask.dataframe.Series
            The discretized (binned) Dask Series with integer bin labels.
    """

    def _calculate_bins(s: dd.Series):
        """This function calculates the number of bins for a continuous numeric series"""
        # bin the data using freedman diaconis rule
        # Why Freedman-Diaconis Rule?
        # Handles Skewed Data and Outliers: The rule uses the Interquartile Range (IQR), which makes it less sensitive to extreme values
        # and outliers compared to methods that rely on the standard deviation.
        # Data-Driven Bin Width: It adapts the bin width based on the actual spread of the data, leading to a more informed
        # and potentially more meaningful binning.
        # Suitable for Different Sizes of Datasets: This method can be effectively applied to both small and large datasets.
        iqr = s.quantile(0.75) - s.quantile(0.25)
        if isinstance(iqr, dd.Scalar):
            iqr = iqr.compute()

        n = s.count()
        if isinstance(n, dd.Scalar):
            n = n.compute()

        bin_width = (2 * iqr) / (n ** (1 / 3))
        # check if bin width is a dask scalar
        if isinstance(bin_width, dd.Scalar):
            bin_width = bin_width.compute()

        epsilon = 1e-6
        bin_count = np.ceil((s.max() - s.min() + epsilon) / (bin_width + epsilon)) + 1
        if isinstance(bin_count, dd.Scalar):
            bin_count = bin_count.compute()
        return int(bin_count)

    def _bin_data(s: dd.Series, bins: list):
        """This function bins the data"""
        return pd.cut(s, bins=bins, labels=False, include_lowest=True)

    try:
        if is_boolean(s):
            # convert boolean to integer
            s = s.map_partitions(lambda x: x.astype(int), meta=(s.name, np.int8))
        # bin the data if it is continuous numeric
        if is_numeric(s) and not is_datetime(s):
            # check if there only one unique value
            if s.nunique().compute() == 1:
                return s.map_partitions(lambda x: 0, meta=(s.name, np.int8))

            # check if values are only 0 and 1
            if (
                s.nunique().compute() == 2
                and s.min().compute() == 0
                and s.max().compute() == 1
            ):
                # convert boolean to integer
                return s.map_partitions(lambda x: x.astype(int), meta=(s.name, np.int8))

            # calculate the number of bins
            num_bins = _calculate_bins(s)
            # calculate equal width bin edges
            bin_edges = np.linspace(s.min(), s.max(), num_bins + 1)
            # keep the bin edges as unique values
            bin_edges = np.unique(bin_edges)
            s = s.map_partitions(_bin_data, bins=bin_edges, meta=(s.name, np.int8))
    except BaseException as _e:
        Status.FAILED("Error in discretization", error=str(_e))
    return s


def optimize_dtypes(_df: dd.DataFrame) -> dd.DataFrame:
    """This function optimizes dataframe data types for memory efficiency"""
    int_types = [np.int8, np.int16, np.int32, np.int64]
    float_types = [np.float16, np.float32, np.float64]
    obj_types = ["category", "object", "string", str, object]

    def _convert_bool_to_int(value):
        """
        Convert a boolean value (as bool or text) to an integer (0 for False, 1 for True).
        """
        if pd.isna(value):
            return np.nan
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            if value.lower() == "true":
                return 1
            if value.lower() == "false":
                return 0
        return 0

    def _get_best_int_type(s: dd.Series):
        """This function returns the best integer type"""
        # check if the series has null values
        if s.isnull().any().compute():
            # return best float type since int cannot handle null values efficiently
            return _get_best_float_type(s)

        min_value = s.min().compute()
        max_value = s.max().compute()

        for int_type in int_types:
            if (
                min_value >= np.iinfo(int_type).min
                and max_value <= np.iinfo(int_type).max
            ):
                return int_type
        return s.dtype

    def _get_best_float_type(s: dd.Series):
        """This function returns the best float type"""
        min_value = s.min().compute()
        max_value = s.max().compute()

        for float_type in float_types:
            if (
                min_value >= np.finfo(float_type).min
                and max_value <= np.finfo(float_type).max
            ):
                return float_type
        return s.dtype

    def _get_best_obj_type(s):
        """This function returns the best object type"""
        total_non_null = s.count().compute()
        total_non_null_unique = s.nunique().compute()

        # set category if the unique values are less than 50% of total values
        # else set to pyarrow string type
        return (
            "category"
            if (total_non_null_unique / total_non_null) < 0.5
            else "string[pyarrow]"
        )

    def _optimize_series(s):
        """This function optimizes the data type of a series"""
        if s.dtype == "bool":
            return s.apply(_convert_bool_to_int, meta=(s.name, np.int8))

        old_dtype = s.dtype
        new_dtype = old_dtype
        if s.dtype in int_types:
            new_dtype = _get_best_int_type(s)
        elif s.dtype in float_types:
            new_dtype = _get_best_float_type(s)
        elif s.dtype in obj_types:
            new_dtype = _get_best_obj_type(s)

        return s.name, new_dtype

    try:
        Status.INFO("Starting data type optimization")
        previous_dtypes = _df.dtypes

        # optimize the data types
        new_meta = [_optimize_series(_df[col]) for col in _df.columns]
        _df = _df.astype(dict(new_meta))

        # check if the dtypes have changed
        if previous_dtypes.equals(_df.dtypes):
            Status.INFO("No data type optimization required")
    except BaseException as _e:
        Status.FAILED("Error in data type optimization", error=str(_e))
    return _df


def process_by_column(
    _df: dd.DataFrame, func: Callable, *args, **kwargs
) -> dd.DataFrame:
    """This function applies a function to each column of the dataframe"""
    try:
        if not func:
            raise ValueError("Function not provided")

        delayed_data = [func(_df[col], *args, **kwargs) for col in _df.columns]
        _df = dd.concat(delayed_data, axis=1).set_index(_df.index.name)
    except BaseException as _e:
        Status.FAILED("Error in processing by column", error=str(_e))
    return _df


def fill_na(s: dd.Series, fill_value: Any):
    """This function fills the missing values in a series"""
    if s.dtype == "category":
        # first add the fill value to the category
        s = s.cat.add_categories(fill_value)
    # then fill the missing values
    s = s.fillna(fill_value)
    return s


def is_boolean(s: Union[dd.Series, pd.Series]) -> bool:
    """
    Check if a given Dask or Pandas Series contains boolean values.
    Parameters:
    s (Union[dd.Series, pd.Series]): The input series to check.
    Returns:
    bool: True if the series contains boolean values, False otherwise.
    """

    if isinstance(s, dd.Series):
        s = s.map_partitions(is_boolean).compute()
        # check if there is any true value in the list
        return any(s.values)

    s = s.dropna()
    return pd.api.types.is_bool_dtype(s) or s.dtype == "bool"


def is_string(s: Union[dd.Series, pd.Series]) -> bool:
    """
    Check if the given series is of string type.
    This function determines whether the provided series is a string type by
    checking if it is not numeric, datetime, or boolean.
    Args:
        s (Union[dd.Series, pd.Series]): The series to check. It can be a Dask
        series (dd.Series) or a Pandas series (pd.Series).
    Returns:
        bool: True if the series is of string type, False otherwise.
    """

    # check if the series is numeric or datetime
    return not is_numeric(s) and not is_datetime(s) and not is_boolean(s)


def is_datetime(s: Union[dd.Series, pd.Series]) -> bool:
    """
    Check if a given Dask or Pandas Series contains datetime values.
    Parameters:
    s (Union[dd.Series, pd.Series]): The input series to check.
    Returns:
    bool: True if the series contains datetime values, False otherwise.
    Notes:
    - For Dask Series, the function maps the check across partitions and computes the result.
    - For Pandas Series, the function first drops null values and then checks if the series is of datetime type.
    - If the series is not initially recognized as datetime, it attempts to convert the series to datetime.
    - The function also checks if there are any non-numeric values in the series.
    - If an exception occurs during conversion, it checks if the dtype contains 'date' or 'time'.
    """

    if isinstance(s, dd.Series):
        x = s.map_partitions(is_datetime).compute()
        # check if there is any true value in the list
        return any(x.values)

    # drop all null values
    s = s.dropna()

    # check if the series is datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return True

    if len(s) == 0:
        return False

    try:
        # check if the series can be converted to datetime
        converted = pd.to_datetime(s, format="mixed", errors="coerce")
        return (
            # check if all values are converted to datetime
            converted.notna().all()
            # check if there are any non-numeric values
            and not s.apply(lambda x: str(x).replace(".", "", 1).isdigit()).any()
        )
    except Exception:
        dtype = str(s.dtype).lower()
        return "date" in dtype or "time" in dtype


def is_numeric(s: Union[dd.Series, pd.Series]) -> bool:
    """
    Check if a Dask or Pandas Series contains numeric values.
    Parameters:
    s (Union[dd.Series, pd.Series]): The input series to check.
    Returns:
    bool: True if the series contains numeric values, False otherwise.
    Notes:
    - For Dask Series, the function will map the is_numeric function to each partition and compute the result.
    - For Pandas Series, the function will drop null values and check if the remaining values are numeric.
    - If the series is empty after dropping null values, the function will return False.
    - The function attempts to convert the series to numeric values and will return True if successful, otherwise False.
    """

    if isinstance(s, dd.Series):
        s = s.map_partitions(is_numeric).compute()
        # check if there is any true value in the list
        return any(s.values)

    # drop all null values and check if the remaining values are numeric
    s = s.dropna()

    if pd.api.types.is_numeric_dtype(s):
        return True

    if len(s) == 0:
        return False

    try:
        pd.to_numeric(s, errors="raise")
        return True
    except Exception:
        return False


def infer_dtypes(df: dd.DataFrame) -> dd.DataFrame:
    """
    Infers and assigns appropriate data types to columns in a Dask DataFrame with ambiguous data types.
    Parameters:
    df (dd.DataFrame): The input Dask DataFrame.
    Returns:
    dd.DataFrame: The Dask DataFrame with inferred data types for columns with ambiguous data types.
    Note:
    - The function uses PyArrow for schema inference and Dask for distributed computation.
    """

    class CustomDType(Enum):
        """
        An enumeration representing custom data types for use with Dask and Pandas.
        Attributes:
            NUMERIC: Represents a 64-bit floating point numeric type ("float64").
            STRING: Represents a string type using PyArrow extension ("string[pyarrow]").
            DATETIME: Represents a timestamp with nanosecond precision ("timestamp[ns]").
            BOOLEAN: Represents a boolean type ("bool").
        """

        NUMERIC = "float64"
        STRING = "string[pyarrow]"
        DATETIME = "timestamp[ns]"
        BOOLEAN = "bool"

    def infer_partition(s: pd.Series):
        """
        Infers the custom data type of a pandas Series partition.
        Parameters
        ----------
        s : pd.Series
            The pandas Series whose data type is to be inferred.
        Returns
        -------
        str
            The name of the inferred custom data type. Possible values are:
            - 'STRING' if the series is empty or does not match other types
            - 'BOOLEAN' if the series dtype is boolean
            - 'DATETIME' if the series is recognized as datetime
            - 'NUMERIC' if the series is recognized as numeric
        Notes
        -----
        The function relies on helper functions `is_datetime` and `is_numeric` to determine
        if the series is of datetime or numeric type, respectively. The `CustomDType` enum
        is used to map the inferred type to its string name.
        """

        if len(s) == 0:
            return CustomDType.STRING.name

        # check if series is boolean
        if s.dtype == "bool":
            return CustomDType.BOOLEAN.name

        # check if series is datetime
        if is_datetime(s):
            return CustomDType.DATETIME.name

        return CustomDType.NUMERIC.name if is_numeric(s) else CustomDType.STRING.name

    def infer_customdtypes(s: dd.Series):
        """
        Infers the custom data type of a Dask Series.
        This function maps the `infer_partition` function over the partitions of the input Dask Series `s`,
        computes the results, and determines the custom data type based on the results.
        Parameters:
        s (dd.Series): The input Dask Series to infer the custom data type from.
        Returns:
        CustomDType: The inferred custom data type, which can be one of CustomDType.DATETIME, CustomDType.NUMERIC, or CustomDType.STRING.
        """
        var = s.map_partitions(infer_partition)
        var = var.compute().tolist()
        if CustomDType.DATETIME.name in var:
            return CustomDType.DATETIME.value

        if CustomDType.BOOLEAN.name in var:
            return CustomDType.BOOLEAN.value

        if CustomDType.NUMERIC.name in var:
            return CustomDType.NUMERIC.value

        return CustomDType.STRING.value

    try:
        # if there are only one partition, return the dataframe
        if df.npartitions == 1:
            return df

        # Infer the schema of the input DataFrame
        schema: pa.lib.Schema = pa.Schema.from_pandas(df.head())
        schema_dict = {field.name: str(field.type) for field in schema}
        # null fields are fields with null data type when multiple partitions may have different data types
        null_fields = [field for field, dtype in schema_dict.items() if dtype == "null"]

        # find all fields with object data type
        # object is also a default for ambiguous data types
        # objects fields should be non null fields
        # object fields appear because of dask limitations in inferring data types
        object_fields = [
            col
            for col, dtype in df.dtypes.items()
            if dtype == "object"
            and df[col].dtype != "string"
            and col not in null_fields
        ]

        total_fields = null_fields + object_fields
        if len(total_fields) == 0:
            return df

        Status.INFO(f"Inferring data types for total {len(total_fields)} fields")

        inferred_dtypes = {
            field: infer_customdtypes(df[field]) for field in total_fields
        }
        # update the schema with the inferred dtypes
        df = df.astype(inferred_dtypes)

        Status.INFO(f"Inferred data types: {inferred_dtypes}")
    except BaseException as e:
        Status.FAILED("Error in inferring data types", error=str(e))
    return df


DASK_KEY = "_temp_parquet_path_"


def compute(
    _df: Union[dd.DataFrame, dd.Series],
) -> Union[dd.DataFrame, dd.Series, None]:
    """
    Compute the given Dask DataFrame or Series and return the result.

    This function performs the following steps:
    1. If the input is a Dask Series, it converts it to a DataFrame.
    2. Generates a new file name for storing the DataFrame in Parquet format.
    3. Infers the data types of the DataFrame.
    4. Writes the DataFrame to a Parquet file with PyArrow engine and Snappy compression.
    5. Reads the Parquet file back into a Dask DataFrame and returns it.
    Args:
    -----
        _df (Union[dd.DataFrame, dd.Series]): The input Dask DataFrame or Series to be computed.
    Returns:
        Union[dd.DataFrame, dd.Series, None]: The computed Dask DataFrame or Series, or None if an error occurs.
    """
    try:
        if isinstance(_df, dd.Series):
            _df = _df.to_frame()
            column_name = _df.columns[0]
            return compute(_df)[column_name]

        # Access the temporary file attribute to check if a previous path exists
        previous_path = None
        if hasattr(_df, DASK_KEY):
            previous_path = getattr(_df, DASK_KEY)

        _df = infer_dtypes(_df)

        # dump as a parquet to force compute
        _, path = file_handler.get_new_file_name("parquet")
        _df.to_parquet(path, compute=True, engine="pyarrow", compression="snappy")

        _df = dd.read_parquet(path)
        # set the temporary file path attribute for future reference
        setattr(_df, DASK_KEY, path)
        # read the parquet file back into a Dask DataFrame
        if previous_path:
            shutil.rmtree(previous_path, ignore_errors=True)

        return _df
    except BaseException as e:
        Status.FAILED("Error in computing", error=str(e), traceback=False)
        return None


def shuffle_and_transform(  # pylint: disable=too-many-positional-arguments
    df: dd.DataFrame,
    group_columns: Union[str, List],
    transform_column: str,
    output_column: str,
    output_dtype: Type,
    transform_func: Callable,
    *args,
    **kwargs,
) -> dd.DataFrame:
    """
    Shuffle the DataFrame and apply a transformation function to a specified column, grouping by one or more columns.
    Parameters:
    ----------
    df (dd.DataFrame): The input Dask DataFrame.
    group_columns (Union[str, List]): Column name or list of column names to group by.
    transform_column (str): The name of the column to apply the transformation function to.
    output_column (str): The name of the column to store the transformed values.
    output_dtype (Type): The data type of the output column.
    transform_func (Callable): The transformation function to apply to the transform_column.
    *args: Additional positional arguments to pass to the transformation function.
    **kwargs: Additional keyword arguments to pass to the transformation function.
    Returns:
    dd.DataFrame: The transformed Dask DataFrame with the shuffled data and the new output column.
    """

    # step 1 - if there is an index, reset it
    # We are doing so because we are going to shuffle the data and during the shuffle, we will ignore the index
    if df is None or len(df) == 0:
        Status.FAILED("Dataframe is empty")
        return df

    index_name = df.index.name
    df = df.reset_index(drop=False) if index_name else df.reset_index(drop=True)
    # step 2 - Shuffle the data so that each group is in same partition
    df = df.shuffle(on=group_columns, ignore_index=True, compute=True).reset_index(
        drop=True
    )

    # step 3 - Apply the transformation function to the data
    df[output_column] = df.map_partitions(
        lambda partition: partition.groupby(group_columns)[transform_column].transform(
            transform_func, *args, **kwargs
        ),
        meta=(output_column, output_dtype),
    ).reset_index(drop=True)
    df[output_column] = df[output_column].astype(output_dtype)

    # step 4 - Reset the index if it was reset in step 1
    if index_name:
        df = df.set_index(index_name)

    return compute(df)


def validate_column(column_name: str, df: dd.DataFrame) -> Union[str, None]:
    """
    Validates if a given column name exists in the Dask DataFrame, either as an index or a column.
    Args:
    ----
        column_name (str): The name of the column to validate.
        df (dd.DataFrame): The Dask DataFrame in which to validate the column name.

    Returns:
        Union[str, None]: The actual column name if it exists in the DataFrame, either as an index or a column,
                          otherwise None.
    """
    if df.index.name and column_name.lower() == df.index.name.lower():
        return df.index.name

    if match_cols := [col for col in df.columns if column_name.lower() == col.lower()]:
        return match_cols[0]

    return None


def filter_by_substring(
    ddf: dd.DataFrame, text_column: str, substrings: List[str]
) -> dd.DataFrame:
    """
    Filters a Dask DataFrame by checking if any of the specified substrings are present in a given text column.
    This function efficiently handles large lists of substrings by processing them in chunks to avoid exceeding regex or memory limits.
    Args:
    ----
        ddf (dd.DataFrame): The Dask DataFrame to filter.
        text_column (str): The name of the column in which to search for substrings.
        substrings (List[str]): A list of substrings to search for within the text column.

    Returns:
        dd.DataFrame: A Dask DataFrame containing only the rows where the text column contains at least one of the specified substrings (case-insensitive).
    """
    chunk_size = max(500, len(substrings) // 10)  # Adjust chunk size as needed

    def chunk_iterable(iterable, size):
        """Yield successive chunks from iterable."""
        it = iter(iterable)
        return iter(lambda: tuple(islice(it, size)), ())

    string_chunks = list(chunk_iterable(substrings, chunk_size))

    masks = []
    for chunk in string_chunks:
        # Create a regex pattern for the chunk
        pattern = "|".join([re.escape(domain) for domain in chunk])
        mask = ddf[text_column].str.contains(pattern, case=False, na=False)
        masks.append(mask)

    final_mask = reduce(operator.or_, masks)
    filtered_ddf = ddf[final_mask]
    return compute(filtered_ddf)
