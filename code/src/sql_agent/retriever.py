# # Copyright (C) KonaAI - All Rights Reserved
"""SQLAgent Retriever Module"""
from typing import List
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from src.sql_agent.data_dictionary import SQLDataDictionary
from src.sql_agent.embedding import TextEmbedder
from src.utils.database_config import SQLDatabaseManager
from src.utils.status import Status



class SQLColumn:
    """SQLColumn is a class representing a column in the SQL data dictionary."""

    name: str = None
    meta: str = ""
    score: float = 0.0

    def __init__(self, name: str, meta: str, score: float) -> None:
        self.name = name
        self.meta = meta
        self.score = score


def get_data_dictionary(
    sql_table_name: str, db: SQLDatabaseManager
) -> Optional[pd.DataFrame]:
    """
    Retrieve the data dictionary (table schema) for a SQL table referenced by name.
    This function expects a fully qualified table name in the format "schema.table_name".
    It logs a retrieval message, parses the provided table identifier into schema and
    table name, constructs an SQLDataDictionary for the target table, and returns the
    schema as a pandas DataFrame.
    Args:
        sql_table_name (str): Fully qualified SQL table name in the form "schema.table_name".
        db (SQLDatabaseManager): Database manager/connection wrapper used to query metadata.
    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame representing the table schema/data dictionary.
            The DataFrame will be non-empty on success.
    Raises:
        ValueError: If sql_table_name is missing or not in the expected "schema.table_name" format.
        LookupError: If no schema is found for the specified table (i.e., the retrieved schema is None or empty).
    Notes:
        - The function logs an informational message via Status.INFO when invoked.
        - It constructs an SQLDataDictionary using the parsed schema and table name and invokes
          its get_schema() method to obtain the schema DataFrame.
    """
    Status.INFO(
        "Retrieving data dictionary for the SQL table associated with the conversation."
    )

    schema, table_name = (
        sql_table_name.split(".", 1) if sql_table_name else (None, None)
    )
    if not schema or not table_name:
        raise ValueError("Invalid SQL table name format. Expected 'schema.table_name'.")

    dd = SQLDataDictionary(table_schema=schema, table_name=table_name, db=db)
    schema: pd.DataFrame = dd.get_schema()
    if schema is None or schema.empty:
        raise LookupError(f"No schema found for the specified table: {sql_table_name}.")

    return schema


def semantic_search(
    user_prompt: str, data_dictionary: pd.DataFrame, k: int = 10
) -> Optional[List[SQLColumn]]:
    """
    Perform a semantic search over a data dictionary to retrieve the most relevant columns
    for a given natural-language user prompt.
    This function embeds the textual column descriptions from the provided data dictionary,
    builds a Faiss inner-product (cosine) index over the normalized embeddings, embeds the
    user prompt, and retrieves the top-k columns by similarity. Returned columns are
    wrapped as SQLColumn objects (containing name, meta and score) and sorted in descending
    order by similarity score.
    Parameters
    ----------
    user_prompt : str
        The natural-language query for which relevant columns are to be retrieved.
    data_dictionary : pandas.DataFrame
        A dataframe representing the data dictionary. Must contain a description column
        (name defined by SQLDataDictionary.DDMetadata.DESCRIPTION) that holds textual
        descriptions for each column. May optionally contain an exclusion column
        (name defined by SQLDataDictionary.DDMetadata.EXCLUDE) to filter out rows where
        exclusion is True.
    k : int, optional (default=10)
        The maximum number of similar columns to return.
    Returns
    -------
    Optional[List[SQLColumn]]
        A list of SQLColumn objects (name, meta, score) sorted by score in descending
        order. Each SQLColumn corresponds to a row in the filtered data dictionary whose
        description was embedded and indexed. The function may raise on failure rather
        than returning None in typical error cases.
    Raises
    ------
    LookupError
        - If the provided data_dictionary is None or empty.
        - If the required description column is not present.
        - If no results are found for the user prompt (e.g., Faiss returns empty results).
    ValueError
        - If no valid (non-empty, non-NaN) descriptions exist in the data dictionary.
    SystemError
        - If embeddings for the data dictionary could not be created.
        - If the embedding for the user prompt could not be created.
        - If the Faiss index contains no embeddings after adding vectors.
    Notes
    -----
    - The function uses a TextEmbedder to convert text to vector embeddings. The description
      texts are embedded in batch, normalized with faiss.normalize_L2, and indexed using
      faiss.IndexFlatIP (inner product on L2-normalized vectors approximates cosine
      similarity).
    - If an exclusion column exists, rows where that column is truthy are filtered out.
    - The mapping between Faiss result indices and the returned SQLColumn objects is done
      against the filtered dataframe (i.e., indices refer to filtered_schema.iloc).
    - The user prompt embedding is reshaped to a single-query 2D array before searching.
    - Status.INFO logging calls are emitted to indicate progress and results.
    """
    Status.INFO(
        "Retrieving relevant columns from the data dictionary based on user prompt."
    )
    # Retrieve the data dictionary schema
    if data_dictionary is None or data_dictionary.empty:
        raise LookupError("Schema retrieval failed.")

    desc_column = SQLDataDictionary.DDMetadata.DESCRIPTION.value.strip()
    if desc_column not in data_dictionary.columns:
        raise LookupError(
            f"Description column '{desc_column}' not found in the schema."
        )

    # filter out empty descriptions
    filtered_schema = data_dictionary[
        data_dictionary[desc_column].notna() & (data_dictionary[desc_column] != "")
    ]
    if filtered_schema.empty:
        raise ValueError("No valid descriptions found in the schema.")

    # filter out columns with exclusion set to True
    exclude_column = SQLDataDictionary.DDMetadata.EXCLUDE.value.strip()
    if exclude_column in filtered_schema.columns:
        filtered_schema = filtered_schema[
            ~filtered_schema[exclude_column].fillna(False).astype(bool)
        ]

    descs = filtered_schema[desc_column].tolist()
    if not descs:
        raise ValueError("No descriptions found in the schema.")

    # create embeddings
    embedder = TextEmbedder()
    col_embeddings = embedder.embed(descs)
    if col_embeddings is None or len(col_embeddings) == 0:
        raise SystemError("No embeddings created for the data dictionary.")

    # normalize the embeddings for cosine similarity
    faiss.normalize_L2(col_embeddings)

    # build a Faiss index (Flat IP = inner-product for cosine vectors are L2-normed)
    dim = col_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(
        col_embeddings.astype(np.float32)
    )  # now index.ntotal is the number of columns
    if index.ntotal == 0:
        raise SystemError("No embeddings found in the index.")

    # embed the user prompt
    q_emb = embedder.embed(user_prompt)
    if q_emb is None or len(q_emb) == 0:
        raise SystemError("No embedding created for the user prompt.")

    # reshape the query embedding to match the index input shape
    q_emb = np.array(q_emb)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    elif q_emb.ndim == 2 and q_emb.shape[0] != 1:
        q_emb = q_emb[:1, :]
    q_emb = q_emb.astype(np.float32)

    # perform the search
    D, I = index.search(q_emb, k)  # D: distances, I: indices # noqa: E741
    if D is None or I is None or len(D) == 0 or len(I) == 0:
        raise LookupError("No results found for the user prompt.")

    results: List[SQLColumn] = []
    results.extend(
        SQLColumn(
            name=filtered_schema.iloc[idx][
                SQLDataDictionary.DDMetadata.COLUMN_NAME.value.strip()
            ],
            meta=_create_column_meta(filtered_schema.iloc[idx]),
            score=score,
        )
        for score, idx in zip(D[0], I[0])
        if 0 <= idx < len(filtered_schema)
    )
    if not results:
        raise LookupError("No relevant columns found for the user prompt.")

    # sort results by score in descending order
    results.sort(key=lambda x: x.score, reverse=True)
    Status.INFO(
        f"Retrieved {len(results)} relevant columns for the user prompt.",
        columns=[col.name for col in results],
    )

    return results


def _create_column_meta(row: pd.Series) -> str:
    """
    Generates a formatted metadata string for a database column based on a pandas Series row.
    The metadata string includes the column name, tags (such as primary key, uniqueness, and data type),
    and a description. If the description is missing or NaN, a default message is used.
    Args:
        row (pd.Series): A pandas Series representing a row from the data dictionary, containing
            metadata fields such as column name, description, primary key status, uniqueness, and data type.
    Returns:
        str: A formatted string describing the column, including its name, tags, and description.
    """
    # fill NaN values in the description column with an empty string
    desc = row[SQLDataDictionary.DDMetadata.DESCRIPTION.value.strip()]
    desc = (
        str(desc).replace("\n", " ").replace("<NA>", "").strip()
        or "No description available"
    )

    tags = []
    # check if the column is a primary key
    if str(row[SQLDataDictionary.DDMetadata.IS_PRIMARY_KEY.value]).lower() == "yes":
        tags.append("PK")

    # check if the column is unique
    if str(row[SQLDataDictionary.DDMetadata.IS_UNIQUE.value]).lower() == "yes":
        tags.append("UNIQUE")

    # add data type to tags
    tags.append(row[SQLDataDictionary.DDMetadata.DATA_TYPE.value.strip()])

    # create the tag string
    # if tags are present, format them as (tag1, tag2, ...)
    tag_str = "(" + ", ".join(tags) + ")" if tags else ""

    # create the column metadata string
    return f"{row[SQLDataDictionary.DDMetadata.COLUMN_NAME.value.strip()]} {tag_str}: {desc}"
