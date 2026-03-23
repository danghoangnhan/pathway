# Copyright © 2026 Pathway

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Iterable

from pathway.internals.expression import ColumnReference

if TYPE_CHECKING:
    from pathway.internals.table import Table


class MilvusType(enum.IntEnum):
    """Milvus field data types for use with :py:func:`pw.io.milvus.write`.

    Pathway-native Milvus type definitions so that users do not need to
    import pymilvus directly.

    Vector types (for ``vector_columns``):
        ``BINARY_VECTOR``, ``FLOAT_VECTOR``, ``FLOAT16_VECTOR``,
        ``BFLOAT16_VECTOR``, ``SPARSE_FLOAT_VECTOR``

    Scalar types:
        ``BOOL``, ``INT8``, ``INT16``, ``INT32``, ``INT64``,
        ``FLOAT``, ``DOUBLE``, ``VARCHAR``, ``ARRAY``, ``JSON``
    """

    # Scalar types
    BOOL = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5
    FLOAT = 10
    DOUBLE = 11
    VARCHAR = 21
    ARRAY = 22
    JSON = 23

    # Vector types
    BINARY_VECTOR = 100
    FLOAT_VECTOR = 101
    FLOAT16_VECTOR = 102
    BFLOAT16_VECTOR = 103
    SPARSE_FLOAT_VECTOR = 104


def write(
    table: Table,
    *,
    uri: str = "http://localhost:19530",
    collection_name: str,
    primary_key_column: str,
    vector_columns: dict[str, dict[str, Any]],
    token: str | None = None,
    create_collection_if_missing: bool = True,
    max_batch_size: int = 1024,
    name: str | None = None,
    sort_by: Iterable[ColumnReference] | None = None,
) -> None:
    """Writes ``table``'s data into a Milvus collection. Rows are upserted by
    primary key; deletions in the Pathway table are propagated as deletes in Milvus.

    The connector maintains the current state of the Pathway table in the target
    Milvus collection. Row updates in Pathway are represented as a delete followed
    by an insert within the same minibatch; the connector handles this correctly
    by using upsert operations and filtering out spurious deletes.

    If the target collection does not exist and ``create_collection_if_missing`` is
    ``True`` (the default), the connector will create it automatically using the
    vector field definitions in ``vector_columns``. Additional columns are stored
    as dynamic fields (requires Milvus 2.4+).

    Supports all Milvus vector types (``FLOAT_VECTOR``, ``FLOAT16_VECTOR``,
    ``BFLOAT16_VECTOR``, ``INT8_VECTOR``, ``BINARY_VECTOR``,
    ``SPARSE_FLOAT_VECTOR``) and up to 10 vector fields per collection.

    Args:
        table: The table to output.
        uri: The Milvus server URI. Defaults to ``"http://localhost:19530"`` for a
            local Milvus standalone instance. Use ``"https://..."`` for Zilliz Cloud
            or ``"./local.db"`` for Milvus Lite (embedded, useful for testing).
        collection_name: The name of the Milvus collection to write to.
        primary_key_column: The name of the Pathway column to use as the Milvus
            primary key. Must be of ``int`` or ``str`` type.
        vector_columns: A dict mapping Pathway column names to their Milvus vector
            field configuration. Each value is a dict with:

            - ``type`` (required): a :py:class:`MilvusType` vector variant
            - ``dimension`` (required for dense/binary types, omitted for sparse)
            - ``index_type`` (optional, default ``"AUTOINDEX"``)
            - ``metric_type`` (optional, default ``"IP"``)
        token: Optional authentication token. Required for Zilliz Cloud and
            authenticated Milvus deployments.
        create_collection_if_missing: If ``True`` (the default), automatically create
            the target collection if it does not exist. If ``False``, an error is
            raised when writing to a non-existent collection.
        max_batch_size: The maximum number of rows to buffer before flushing to Milvus.
            Rows are also flushed at the end of each minibatch regardless of buffer size.
        name: A unique name for the connector. If provided, this name will be used in
            logs and monitoring dashboards.
        sort_by: If specified, the output will be sorted in ascending order based on the
            values of the given columns within each minibatch. When multiple columns are
            provided, the corresponding value tuples will be compared lexicographically.

    Returns:
        None

    Example:

    To get started, you need a running Milvus instance. The easiest way to set one up
    locally is with Docker:

    .. code-block:: bash

        docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest milvus run standalone

    Single-vector usage:

    >>> import pathway as pw  # doctest: +SKIP
    >>> table = pw.debug.table_from_markdown(  # doctest: +SKIP
    ...     '''
    ...     | doc_id | text        | vector
    ...     1 | 1      | "hello"     | [0.1, 0.2, 0.3]
    ...     2 | 2      | "world"     | [0.4, 0.5, 0.6]
    ...     '''
    ... )
    >>> pw.io.milvus.write(  # doctest: +SKIP
    ...     table,
    ...     collection_name="documents",
    ...     primary_key_column="doc_id",
    ...     vector_columns={
    ...         "vector": {"type": pw.io.milvus.MilvusType.FLOAT_VECTOR, "dimension": 3},
    ...     },
    ... )
    >>> pw.run()  # doctest: +SKIP

    Multi-vector (dense + sparse) for hybrid search:

    >>> import pathway as pw  # doctest: +SKIP
    >>> table = pw.debug.table_from_markdown(  # doctest: +SKIP
    ...     '''
    ...     | doc_id | text    | dense_vector     | sparse_vector
    ...     1 | 1      | "hello" | [0.1, 0.2, 0.3]  | {0: 1.0, 5: 0.5}
    ...     2 | 2      | "world" | [0.4, 0.5, 0.6]  | {1: 0.8, 3: 0.2}
    ...     '''
    ... )
    >>> pw.io.milvus.write(  # doctest: +SKIP
    ...     table,
    ...     collection_name="hybrid_docs",
    ...     primary_key_column="doc_id",
    ...     vector_columns={
    ...         "dense_vector": {"type": pw.io.milvus.MilvusType.FLOAT_VECTOR, "dimension": 3},
    ...         "sparse_vector": {"type": pw.io.milvus.MilvusType.SPARSE_FLOAT_VECTOR},
    ...     },
    ... )
    >>> pw.run()  # doctest: +SKIP
    """
    if not vector_columns:
        raise ValueError("vector_columns must contain at least one vector field.")

    column_names = table.schema.column_names()
    if primary_key_column not in column_names:
        raise ValueError(
            f"Column '{primary_key_column}' not found in table schema. "
            f"Available columns: {column_names}"
        )
    for col_name, config in vector_columns.items():
        if col_name not in column_names:
            raise ValueError(
                f"Vector column '{col_name}' not found in table schema. "
                f"Available columns: {column_names}"
            )
        if "type" not in config:
            raise ValueError(
                f"Vector column '{col_name}' is missing required 'type' key. "
                f"Provide a pw.io.milvus.MilvusType vector variant."
            )

    # Convert vector_columns to string-only dicts for the Rust bridge
    vector_columns_str: dict[str, dict[str, str]] = {}
    for col_name, config in vector_columns.items():
        col_config: dict[str, str] = {}
        milvus_type = config["type"]
        if isinstance(milvus_type, MilvusType):
            col_config["type"] = milvus_type.name
        else:
            col_config["type"] = str(milvus_type)
        if "dimension" in config:
            col_config["dimension"] = str(config["dimension"])
        if "index_type" in config:
            col_config["index_type"] = str(config["index_type"])
        if "metric_type" in config:
            col_config["metric_type"] = str(config["metric_type"])
        vector_columns_str[col_name] = col_config

    import pathway.engine as api
    from pathway.internals import datasink
    from pathway.internals._io_helpers import _format_output_value_fields

    data_storage = api.DataStorage(
        storage_type="milvus",
        milvus_params=api.MilvusParams(
            uri=uri,
            collection_name=collection_name,
            primary_key_column=primary_key_column,
            vector_columns=vector_columns_str,
            token=token,
            create_collection_if_missing=create_collection_if_missing,
        ),
        max_batch_size=max_batch_size,
    )
    data_format = api.DataFormat(
        format_type="identity",
        value_fields=_format_output_value_fields(table),
    )
    table.to(
        datasink.GenericDataSink(
            data_storage,
            data_format,
            datasink_name="milvus",
            unique_name=name,
            sort_by=sort_by,
        )
    )
