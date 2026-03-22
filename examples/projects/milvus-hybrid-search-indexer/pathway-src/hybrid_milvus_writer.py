# Copyright © 2026 Pathway
#
# Custom Milvus writer for hybrid (dense + sparse) vector collections.
#
# The built-in pw.io.milvus.write() supports a single vector field. This module
# extends the pattern to support collections with multiple vector fields, enabling
# hybrid dense+sparse search.
#
# It follows the _MilvusOutputBuffer pattern from pathway.io.milvus, using
# pw.io.subscribe() for change tracking and buffered batch writes.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from pathway.io._subscribe import subscribe

if TYPE_CHECKING:
    from pathway.internals.api import Pointer
    from pathway.internals.table import Table


class _HybridMilvusOutputBuffer:
    """Buffers row changes and flushes them to a multi-vector Milvus collection."""

    def __init__(
        self,
        uri: str,
        collection_name: str,
        primary_key_column: str,
        dense_vector_column: str,
        sparse_vector_column: str,
        dense_dimension: int,
        token: str | None,
        max_batch_size: int,
    ) -> None:
        from pymilvus import MilvusClient

        self._collection_name = collection_name
        self._primary_key_column = primary_key_column
        self._dense_vector_column = dense_vector_column
        self._sparse_vector_column = sparse_vector_column
        self._max_batch_size = max_batch_size

        connect_kwargs: dict[str, Any] = {"uri": uri}
        if token is not None:
            connect_kwargs["token"] = token
        self._client = MilvusClient(**connect_kwargs)

        self._ensure_collection_exists(dense_dimension)

        self._upsert_buffer: list[dict[str, Any]] = []
        self._delete_buffer: list[Any] = []

    def on_change(
        self, key: Pointer, row: dict[str, Any], time: int, is_addition: bool
    ) -> None:
        if is_addition:
            prepared = self._prepare_row(row)
            self._upsert_buffer.append(prepared)
            if len(self._upsert_buffer) >= self._max_batch_size:
                self._flush_upserts()
        else:
            pk_value = row[self._primary_key_column]
            self._delete_buffer.append(pk_value)
            if len(self._delete_buffer) >= self._max_batch_size:
                self._flush_deletes()

    def on_time_end(self, time: int) -> None:
        # PKs present in both buffers are updates (delete old + insert new),
        # not true deletes. Only delete PKs that are not being re-inserted.
        upserted_pks = {
            row[self._primary_key_column] for row in self._upsert_buffer
        }
        true_deletes = [pk for pk in self._delete_buffer if pk not in upserted_pks]

        if self._upsert_buffer:
            self._flush_upserts()
        if true_deletes:
            self._delete_buffer = true_deletes
            self._flush_deletes()
        else:
            self._delete_buffer = []

    def _prepare_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy arrays to Python lists for pymilvus compatibility."""
        result = {}
        for k, v in row.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        return result

    def _flush_upserts(self) -> None:
        try:
            self._client.upsert(
                collection_name=self._collection_name,
                data=self._upsert_buffer,
            )
        except Exception:
            logging.error(
                "Failed to upsert %d rows into Milvus collection '%s'",
                len(self._upsert_buffer),
                self._collection_name,
                exc_info=True,
            )
            raise
        finally:
            self._upsert_buffer = []

    def _flush_deletes(self) -> None:
        try:
            self._client.delete(
                collection_name=self._collection_name,
                pks=self._delete_buffer,
            )
        except Exception:
            logging.error(
                "Failed to delete %d rows from Milvus collection '%s'",
                len(self._delete_buffer),
                self._collection_name,
                exc_info=True,
            )
            raise
        finally:
            self._delete_buffer = []

    def _ensure_collection_exists(self, dense_dimension: int) -> None:
        from pymilvus import CollectionSchema, DataType, FieldSchema

        if self._client.has_collection(self._collection_name):
            return

        fields = [
            FieldSchema(
                name=self._primary_key_column,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name=self._dense_vector_column,
                dtype=DataType.FLOAT_VECTOR,
                dim=dense_dimension,
            ),
            FieldSchema(
                name=self._sparse_vector_column,
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        self._client.create_collection(
            collection_name=self._collection_name, schema=schema
        )

        # Create indexes for both vector fields
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=self._dense_vector_column,
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name=self._sparse_vector_column,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self._client.create_index(
            collection_name=self._collection_name, index_params=index_params
        )
        self._client.load_collection(self._collection_name)


def write_hybrid(
    table: Table,
    *,
    uri: str = "http://localhost:19530",
    collection_name: str,
    primary_key_column: str,
    dense_vector_column: str,
    sparse_vector_column: str,
    dense_dimension: int,
    token: str | None = None,
    max_batch_size: int = 1024,
) -> None:
    """Write a table with both dense and sparse vector columns to Milvus.

    Creates a collection with two vector fields (FLOAT_VECTOR + SPARSE_FLOAT_VECTOR)
    and indexes for hybrid search. Additional columns beyond the primary key and
    vectors are stored as dynamic fields.

    Args:
        table: The table to output.
        uri: The Milvus server URI.
        collection_name: The name of the Milvus collection.
        primary_key_column: Column to use as Milvus primary key.
        dense_vector_column: Column containing dense vector embeddings (numpy arrays).
        sparse_vector_column: Column containing sparse vectors as {int: float} dicts.
        dense_dimension: Dimensionality of the dense embeddings.
        token: Optional authentication token.
        max_batch_size: Maximum rows to buffer before flushing.
    """
    output_buffer = _HybridMilvusOutputBuffer(
        uri=uri,
        collection_name=collection_name,
        primary_key_column=primary_key_column,
        dense_vector_column=dense_vector_column,
        sparse_vector_column=sparse_vector_column,
        dense_dimension=dense_dimension,
        token=token,
        max_batch_size=max_batch_size,
    )
    subscribe(
        table,
        on_change=output_buffer.on_change,
        on_time_end=output_buffer.on_time_end,
    )
