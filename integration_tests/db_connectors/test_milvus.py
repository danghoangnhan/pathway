import json
import pathlib
import uuid

from pymilvus import MilvusClient

import pathway as pw
from pathway.internals.parse_graph import G


MILVUS_URI = "./test_milvus.db"
DIMENSION = 3


def _generate_collection_name() -> str:
    return f"test_{uuid.uuid4().hex[:12]}"


def _write_and_query(
    *,
    test_items: list[dict],
    input_path: pathlib.Path,
    schema: type[pw.Schema],
    collection_name: str,
    primary_key_column: str,
    vector_column: str,
) -> list[dict]:
    """Write items via the Milvus connector and query results back."""
    G.clear()
    with open(input_path, "w") as f:
        for item in test_items:
            f.write(json.dumps(item) + "\n")

    table = pw.io.jsonlines.read(input_path, schema=schema, mode="static")
    pw.io.milvus.write(
        table,
        uri=MILVUS_URI,
        collection_name=collection_name,
        primary_key_column=primary_key_column,
        vector_column=vector_column,
        dimension=DIMENSION,
    )
    pw.run()

    client = MilvusClient(uri=MILVUS_URI)
    results = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=list(schema.column_names()),
        limit=1000,
    )
    client.close()
    return results


def test_milvus_basic_write(tmp_path):
    class InputSchema(pw.Schema):
        doc_id: int
        text: str
        vector: list[float]

    input_path = tmp_path / "input.txt"
    collection_name = _generate_collection_name()

    test_items = [
        {"doc_id": 1, "text": "hello", "vector": [0.1, 0.2, 0.3]},
        {"doc_id": 2, "text": "world", "vector": [0.4, 0.5, 0.6]},
    ]

    results = _write_and_query(
        test_items=test_items,
        input_path=input_path,
        schema=InputSchema,
        collection_name=collection_name,
        primary_key_column="doc_id",
        vector_column="vector",
    )

    results.sort(key=lambda x: x["doc_id"])
    assert len(results) == 2
    assert results[0]["doc_id"] == 1
    assert results[0]["text"] == "hello"
    assert results[1]["doc_id"] == 2
    assert results[1]["text"] == "world"


def test_milvus_upsert_on_update(tmp_path):
    class InputSchema(pw.Schema):
        doc_id: int
        text: str
        vector: list[float]

    input_path = tmp_path / "input.txt"
    collection_name = _generate_collection_name()

    # First write
    test_items = [
        {"doc_id": 1, "text": "hello", "vector": [0.1, 0.2, 0.3]},
        {"doc_id": 2, "text": "world", "vector": [0.4, 0.5, 0.6]},
    ]
    _write_and_query(
        test_items=test_items,
        input_path=input_path,
        schema=InputSchema,
        collection_name=collection_name,
        primary_key_column="doc_id",
        vector_column="vector",
    )

    # Second write with updated text for doc_id=1
    updated_items = [
        {"doc_id": 1, "text": "hello updated", "vector": [0.1, 0.2, 0.3]},
        {"doc_id": 2, "text": "world", "vector": [0.4, 0.5, 0.6]},
    ]
    results = _write_and_query(
        test_items=updated_items,
        input_path=input_path,
        schema=InputSchema,
        collection_name=collection_name,
        primary_key_column="doc_id",
        vector_column="vector",
    )

    results.sort(key=lambda x: x["doc_id"])
    assert len(results) == 2
    assert results[0]["text"] == "hello updated"
    assert results[1]["text"] == "world"


def test_milvus_auto_create_collection(tmp_path):
    class InputSchema(pw.Schema):
        doc_id: int
        text: str
        vector: list[float]

    input_path = tmp_path / "input.txt"
    collection_name = _generate_collection_name()

    # Verify collection does not exist
    client = MilvusClient(uri=MILVUS_URI)
    assert not client.has_collection(collection_name)
    client.close()

    test_items = [
        {"doc_id": 1, "text": "hello", "vector": [0.1, 0.2, 0.3]},
    ]
    _write_and_query(
        test_items=test_items,
        input_path=input_path,
        schema=InputSchema,
        collection_name=collection_name,
        primary_key_column="doc_id",
        vector_column="vector",
    )

    # Verify collection was created
    client = MilvusClient(uri=MILVUS_URI)
    assert client.has_collection(collection_name)
    client.close()


def test_milvus_delete(tmp_path):
    class InputSchema(pw.Schema):
        doc_id: int
        text: str
        vector: list[float]

    input_path = tmp_path / "input.txt"
    pstorage_path = tmp_path / "PStorage"
    collection_name = _generate_collection_name()

    # First write: two items
    G.clear()
    test_items = [
        {"doc_id": 1, "text": "hello", "vector": [0.1, 0.2, 0.3]},
        {"doc_id": 2, "text": "world", "vector": [0.4, 0.5, 0.6]},
    ]
    with open(input_path, "w") as f:
        for item in test_items:
            f.write(json.dumps(item) + "\n")

    persistence_config = pw.persistence.Config(
        backend=pw.persistence.Backend.filesystem(pstorage_path)
    )
    table = pw.io.jsonlines.read(input_path, schema=InputSchema, mode="static")
    pw.io.milvus.write(
        table,
        uri=MILVUS_URI,
        collection_name=collection_name,
        primary_key_column="doc_id",
        vector_column="vector",
        dimension=DIMENSION,
    )
    pw.run(persistence_config=persistence_config)

    # Second write: only one item (doc_id=2 removed)
    G.clear()
    remaining_items = [
        {"doc_id": 1, "text": "hello", "vector": [0.1, 0.2, 0.3]},
    ]
    with open(input_path, "w") as f:
        for item in remaining_items:
            f.write(json.dumps(item) + "\n")

    table = pw.io.jsonlines.read(input_path, schema=InputSchema, mode="static")
    pw.io.milvus.write(
        table,
        uri=MILVUS_URI,
        collection_name=collection_name,
        primary_key_column="doc_id",
        vector_column="vector",
        dimension=DIMENSION,
    )
    pw.run(persistence_config=persistence_config)

    # Verify only doc_id=1 remains
    client = MilvusClient(uri=MILVUS_URI)
    results = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["doc_id", "text"],
        limit=1000,
    )
    client.close()

    assert len(results) == 1
    assert results[0]["doc_id"] == 1
