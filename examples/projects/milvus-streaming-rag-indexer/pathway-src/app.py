# Copyright © 2026 Pathway
#
# Streaming RAG Document Indexer
#
# Watches a directory for text/markdown files, parses them, chunks them,
# generates embeddings with SentenceTransformerEmbedder, and writes the
# chunks to Milvus. File additions, edits, and deletions are all
# automatically propagated to the Milvus collection.
#
# This example demonstrates Pathway's streaming ETL capabilities for
# building a live-updating RAG index. Compare with the Milvus bootcamp
# RAG tutorial which is a one-shot batch script.

import hashlib

import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.parsers import Utf8Parser
from pathway.xpacks.llm.splitters import TokenCountSplitter

# To use advanced features with Pathway Scale, get your free license key from
# https://pathway.com/features and paste it below.
# To use Pathway Community, comment out the line below.
pw.set_license_key("demo-license-key-with-telemetry")

MILVUS_URI = "http://milvus:19530"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384


# --- UDFs ---


@pw.udf
def extract_path(metadata: pw.Json) -> str:
    """Extract the file path from filesystem metadata."""
    return str(metadata["path"])


@pw.udf
def get_chunk_text(chunk_pair: tuple) -> str:
    """Extract the text string from a (text, metadata) tuple returned by splitter."""
    return chunk_pair[0]


@pw.udf
def make_chunk_id(path: str, chunk_text: str) -> str:
    """Generate a stable, deterministic chunk ID from file path and content.

    Using a content hash ensures that:
    - Same content always gets the same ID (idempotent)
    - Changed content gets a new ID (old entry deleted, new entry inserted)
    - Different files with same content get different IDs (path prefix)
    """
    content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
    safe_path = path.replace("/", "_").replace(".", "_")
    return f"{safe_path}_{content_hash}"


# --- Verification ---


def verify_results():
    """Query Milvus to verify chunks were written and run similarity search."""
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found.")
        return

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["chunk_id", "chunk_text", "source_path"],
        limit=100,
    )
    print(
        f"\nVerification: found {len(results)} chunks in '{COLLECTION_NAME}'"
    )
    for row in results[:5]:
        print(
            f"  [{row['source_path']}] {row['chunk_text'][:80]}..."
        )
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more chunks")

    # Similarity search
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_text = "How does retrieval-augmented generation work?"
    query_vec = model.encode(query_text).tolist()

    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=3,
        output_fields=["chunk_id", "chunk_text", "source_path"],
    )
    print(f"\nSimilarity search for: '{query_text}'")
    for hits in search_results:
        for hit in hits:
            entity = hit["entity"]
            print(
                f"  score={hit['distance']:.4f} [{entity['source_path']}] "
                f"{entity['chunk_text'][:80]}..."
            )

    # Filtered search by source path
    filtered_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=3,
        filter='source_path like "%rag%"',
        output_fields=["chunk_id", "chunk_text", "source_path"],
    )
    print(f"\nFiltered search (path contains 'rag') for: '{query_text}'")
    for hits in filtered_results:
        for hit in hits:
            entity = hit["entity"]
            print(
                f"  score={hit['distance']:.4f} [{entity['source_path']}] "
                f"{entity['chunk_text'][:80]}..."
            )

    client.close()


# --- Pipeline ---


def main():
    print("Starting Streaming RAG Document Indexer")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL} ({DIMENSION}-dim)")

    # Initialize components
    parser = Utf8Parser()
    splitter = TokenCountSplitter(min_tokens=50, max_tokens=500)
    embedder = SentenceTransformerEmbedder(model=EMBEDDING_MODEL)

    # 1. Watch directory for text/markdown files (streaming mode)
    #    In streaming mode, Pathway continuously monitors the directory and
    #    processes new, modified, and deleted files automatically.
    #    Switch to mode="static" for a one-time batch run.
    raw_docs = pw.io.fs.read(
        "/app/data/sample_docs/",
        format="binary",
        with_metadata=True,
        mode="streaming",
    )

    # 2. Extract file path from metadata for tracking
    docs_with_path = raw_docs.select(
        data=pw.this.data,
        source_path=extract_path(pw.this._metadata),
    )

    # 3. Parse: decode UTF-8 bytes to text
    #    Utf8Parser is async, returns list[tuple[str, dict]]
    parsed = docs_with_path.select(
        text=parser(pw.this.data),
        source_path=pw.this.source_path,
    ).await_futures()

    # 4. Flatten parsed output (Utf8Parser returns single-element list per file)
    parsed_flat = parsed.flatten(pw.this.text)

    # 5. Chunk with TokenCountSplitter
    #    Splitter accepts (str, dict) tuples, returns list[tuple[str, dict]]
    chunked = parsed_flat.select(
        chunks=splitter(pw.this.text),
        source_path=pw.this.source_path,
    )

    # 6. Flatten chunks — one row per chunk
    chunks_flat = chunked.flatten(pw.this.chunks)

    # 7. Extract text from (text, metadata) tuple and generate stable chunk ID
    chunks_with_id = chunks_flat.select(
        chunk_text=get_chunk_text(pw.this.chunks),
        source_path=pw.this.source_path,
    )
    chunks_with_id = chunks_with_id.select(
        chunk_id=make_chunk_id(pw.this.source_path, pw.this.chunk_text),
        chunk_text=pw.this.chunk_text,
        source_path=pw.this.source_path,
    )

    # 8. Embed with SentenceTransformerEmbedder (384-dim, local model)
    embedded = chunks_with_id.select(
        chunk_id=pw.this.chunk_id,
        chunk_text=pw.this.chunk_text,
        source_path=pw.this.source_path,
        vector=embedder(pw.this.chunk_text),
    )

    # 9. Write to Milvus
    #    - chunk_id is the primary key (str, deterministic per chunk)
    #    - chunk_text and source_path are stored as dynamic fields
    #    - Deletions propagate: if a source file is removed, its chunks
    #      disappear from Milvus automatically
    pw.io.milvus.write(
        embedded,
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        primary_key_column="chunk_id",
        vector_columns={
            "vector": {"type": pw.io.milvus.MilvusType.FLOAT_VECTOR, "dimension": DIMENSION},
        },
    )

    # Run the pipeline (blocks in streaming mode, monitoring for file changes)
    pw.run()

    print("\nPipeline completed. Verifying results...")
    verify_results()


if __name__ == "__main__":
    main()
