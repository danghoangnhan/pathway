# Copyright © 2026 Pathway
#
# Milvus Vector Embedding Example
#
# This example demonstrates how to use Pathway to read documents, generate
# vector embeddings, and write them to a Milvus collection. It follows the
# Milvus quickstart pattern: https://github.com/milvus-io/bootcamp
#
# The example uses a deterministic hash-based embedding for simplicity (no
# API keys needed). See the comments at the bottom for how to swap in a
# real embedding model.

import hashlib
import struct

import pathway as pw

# To use advanced features with Pathway Scale, get your free license key from
# https://pathway.com/features and paste it below.
# To use Pathway Community, comment out the line below.
pw.set_license_key("demo-license-key-with-telemetry")

MILVUS_URI = "http://milvus:19530"
COLLECTION_NAME = "ai_knowledge_base"
DIMENSION = 128


class DocumentSchema(pw.Schema):
    doc_id: int
    text: str
    subject: str


def _compute_hash_embedding(text: str) -> list[float]:
    """Generate a deterministic embedding from text using SHA-512 hashing.

    This is a lightweight stand-in for a real embedding model. It produces
    consistent vectors where similar inputs yield somewhat similar outputs
    (due to the mixing properties of the hash), but it does NOT capture
    true semantic similarity. Replace with a real embedder for production use.
    """
    digest = hashlib.sha512(text.encode("utf-8")).digest()
    # Repeat digest to get enough bytes, unpack as floats, and normalize
    padded = (digest * (DIMENSION * 4 // len(digest) + 1))[: DIMENSION * 4]
    raw = struct.unpack(f"{DIMENSION}f", padded)
    norm = sum(x * x for x in raw) ** 0.5
    if norm == 0:
        return list(raw)
    return [x / norm for x in raw]


@pw.udf
def hash_embed(text: str) -> list[float]:
    """UDF wrapper around _compute_hash_embedding for use in Pathway pipelines."""
    return _compute_hash_embedding(text)


def verify_results():
    """Query Milvus to verify data was written successfully."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found.")
        return

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["doc_id", "text", "subject"],
        limit=20,
    )
    print(f"\nVerification: found {len(results)} documents in Milvus collection '{COLLECTION_NAME}':")
    for row in sorted(results, key=lambda r: r["doc_id"]):
        print(f"  [{row['subject']}] doc_id={row['doc_id']}: {row['text'][:80]}...")

    # Demo: similarity search
    query_text = "What is retrieval-augmented generation?"
    query_vec = _compute_hash_embedding(query_text)

    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=3,
        output_fields=["doc_id", "text", "subject"],
    )
    print(f"\nSimilarity search for: '{query_text}'")
    for hits in search_results:
        for hit in hits:
            entity = hit["entity"]
            print(f"  score={hit['distance']:.4f} [{entity['subject']}] {entity['text'][:80]}...")

    # Demo: metadata-filtered similarity search (common RAG pattern)
    filtered_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=3,
        filter='subject == "machine_learning"',
        output_fields=["doc_id", "text", "subject"],
    )
    print(f"\nFiltered search (subject='machine_learning') for: '{query_text}'")
    for hits in filtered_results:
        for hit in hits:
            entity = hit["entity"]
            print(f"  score={hit['distance']:.4f} [{entity['subject']}] {entity['text'][:80]}...")

    client.close()


def main():
    print("Starting Pathway → Milvus vector embedding pipeline")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding dimension: {DIMENSION}")

    # Read documents (static mode for this demo).
    # For a live-updating RAG pipeline, switch to streaming mode so new
    # documents are automatically embedded and indexed as they arrive:
    #
    #   documents = pw.io.jsonlines.read(
    #       "/app/data/",
    #       schema=DocumentSchema,
    #       mode="streaming",  # watches directory for new/changed files
    #   )
    documents = pw.io.jsonlines.read(
        "/app/data/",
        schema=DocumentSchema,
        mode="static",
    )

    # Generate embeddings
    embedded = documents.select(
        doc_id=pw.this.doc_id,
        text=pw.this.text,
        subject=pw.this.subject,
        vector=hash_embed(pw.this.text),
    )

    # Write to Milvus
    pw.io.milvus.write(
        embedded,
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        primary_key_column="doc_id",
        vector_column="vector",
        dimension=DIMENSION,
    )

    # Run the pipeline
    pw.run()

    print("\nPipeline completed. Verifying results...")
    verify_results()


# ---------------------------------------------------------------------------
# For real embedding models, see the standalone examples:
#   - app_pymilvus_embed.py  — pymilvus DefaultEmbeddingFunction (768-dim, no API key)
#   - app_openai_embed.py    — OpenAI text-embedding-3-small (1536-dim, requires API key)
#
# Each has its own Dockerfile and docker-compose file. See README.md for details.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
