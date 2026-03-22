# Copyright © 2026 Pathway
#
# Milvus Vector Embedding Example — OpenAI Embeddings
#
# This example uses Pathway's OpenAIEmbedder to generate high-quality semantic
# embeddings (1536-dim) via OpenAI's text-embedding-3-small model.
#
# Requires: OPENAI_API_KEY environment variable

import os

import pathway as pw

# To use advanced features with Pathway Scale, get your free license key from
# https://pathway.com/features and paste it below.
# To use Pathway Community, comment out the line below.
pw.set_license_key("demo-license-key-with-telemetry")

MILVUS_URI = "http://milvus:19530"
COLLECTION_NAME = "ai_knowledge_base_openai"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536


class DocumentSchema(pw.Schema):
    doc_id: int
    text: str
    subject: str


# Pathway's OpenAIEmbedder is a UDF — use it directly in select() calls.
# It handles batching, retries, and async execution automatically.
from pathway.xpacks.llm.embedders import OpenAIEmbedder

embedder = OpenAIEmbedder(model=EMBEDDING_MODEL)


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

    # Demo: similarity search using OpenAI embeddings
    from openai import OpenAI

    query_text = "What is retrieval-augmented generation?"
    openai_client = OpenAI()
    response = openai_client.embeddings.create(
        input=[query_text], model=EMBEDDING_MODEL
    )
    query_vec = response.data[0].embedding

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
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it before running: export OPENAI_API_KEY=sk-..."
        )

    print("Starting Pathway → Milvus vector embedding pipeline (OpenAI embedder)")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL} (dimension: {DIMENSION})")

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

    # Generate embeddings using OpenAI text-embedding-3-small.
    # The OpenAIEmbedder is a Pathway UDF — use it directly in select().
    embedded = documents.select(
        doc_id=pw.this.doc_id,
        text=pw.this.text,
        subject=pw.this.subject,
        vector=embedder(pw.this.text),
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


if __name__ == "__main__":
    main()
