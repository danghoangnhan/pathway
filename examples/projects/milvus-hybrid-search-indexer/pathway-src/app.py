# Copyright © 2026 Pathway
#
# Hybrid Dense+Sparse Search Indexer
#
# Computes both dense embeddings (SentenceTransformer) and sparse TF vectors
# (hash-based term frequency) in a single Pathway pipeline. Writes both to a
# multi-vector Milvus collection via a custom hybrid writer.
#
# This demonstrates:
#   - Parallel dual-embedding computation in one declarative pipeline
#   - Custom Milvus writer extending the pw.io.subscribe() pattern
#   - Hybrid search combining semantic and keyword retrieval

import math
import re
from collections import Counter

import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

from hybrid_milvus_writer import write_hybrid

# To use advanced features with Pathway Scale, get your free license key from
# https://pathway.com/features and paste it below.
# To use Pathway Community, comment out the line below.
pw.set_license_key("demo-license-key-with-telemetry")

MILVUS_URI = "http://milvus:19530"
COLLECTION_NAME = "hybrid_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DENSE_DIMENSION = 384
VOCAB_SIZE = 30000  # Hash space for sparse vector term indices


class DocumentSchema(pw.Schema):
    doc_id: int
    text: str
    subject: str


# --- Sparse Embedding UDF ---


def _tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic tokens (2+ chars)."""
    return re.findall(r"\b[a-z][a-z0-9]{1,}\b", text.lower())


def _compute_sparse_vector(text: str) -> dict:
    """Compute a sparse TF vector with hash-based term indexing.

    Each term is mapped to a stable index via hash(term) % VOCAB_SIZE.
    Weights use log-normalized term frequency: 1 + log(count).
    """
    tokens = _tokenize(text)
    if not tokens:
        return {0: 1.0}  # pymilvus requires non-empty sparse vectors
    tf = Counter(tokens)
    sparse: dict[int, float] = {}
    for term, count in tf.items():
        idx = hash(term) % VOCAB_SIZE
        weight = 1.0 + math.log(count) if count > 0 else 0.0
        # On hash collision, keep the higher weight
        if idx not in sparse or sparse[idx] < weight:
            sparse[idx] = weight
    return sparse


@pw.udf
def compute_sparse(text: str) -> dict:
    """UDF: compute sparse TF vector for hybrid search."""
    return _compute_sparse_vector(text)


# --- Verification ---


def verify_results():
    """Query Milvus to verify data and demonstrate hybrid search."""
    from pymilvus import (
        AnnSearchRequest,
        MilvusClient,
        RRFRanker,
        WeightedRanker,
    )
    from sentence_transformers import SentenceTransformer

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found.")
        return

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["doc_id", "text", "subject"],
        limit=100,
    )
    print(f"\nVerification: found {len(results)} documents in '{COLLECTION_NAME}'")

    # Prepare query vectors
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_text = "python programming language"
    dense_vec = model.encode(query_text).tolist()
    sparse_vec = _compute_sparse_vector(query_text)

    # 1. Dense-only search (semantic similarity)
    print(f"\n=== Dense-only search for: '{query_text}' ===")
    dense_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_vec],
        anns_field="dense_vector",
        limit=5,
        output_fields=["doc_id", "text", "subject"],
    )
    for hits in dense_results:
        for hit in hits:
            e = hit["entity"]
            print(
                f"  score={hit['distance']:.4f} [{e['subject']}] {e['text'][:80]}..."
            )

    # 2. Sparse-only search (keyword matching)
    print(f"\n=== Sparse-only search for: '{query_text}' ===")
    sparse_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[sparse_vec],
        anns_field="sparse_vector",
        limit=5,
        output_fields=["doc_id", "text", "subject"],
    )
    for hits in sparse_results:
        for hit in hits:
            e = hit["entity"]
            print(
                f"  score={hit['distance']:.4f} [{e['subject']}] {e['text'][:80]}..."
            )

    # 3. Hybrid search with WeightedRanker
    print(f"\n=== Hybrid search (WeightedRanker 0.7/0.3) for: '{query_text}' ===")
    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "IP"},
        limit=5,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=5,
    )
    hybrid_results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(0.7, 0.3),
        limit=5,
        output_fields=["doc_id", "text", "subject"],
    )
    for hit in hybrid_results:
        for h in hit:
            print(
                f"  score={h.distance:.4f} [{h.entity['subject']}] "
                f"{h.entity['text'][:80]}..."
            )

    # 4. Hybrid search with RRFRanker
    print(f"\n=== Hybrid search (RRFRanker k=60) for: '{query_text}' ===")
    rrf_results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=5,
        output_fields=["doc_id", "text", "subject"],
    )
    for hit in rrf_results:
        for h in hit:
            print(
                f"  score={h.distance:.4f} [{h.entity['subject']}] "
                f"{h.entity['text'][:80]}..."
            )

    client.close()


# --- Pipeline ---


def main():
    print("Starting Hybrid Dense+Sparse Search Indexer")
    print(f"Dense: {EMBEDDING_MODEL} ({DENSE_DIMENSION}-dim)")
    print(f"Sparse: hash-based TF ({VOCAB_SIZE} buckets)")

    embedder = SentenceTransformerEmbedder(model=EMBEDDING_MODEL)

    # Read documents
    documents = pw.io.jsonlines.read(
        "/app/data/",
        schema=DocumentSchema,
        mode="static",
    )

    # Compute both dense and sparse embeddings in a single select
    embedded = documents.select(
        doc_id=pw.this.doc_id,
        text=pw.this.text,
        subject=pw.this.subject,
        dense_vector=embedder(pw.this.text),
        sparse_vector=compute_sparse(pw.this.text),
    )

    # Write to Milvus with custom hybrid writer
    write_hybrid(
        embedded,
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        primary_key_column="doc_id",
        dense_vector_column="dense_vector",
        sparse_vector_column="sparse_vector",
        dense_dimension=DENSE_DIMENSION,
    )

    pw.run()

    print("\nPipeline completed. Verifying results...")
    verify_results()


if __name__ == "__main__":
    main()
