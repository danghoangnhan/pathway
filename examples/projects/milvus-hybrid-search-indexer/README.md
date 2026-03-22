# Hybrid Dense+Sparse Search Indexer with Pathway + Milvus

A Pathway pipeline that computes both dense semantic embeddings and sparse keyword vectors for each document, writing both to a Milvus collection with dual vector fields. Demonstrates hybrid search that combines the strengths of semantic similarity and keyword matching.

## Architecture

```
documents.jsonl         Pathway Pipeline              Milvus
┌──────────────┐    ┌──────────────────────┐    ┌──────────────┐
│              │    │ jsonlines.read()     │    │ hybrid_docs  │
│ 25 documents │───>│  → SentenceTransform │───>│   collection │
│ (ambiguous   │    │     (384-dim dense)  │    │              │
│  keywords)   │    │  → TF hash UDF       │    │ dense_vector │
│              │    │     (sparse vector)  │    │ sparse_vector│
└──────────────┘    │  → write_hybrid()    │    └──────────────┘
                    └──────────────────────┘
```

## Why Hybrid Search?

The demo data deliberately includes documents where the same keyword appears in different semantic contexts:

| Keyword | Programming | Biology | Geography | Entertainment |
|---------|------------|---------|-----------|---------------|
| "python" | Python language | Python snake | — | Monty Python |
| "java" | Java language | — | Java island | Java coffee |

- **Dense-only search** for "python programming" finds semantically related documents about coding, but may also surface the snake or Monty Python docs
- **Sparse-only search** finds all documents containing the exact word "python" regardless of context
- **Hybrid search** combines both signals: documents that are both semantically relevant AND contain the right keywords rank highest

## Key Features

- **Dual embedding pipeline**: Dense (SentenceTransformer) + sparse (hash-based TF) computed in a single `select()`
- **Custom Milvus writer**: Extends `pw.io.subscribe()` to support multi-vector collections (since `pw.io.milvus.write()` supports a single vector field)
- **Four search modes**: Dense-only, sparse-only, WeightedRanker hybrid, RRFRanker hybrid
- **No API keys**: SentenceTransformerEmbedder runs locally

## Quick Start

```bash
docker compose up --build
```

First run downloads the embedding model (~80 MB). The pipeline processes 25 documents and demonstrates all four search modes.

## How the Sparse Vectors Work

The sparse embedding uses feature hashing (also called the "hashing trick"):

1. Tokenize text: lowercase, extract alphabetic tokens
2. Count term frequencies (TF)
3. Map each term to an index: `hash(term) % 30000`
4. Weight: `1 + log(count)` (log-normalized TF)

This produces a `{int: float}` dict that pymilvus stores as `SPARSE_FLOAT_VECTOR`. No global vocabulary is needed, making it suitable for streaming pipelines.

## Custom Writer vs Built-in Writer

The built-in `pw.io.milvus.write()` creates collections with a single vector field. This example includes `hybrid_milvus_writer.py` which:

- Creates a collection with explicit `FieldSchema` for both `FLOAT_VECTOR` and `SPARSE_FLOAT_VECTOR`
- Indexes: `AUTOINDEX` (IP) for dense, `SPARSE_INVERTED_INDEX` (IP) for sparse
- Follows the same `on_change`/`on_time_end` buffering pattern from the built-in connector
- Supports upsert and delete propagation

## Docker Services

| Service | Image | Purpose |
|---------|-------|---------|
| etcd | quay.io/coreos/etcd:v3.5.16 | Milvus metadata store |
| minio | minio/minio:RELEASE.2023-03-20 | Milvus object storage |
| milvus | milvusdb/milvus:v2.5.0 | Vector database |
| pathway | pathwaycom/pathway:latest | Dual-embedding ETL pipeline |

## Cleanup

```bash
docker compose down -v
```
