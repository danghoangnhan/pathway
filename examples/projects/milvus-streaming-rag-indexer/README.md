# Streaming RAG Document Indexer with Pathway + Milvus

A real-time document indexing pipeline that watches a directory for text and markdown files, automatically parses, chunks, embeds, and indexes them in Milvus. When files are added, modified, or deleted, the Milvus vector index updates automatically.

This showcases Pathway's streaming ETL capabilities compared to one-shot batch indexing scripts.

## Architecture

```
data/sample_docs/         Pathway Pipeline              Milvus
┌──────────────┐    ┌──────────────────────┐    ┌──────────────┐
│ .md / .txt   │───>│ fs.read (streaming)  │    │              │
│ files        │    │  → Utf8Parser        │    │ rag_documents│
│              │    │  → TokenCountSplitter │───>│ collection   │
│ (add/edit/   │    │  → SentenceTransform │    │              │
│  delete)     │    │  → milvus.write()    │    │ (auto-sync)  │
└──────────────┘    └──────────────────────┘    └──────────────┘
```

## Key Features

- **Streaming mode**: Continuously monitors the data directory for changes
- **Automatic chunking**: TokenCountSplitter breaks documents into 50-500 token chunks
- **Stable chunk IDs**: Content-hash-based IDs ensure correct update/delete propagation
- **Local embeddings**: SentenceTransformerEmbedder (all-MiniLM-L6-v2, 384-dim) — no API key needed
- **Change propagation**: File edits update affected chunks; file deletions remove chunks from Milvus

## Quick Start

```bash
docker compose up --build
```

The pipeline starts in streaming mode. On first run, it:
1. Downloads the embedding model (~80 MB) — this takes a minute on first run
2. Processes the 5 sample documents into ~20-40 chunks
3. Embeds and writes all chunks to Milvus
4. Continues watching for file changes

## Live Demo

While the pipeline is running, try adding a new document:

```bash
# Add a new file to the watched directory
echo "Quantum computing uses quantum mechanical phenomena like superposition
and entanglement to perform computations..." > data/sample_docs/quantum_computing.txt
```

The pipeline detects the new file, chunks it, embeds the chunks, and writes them to Milvus within seconds.

To verify, query Milvus directly:

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(uri="http://localhost:19530")
model = SentenceTransformer("all-MiniLM-L6-v2")

query_vec = model.encode("How does RAG work?").tolist()
results = client.search(
    collection_name="rag_documents",
    data=[query_vec],
    limit=3,
    output_fields=["chunk_text", "source_path"],
)
for hits in results:
    for hit in hits:
        print(f"  score={hit['distance']:.4f} [{hit['entity']['source_path']}]")
        print(f"    {hit['entity']['chunk_text'][:100]}...")
```

## Components

| Component | Details |
|-----------|---------|
| Parser | `Utf8Parser` — lightweight UTF-8 decoder |
| Splitter | `TokenCountSplitter(min_tokens=50, max_tokens=500)` |
| Embedder | `SentenceTransformerEmbedder("all-MiniLM-L6-v2")` — 384-dim |
| Connector | `pw.io.milvus.write()` with string primary key |

## Docker Services

| Service | Image | Purpose |
|---------|-------|---------|
| etcd | quay.io/coreos/etcd:v3.5.16 | Milvus metadata store |
| minio | minio/minio:RELEASE.2023-03-20 | Milvus object storage |
| milvus | milvusdb/milvus:v2.5.0 | Vector database |
| pathway | pathwaycom/pathway:latest | Streaming ETL pipeline |

## Cleanup

```bash
docker compose down -v
```
