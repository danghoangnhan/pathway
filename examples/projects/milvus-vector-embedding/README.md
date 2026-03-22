# Pathway + Milvus: Vector Embedding Pipeline

This example demonstrates how to use **Pathway** to read documents, generate vector embeddings, and write them into a **Milvus** vector database. It follows the [Milvus quickstart](https://github.com/milvus-io/bootcamp/blob/master/tutorials/quickstart/quickstart.ipynb) pattern with AI/ML knowledge-base documents.

## What It Does

1. Reads 12 AI/ML documents from a JSONL file (covering history, machine learning, and applications)
2. Generates vector embeddings for each document using a hash-based UDF
3. Writes the documents and their embeddings to a Milvus collection
4. Queries Milvus to verify the data and runs a similarity search demo

## Architecture

```
documents.jsonl → Pathway (read → embed → write) → Milvus
```

The pipeline uses Pathway's `pw.io.milvus.write()` connector, which:
- Upserts documents by primary key
- Automatically creates the Milvus collection if it doesn't exist
- Batches writes for efficiency
- Propagates deletions from Pathway to Milvus

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

## How to Run

```bash
docker compose up --build
```

This starts four services:
- **etcd** — metadata store for Milvus
- **minio** — object storage backend for Milvus
- **milvus** — vector database (standalone mode, port 19530)
- **pathway** — runs the embedding pipeline

The Pathway container will process the documents, write them to Milvus, and print verification output showing the stored documents and a similarity search result.

## How to Verify

After the pipeline completes, you can query Milvus directly:

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
results = client.query(
    collection_name="ai_knowledge_base",
    filter="",
    output_fields=["doc_id", "text", "subject"],
    limit=20,
)
for r in results:
    print(f"[{r['subject']}] {r['text'][:80]}...")
```

## Example Variants

Three embedding options are available, each as a standalone example with its own Dockerfile and docker-compose file.

### Default: Hash-based embedding (no API key)

Uses a deterministic hash-based embedding (128-dim) for a quick demo with zero dependencies.

```bash
docker compose up --build
```

### Option 1: pymilvus DefaultEmbeddingFunction (no API key)

Uses pymilvus's built-in open-source embedding model (768-dim). Downloads the model (~100 MB) on first run.

```bash
docker compose -f docker-compose.pymilvus.yml up --build
```

### Option 2: OpenAI embeddings (requires API key)

Uses Pathway's `OpenAIEmbedder` with OpenAI's `text-embedding-3-small` model (1536-dim) for production-quality semantic embeddings.

```bash
export OPENAI_API_KEY=sk-...
docker compose -f docker-compose.openai.yml up --build
```

## Adding Your Own Documents

Edit `data/documents.jsonl` to add your own documents. Each line should be a JSON object with:

```json
{"doc_id": 1, "text": "Your document text here.", "subject": "your_category"}
```

The `doc_id` field is the primary key, `text` is embedded, and `subject` is stored as metadata.

## Cleanup

```bash
docker compose down -v
```

The `-v` flag removes the data volumes (etcd, minio, milvus storage).
