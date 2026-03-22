# CDC Postgres-to-Milvus Sync with Pathway

A real-time change data capture pipeline that synchronizes a PostgreSQL products table with a Milvus vector index. When rows are inserted, updated, or deleted in Postgres, the changes automatically propagate through Debezium → Kafka → Pathway → Milvus.

This demonstrates a production-ready pattern: keep a relational database as the source of truth while maintaining a synchronized vector search index.

## Architecture

```
PostgreSQL        Debezium        Kafka         Pathway          Milvus
┌──────────┐    ┌──────────┐    ┌──────┐    ┌──────────────┐  ┌──────────┐
│ products │    │          │    │      │    │              │  │          │
│ table    │───>│ CDC      │───>│ topic│───>│ debezium.read│  │ products │
│          │    │ connector│    │      │    │ → embed()    │─>│ collection│
│ INSERT   │    │          │    │      │    │ → milvus     │  │          │
│ UPDATE   │    │ WAL      │    │      │    │   .write()   │  │ (synced) │
│ DELETE   │    │ monitor  │    │      │    │              │  │          │
└──────────┘    └──────────┘    └──────┘    └──────────────┘  └──────────┘
```

## CDC Operation Mapping

| Postgres Operation | Pathway Representation | Milvus Result |
|---|---|---|
| INSERT | Row addition | Upsert new vector |
| UPDATE | Delete old + insert new (same minibatch) | Upsert updated vector |
| DELETE | Row deletion | Delete vector |

The Milvus connector's `on_time_end` logic intelligently distinguishes UPDATE (delete+insert of same PK) from true DELETE, ensuring correct behavior.

## Quick Start

### Option A: Automated demo

```bash
chmod +x run_demo.sh debezium/connector.sh sql/simulate_changes.sh
./run_demo.sh
```

### Option B: Manual setup

```bash
# Start all services
docker compose up -d --build

# Wait for Milvus to be healthy (~90 seconds)
# Register Debezium connector
docker compose exec debezium bash /kafka/connector.sh

# Wait for initial sync (~30 seconds), then verify
pip install pymilvus sentence-transformers
python verify.py

# Simulate database changes
docker compose exec postgres bash /simulate_changes.sh

# Wait (~20 seconds), then verify again
python verify.py
```

## What the Demo Does

1. **Initial sync**: Debezium snapshots 25 products from Postgres → all embedded and indexed in Milvus
2. **INSERT**: 2 new products added → appear in Milvus within seconds
3. **UPDATE**: 2 products modified (description/price) → re-embedded and upserted in Milvus
4. **DELETE**: 1 product removed → deleted from Milvus

## Sample Queries After Sync

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(uri="http://localhost:19530")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Semantic search across all products
query_vec = model.encode("comfortable running shoes").tolist()
results = client.search(
    collection_name="products", data=[query_vec], limit=5,
    output_fields=["name", "category", "price"],
)

# Filtered: books about data and AI
query_vec = model.encode("data processing and machine learning").tolist()
results = client.search(
    collection_name="products", data=[query_vec], limit=5,
    filter='category == "books"',
    output_fields=["name", "price"],
)
```

## Docker Services (8 containers)

| Service | Image | Purpose |
|---------|-------|---------|
| postgres | debezium/postgres:13 | Source database with WAL enabled |
| zookeeper | confluentinc/cp-zookeeper:5.5.3 | Kafka coordination |
| kafka | confluentinc/cp-enterprise-kafka:5.5.3 | CDC event transport |
| debezium | debezium/connect:1.4 | Change data capture connector |
| etcd | quay.io/coreos/etcd:v3.5.16 | Milvus metadata store |
| minio | minio/minio:RELEASE.2023-03-20 | Milvus object storage |
| milvus | milvusdb/milvus:v2.5.0 | Vector database |
| pathway | pathwaycom/pathway:latest | Streaming ETL pipeline |

## Cleanup

```bash
docker compose down -v
```
