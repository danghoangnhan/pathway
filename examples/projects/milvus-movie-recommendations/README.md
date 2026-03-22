# Multi-Source Movie Recommendation Index with Pathway + Milvus

A streaming ETL pipeline that combines a static movie catalog (CSV) with live user ratings from Kafka. Pathway aggregates ratings per movie, joins them with the catalog, embeds descriptions, and writes enriched vectors to Milvus. As new ratings arrive, movies are automatically re-indexed with updated metadata.

## Architecture

```
movies.csv (static)       Pathway Pipeline              Milvus
┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│ 50 movies    │──┐ │ csv.read (static)    │    │                  │
│ title, desc, │  │ │                      │    │ movie_           │
│ genre, year  │  ├>│ join_left ──────────>│───>│ recommendations  │
│              │  │ │  + embed(description)│    │                  │
└──────────────┘  │ │                      │    │ vector + title,  │
                  │ │ groupby.reduce       │    │ genre, year,     │
Kafka (streaming) │ │  avg_rating          │    │ avg_rating,      │
┌──────────────┐  │ │  num_ratings         │    │ num_ratings      │
│ ratings topic│──┘ │                      │    │                  │
│ {movie_id,   │    │ kafka.read (stream)  │    │ (auto-updated)   │
│  rating}     │    └──────────────────────┘    └──────────────────┘
└──────────────┘
```

## Key Features

- **Multi-source ETL**: Joins static CSV data with a streaming Kafka topic
- **Streaming aggregation**: `groupby().reduce()` computes rolling avg_rating and num_ratings
- **Left join**: Movies with no ratings yet still appear in Milvus (defaults to 0)
- **Live updates**: As ratings arrive, Milvus entries are upserted with fresh metadata
- **Filtered vector search**: Query by semantic similarity + genre, year, or rating filters

## How It Works

1. **Static catalog**: 50 movies loaded from CSV at startup
2. **Streaming ratings**: Kafka producer sends ~200 ratings at 2/sec
3. **Aggregation**: Pathway computes per-movie avg_rating and count in real-time
4. **Join**: Static movies joined with streaming aggregates via `join_left`
5. **Embed + write**: Each movie's description is embedded; all fields written to Milvus
6. **Update flow**: New rating → aggregate updates → join re-emits → Milvus upserts

## Quick Start

```bash
docker compose up --build
```

The producer starts sending ratings after ~30 seconds. Watch the Pathway container logs to see embeddings being computed and written.

## Querying Milvus

After the pipeline has processed some ratings:

```python
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(uri="http://localhost:19530")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Semantic search: find space adventure movies
query_vec = model.encode("a thrilling space adventure").tolist()
results = client.search(
    collection_name="movie_recommendations",
    data=[query_vec], limit=5,
    output_fields=["title", "genre", "year", "avg_rating"],
)

# Filtered: only Sci-Fi with high ratings
filtered = client.search(
    collection_name="movie_recommendations",
    data=[query_vec], limit=5,
    filter='genre == "Sci-Fi" and avg_rating > 3.0',
    output_fields=["title", "year", "avg_rating"],
)
```

## Docker Services

| Service | Image | Purpose |
|---------|-------|---------|
| etcd | quay.io/coreos/etcd:v3.5.16 | Milvus metadata store |
| minio | minio/minio:RELEASE.2023-03-20 | Milvus object storage |
| milvus | milvusdb/milvus:v2.5.0 | Vector database |
| zookeeper | confluentinc/cp-zookeeper:5.5.3 | Kafka coordination |
| kafka | confluentinc/cp-enterprise-kafka:5.5.3 | Message broker |
| pathway | pathwaycom/pathway:latest | Streaming ETL pipeline |
| stream-producer | python:3.10-slim | Simulated rating stream |

## Cleanup

```bash
docker compose down -v
```
