# Copyright © 2026 Pathway
#
# Multi-Source Movie Recommendation Index
#
# Reads a static movie catalog (CSV) and streams user ratings from Kafka.
# Aggregates ratings per movie, joins with catalog, embeds descriptions,
# and writes enriched vectors to Milvus. As new ratings arrive, Milvus
# entries are automatically updated with fresh avg_rating and num_ratings.
#
# This demonstrates Pathway's multi-source ETL: joining static reference
# data with a streaming event source, computing rolling aggregates, and
# keeping a vector index continuously synchronized.

import time

import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

# To use advanced features with Pathway Scale, get your free license key from
# https://pathway.com/features and paste it below.
# To use Pathway Community, comment out the line below.
pw.set_license_key("demo-license-key-with-telemetry")

MILVUS_URI = "http://milvus:19530"
COLLECTION_NAME = "movie_recommendations"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384


# --- Schemas ---


class MovieSchema(pw.Schema):
    movie_id: int
    title: str
    description: str
    genre: str
    year: int


class RatingSchema(pw.Schema):
    movie_id: int
    user_id: int
    rating: float


# --- UDFs ---


@pw.udf
def safe_float(val: float | None) -> float:
    """Convert nullable float to 0.0 (handles None from left join)."""
    return float(val) if val is not None else 0.0


@pw.udf
def safe_int(val: int | None) -> int:
    """Convert nullable int to 0 (handles None from left join)."""
    return int(val) if val is not None else 0


# --- Verification ---


def verify_results():
    """Query Milvus to verify movies and demonstrate filtered search."""
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found.")
        return

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["movie_id", "title", "genre", "year", "avg_rating", "num_ratings"],
        limit=100,
    )
    print(f"\nFound {len(results)} movies in Milvus:")
    for row in sorted(results, key=lambda r: r.get("avg_rating", 0), reverse=True)[:10]:
        print(
            f"  [{row['genre']}] {row['title']} ({row['year']}) "
            f"- avg_rating={row.get('avg_rating', 0):.2f}, "
            f"num_ratings={row.get('num_ratings', 0)}"
        )

    # Semantic search
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_text = "a thrilling space adventure with aliens"
    query_vec = model.encode(query_text).tolist()

    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=5,
        output_fields=["movie_id", "title", "genre", "year", "avg_rating"],
    )
    print(f"\nSemantic search: '{query_text}'")
    for hits in search_results:
        for hit in hits:
            e = hit["entity"]
            print(
                f"  score={hit['distance']:.4f} {e['title']} ({e['year']}) "
                f"[{e['genre']}] avg={e.get('avg_rating', 0):.2f}"
            )

    # Filtered search: Sci-Fi only
    filtered = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=5,
        filter='genre == "Sci-Fi"',
        output_fields=["movie_id", "title", "genre", "year", "avg_rating"],
    )
    print(f"\nFiltered (genre='Sci-Fi'): '{query_text}'")
    for hits in filtered:
        for hit in hits:
            e = hit["entity"]
            print(
                f"  score={hit['distance']:.4f} {e['title']} ({e['year']}) "
                f"avg={e.get('avg_rating', 0):.2f}"
            )

    # Comedy search
    comedy_query = "a funny lighthearted movie"
    comedy_vec = model.encode(comedy_query).tolist()
    comedy_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[comedy_vec],
        limit=5,
        filter='genre == "Comedy"',
        output_fields=["movie_id", "title", "genre", "year", "avg_rating"],
    )
    print(f"\nFiltered (genre='Comedy'): '{comedy_query}'")
    for hits in comedy_results:
        for hit in hits:
            e = hit["entity"]
            print(
                f"  score={hit['distance']:.4f} {e['title']} ({e['year']}) "
                f"avg={e.get('avg_rating', 0):.2f}"
            )

    client.close()


# --- Pipeline ---


def main():
    print("Starting Movie Recommendation Index Pipeline")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL} ({DIMENSION}-dim)")

    embedder = SentenceTransformerEmbedder(model=EMBEDDING_MODEL)

    # 1. Read static movie catalog from CSV
    movies = pw.io.csv.read(
        "/app/data/",
        schema=MovieSchema,
        mode="static",
    )

    # 2. Read streaming ratings from Kafka
    rdkafka_settings = {
        "bootstrap.servers": "kafka:9092",
        "security.protocol": "plaintext",
        "group.id": "movie-recs",
        "session.timeout.ms": "6000",
        "auto.offset.reset": "earliest",
    }

    ratings = pw.io.kafka.read(
        rdkafka_settings,
        topic="ratings",
        format="json",
        schema=RatingSchema,
        autocommit_duration_ms=1000,
    )

    # 3. Aggregate ratings per movie (streaming aggregation)
    #    As new ratings arrive, these aggregates update automatically
    rating_aggs = ratings.groupby(pw.this.movie_id).reduce(
        movie_id=pw.this.movie_id,
        avg_rating=pw.reducers.avg(pw.this.rating),
        num_ratings=pw.reducers.count(),
    )

    # 4. Left join: movies with rating aggregates
    #    Movies with no ratings yet still appear (with 0.0/0 defaults)
    enriched = movies.join_left(
        rating_aggs,
        pw.left.movie_id == pw.right.movie_id,
    ).select(
        movie_id=pw.left.movie_id,
        title=pw.left.title,
        description=pw.left.description,
        genre=pw.left.genre,
        year=pw.left.year,
        avg_rating=safe_float(pw.right.avg_rating),
        num_ratings=safe_int(pw.right.num_ratings),
    )

    # 5. Embed movie descriptions
    #    When ratings change, the row is re-emitted and re-embedded.
    #    The description hasn't changed, so the same vector is produced.
    embedded = enriched.select(
        movie_id=pw.this.movie_id,
        title=pw.this.title,
        description=pw.this.description,
        genre=pw.this.genre,
        year=pw.this.year,
        avg_rating=pw.this.avg_rating,
        num_ratings=pw.this.num_ratings,
        vector=embedder(pw.this.description),
    )

    # 6. Write to Milvus
    #    - movie_id is the primary key (one entry per movie)
    #    - title, genre, year, avg_rating, num_ratings are dynamic fields
    #    - When ratings update, the same movie_id is upserted with new metadata
    pw.io.milvus.write(
        embedded,
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        primary_key_column="movie_id",
        vector_columns={
            "vector": {"type": pw.io.milvus.MilvusType.FLOAT_VECTOR, "dimension": DIMENSION},
        },
    )

    # Wait for Kafka to be ready
    time.sleep(20)

    pw.run()

    print("\nPipeline completed. Verifying results...")
    verify_results()


if __name__ == "__main__":
    main()
