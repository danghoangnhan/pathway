# Copyright © 2026 Pathway
#
# CDC Postgres-to-Milvus Sync
#
# Uses Debezium change data capture to stream rows from a PostgreSQL
# products table into Milvus. INSERT, UPDATE, and DELETE operations
# in Postgres automatically propagate to the Milvus vector index.
#
# This demonstrates a production-ready pattern: keep a relational
# database as the source of truth while maintaining a synchronized
# vector index for semantic search.

import time

import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

# To use advanced features with Pathway Scale, get your free license key from
# https://pathway.com/features and paste it below.
# To use Pathway Community, comment out the line below.
pw.set_license_key("demo-license-key-with-telemetry")

MILVUS_URI = "http://milvus:19530"
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384

KAFKA_SETTINGS = {
    "bootstrap.servers": "kafka:9092",
    "security.protocol": "plaintext",
    "group.id": "0",
    "session.timeout.ms": "6000",
    "auto.offset.reset": "earliest",
}


class ProductSchema(pw.Schema):
    id: int
    name: str
    description: str
    category: str
    price: float


def main():
    print("Starting CDC Postgres → Milvus sync pipeline")
    print(f"Milvus URI: {MILVUS_URI}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL} ({DIMENSION}-dim)")

    # Wait for Debezium to register connector and send initial snapshot
    print("Waiting for Debezium to initialize...")
    time.sleep(15)

    embedder = SentenceTransformerEmbedder(model=EMBEDDING_MODEL)

    # Read CDC stream from Debezium via Kafka
    # Debezium captures INSERT/UPDATE/DELETE from Postgres and publishes
    # them as events on the "postgres.public.products" Kafka topic.
    products = pw.io.debezium.read(
        KAFKA_SETTINGS,
        topic_name="postgres.public.products",
        schema=ProductSchema,
        autocommit_duration_ms=100,
    )

    # Embed product descriptions
    # Pathway represents each CDC operation as:
    #   INSERT  → row addition (is_addition=True)
    #   UPDATE  → delete old row + insert new row (in same minibatch)
    #   DELETE  → row deletion (is_addition=False)
    embedded = products.select(
        product_id=pw.this.id,
        name=pw.this.name,
        description=pw.this.description,
        category=pw.this.category,
        price=pw.this.price,
        vector=embedder(pw.this.description),
    )

    # Write to Milvus — all CDC operations propagate automatically:
    #   INSERT  → Milvus upsert
    #   UPDATE  → _MilvusOutputBuffer.on_time_end filters spurious delete → upsert
    #   DELETE  → Milvus delete
    pw.io.milvus.write(
        embedded,
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        primary_key_column="product_id",
        vector_column="vector",
        dimension=DIMENSION,
    )

    print("Pipeline running. Listening for CDC events...")
    pw.run()


if __name__ == "__main__":
    main()
