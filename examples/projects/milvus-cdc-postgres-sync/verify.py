#!/usr/bin/env python3
"""Verify CDC sync results in Milvus.

Run this script after the pipeline has processed the initial snapshot
and any simulated changes.

Usage:
    pip install pymilvus sentence-transformers
    python verify.py
"""

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

client = MilvusClient(uri=MILVUS_URI)
model = SentenceTransformer(EMBEDDING_MODEL)

if not client.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' not found. Is the pipeline running?")
    exit(1)

# 1. List all products
results = client.query(
    collection_name=COLLECTION_NAME,
    filter="",
    output_fields=["product_id", "name", "category", "price"],
    limit=100,
)
print(f"Total products in Milvus: {len(results)}")
for r in sorted(results, key=lambda x: x["product_id"]):
    print(f"  id={r['product_id']:3d} [{r['category']:12s}] {r['name']:30s} ${r['price']}")

# 2. Semantic search
query = "comfortable footwear for running and exercise"
query_vec = model.encode(query).tolist()
print(f"\nSemantic search: '{query}'")
search_results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vec],
    limit=5,
    output_fields=["product_id", "name", "category", "price"],
)
for hits in search_results:
    for hit in hits:
        e = hit["entity"]
        print(
            f"  score={hit['distance']:.4f} id={e['product_id']} "
            f"[{e['category']}] {e['name']} ${e['price']}"
        )

# 3. Filtered search: electronics under $100
print(f"\nFiltered search (electronics under $100): '{query}'")
filtered = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vec],
    limit=5,
    filter='category == "electronics" and price < 100.0',
    output_fields=["product_id", "name", "category", "price"],
)
for hits in filtered:
    for hit in hits:
        e = hit["entity"]
        print(
            f"  score={hit['distance']:.4f} id={e['product_id']} "
            f"[{e['category']}] {e['name']} ${e['price']}"
        )

# 4. Search for books about data and AI
book_query = "data processing and machine learning"
book_vec = model.encode(book_query).tolist()
print(f"\nFiltered search (books): '{book_query}'")
book_results = client.search(
    collection_name=COLLECTION_NAME,
    data=[book_vec],
    limit=5,
    filter='category == "books"',
    output_fields=["product_id", "name", "category", "price"],
)
for hits in book_results:
    for hit in hits:
        e = hit["entity"]
        print(
            f"  score={hit['distance']:.4f} id={e['product_id']} "
            f"[{e['category']}] {e['name']} ${e['price']}"
        )

# 5. Verify DELETE propagation (if simulate_changes.sh was run)
deleted_check = client.query(
    collection_name=COLLECTION_NAME,
    filter="product_id == 4",
    output_fields=["product_id", "name"],
    limit=1,
)
print(f"\nDELETE check: product_id=4 exists = {len(deleted_check) > 0}")
if deleted_check:
    print(f"  (product still present: {deleted_check[0]['name']})")
else:
    print("  (product successfully removed from Milvus)")

client.close()
