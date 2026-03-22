# Pathway: Real-Time Data Processing Framework

Pathway is a data processing framework designed for building real-time ETL pipelines and streaming applications. It provides a Python API that allows developers to write data transformations declaratively while the engine handles incremental computation, ensuring that outputs stay synchronized with changing inputs automatically.

## Core Concepts

Pathway operates on tables that represent potentially infinite streams of data. When input data changes, Pathway incrementally recomputes only the affected downstream results rather than reprocessing everything from scratch. This incremental approach makes it highly efficient for scenarios where data arrives continuously or existing records get updated.

The framework supports a wide range of input connectors including file systems, Kafka, PostgreSQL, MongoDB, and HTTP endpoints. Similarly, output connectors can write results to databases, message queues, vector databases like Milvus, and REST APIs. This makes Pathway suitable as the backbone of data integration architectures.

## Streaming ETL Pipelines

One of Pathway's primary use cases is building streaming ETL (Extract, Transform, Load) pipelines. In a traditional batch ETL system, data is periodically extracted from sources, transformed, and loaded into a destination. With Pathway, this process happens continuously: as soon as new data arrives at the source, it flows through the transformation pipeline and appears at the destination.

For example, a Pathway pipeline can watch a directory for new documents, parse and chunk them, generate vector embeddings, and write the results to a vector database like Milvus. When documents are added, modified, or deleted, the pipeline automatically propagates these changes to the vector database, keeping the index perfectly synchronized with the source files.

## Data Transformation Capabilities

Pathway provides rich transformation operators familiar to anyone who has used SQL or pandas. Tables support select, filter, join, groupby with aggregation, and user-defined functions (UDFs). Joins can combine streaming and static data sources, enabling patterns like enriching a real-time event stream with reference data from a CSV file or database table.

The framework also includes specialized modules for AI workloads. The LLM integration package provides parsers for various document formats, text splitters for chunking, and embedders that wrap popular models from OpenAI, Hugging Face, and other providers. These components can be composed into complete RAG indexing pipelines with just a few lines of code.

## Change Data Capture Integration

Pathway integrates with Debezium for change data capture (CDC) from relational databases. When rows are inserted, updated, or deleted in a source database like PostgreSQL, these changes flow through Kafka into Pathway, which can transform them and write the results to any output connector. This enables patterns like maintaining a synchronized vector search index alongside a relational database without writing any change-tracking code.
