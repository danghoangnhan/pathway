#!/bin/bash
# Register the Debezium PostgreSQL connector.
# Retries until Debezium Connect is ready to accept the configuration.

while true; do
  http_code=$(curl -o /dev/null -w "%{http_code}" -H 'Content-Type: application/json' debezium:8083/connectors --data '{
    "name": "products-connector",
    "config": {
      "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
      "plugin.name": "pgoutput",
      "database.hostname": "postgres",
      "database.port": "5432",
      "database.user": "user",
      "database.password": "password",
      "database.dbname": "products_db",
      "database.server.name": "postgres",
      "table.include.list": "public.products",
      "database.history.kafka.bootstrap.servers": "kafka:9092",
      "database.history.kafka.topic": "schema-changes.products"
    }
  }')
  if [ "$http_code" -eq 201 ]; then
    echo "Debezium connector registered successfully"
    break
  else
    echo "Retrying Debezium connector registration in 1 second..."
    sleep 1
  fi
done
