#!/bin/bash
set -e

echo "=== Starting CDC Postgres → Milvus sync demo ==="
echo ""

# 1. Start the stack
echo "Step 1: Starting all services..."
docker compose up -d --build

# 2. Wait for Milvus
echo ""
echo "Step 2: Waiting for Milvus to be healthy..."
until docker compose exec milvus curl -sf http://localhost:9091/healthz > /dev/null 2>&1; do
    sleep 5
    echo "  waiting for Milvus..."
done
echo "  Milvus is ready."

# 3. Register Debezium connector
echo ""
echo "Step 3: Registering Debezium connector..."
docker compose exec debezium bash /kafka/connector.sh

# 4. Wait for initial sync
echo ""
echo "Step 4: Waiting for initial CDC sync (30s)..."
sleep 30

# 5. Verify initial state
echo ""
echo "Step 5: Verifying initial sync (25 products)..."
python verify.py 2>/dev/null || echo "  (run 'pip install pymilvus sentence-transformers' to use verify.py)"

# 6. Simulate changes
echo ""
echo "Step 6: Simulating database changes..."
docker compose exec postgres bash /simulate_changes.sh

# 7. Wait for changes to propagate
echo ""
echo "Step 7: Waiting for changes to propagate (20s)..."
sleep 20

# 8. Verify final state
echo ""
echo "Step 8: Verifying final state (26 products, id=4 deleted)..."
python verify.py 2>/dev/null || echo "  (run 'pip install pymilvus sentence-transformers' to use verify.py)"

echo ""
echo "=== Demo complete ==="
echo "To clean up: docker compose down -v"
