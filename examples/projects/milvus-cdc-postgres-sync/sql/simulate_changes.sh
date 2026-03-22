#!/bin/bash
# Simulate INSERT, UPDATE, and DELETE operations on the products table.
# These changes are captured by Debezium and propagated through
# Kafka → Pathway → Milvus.

export PGPASSWORD='password'
DB_ARGS="-h localhost -d products_db -U user"

echo "=== Phase 1: INSERT new products ==="

psql $DB_ARGS -c "INSERT INTO products (name, description, category, price) VALUES
('Noise-Canceling Earbuds Pro', 'Next-generation earbuds with adaptive noise cancellation that uses external microphones to analyze ambient sound and create an anti-noise signal. Features premium drivers and lossless audio codec support.', 'electronics', 179.99);"
echo "  Inserted: Noise-Canceling Earbuds Pro"
sleep 2

psql $DB_ARGS -c "INSERT INTO products (name, description, category, price) VALUES
('Trail Running Pack', 'Lightweight hydration vest designed for ultra-distance trail running with two 500ml soft flasks and multiple storage pockets. Breathable mesh back panel prevents overheating.', 'sports', 89.99);"
echo "  Inserted: Trail Running Pack"
sleep 2

echo ""
echo "=== Phase 2: UPDATE existing products ==="

psql $DB_ARGS -c "UPDATE products SET description='Completely redesigned noise-canceling over-ear headphones with 40-hour battery life, Bluetooth 5.3, and multipoint connection supporting three simultaneous devices. Premium titanium drivers deliver studio-quality sound.', price=99.99 WHERE id=1;"
echo "  Updated: Wireless Headphones (id=1) - new description and price"
sleep 2

psql $DB_ARGS -c "UPDATE products SET price=29.99 WHERE id=22;"
echo "  Updated: Yoga Mat Premium (id=22) - price reduction"
sleep 2

echo ""
echo "=== Phase 3: DELETE a product ==="

psql $DB_ARGS -c "DELETE FROM products WHERE id=4;"
echo "  Deleted: Bluetooth Speaker (id=4)"
sleep 2

echo ""
echo "=== All changes applied ==="
echo "  2 INSERTs, 2 UPDATEs, 1 DELETE"
echo "  Expected: 26 products in Milvus (25 original + 2 new - 1 deleted)"
