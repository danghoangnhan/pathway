# Copyright © 2026 Pathway
#
# Simulated Kafka stream of movie ratings.
# Sends random ratings for movies 1-50 at ~2 ratings/sec.

import json
import random
import time

from kafka import KafkaProducer

random.seed(42)

NUM_MOVIES = 50
NUM_RATINGS = 200
TOPIC = "ratings"


def generate_stream():
    # Wait for Kafka to be ready
    time.sleep(30)

    producer = KafkaProducer(
        bootstrap_servers=["kafka:9092"],
        security_protocol="PLAINTEXT",
        api_version=(0, 10, 2),
    )

    print(f"Sending {NUM_RATINGS} ratings to topic '{TOPIC}'...")

    for i in range(NUM_RATINGS):
        rating = {
            "movie_id": random.randint(1, NUM_MOVIES),
            "user_id": random.randint(1, 500),
            "rating": round(random.uniform(1.0, 5.0), 1),
        }
        producer.send(TOPIC, json.dumps(rating).encode("utf-8"))

        if (i + 1) % 20 == 0:
            print(f"  Sent {i + 1}/{NUM_RATINGS} ratings")

        time.sleep(0.5)

    print("All ratings sent.")
    time.sleep(5)
    producer.close()


if __name__ == "__main__":
    generate_stream()
