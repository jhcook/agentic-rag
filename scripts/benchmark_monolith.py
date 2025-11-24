"""Benchmark script for the monolith RAG server."""
import statistics
import time

import requests

BASE_URL = "http://127.0.0.1:8001/api"
QUERIES = [
    "who is justin cook?",
    "what is his experience?",
    "tell me about his skills",
    "what companies has he worked for?",
    "does he know python?"
]
TIMEOUT = 30


def benchmark():
    """Run benchmark queries against the monolith server."""
    print(f"Benchmarking Monolith at {BASE_URL}...")
    latencies = []
    success_count = 0

    # Warmup
    print("Warming up...")
    try:
        requests.post(
            f"{BASE_URL}/search",
            json={"query": "warmup"},
            timeout=TIMEOUT)
    except Exception:
        pass

    print("Running queries...")
    for i, query in enumerate(QUERIES):
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/search",
                json={"query": query},
                timeout=TIMEOUT)
            end_time = time.time()

            if response.status_code == 200:
                latency = end_time - start_time
                latencies.append(latency)
                success_count += 1
                print(f"Query {i+1}: '{query}' - {latency:.2f}s (Success)")
            else:
                print(f"Query {i+1}: '{query}' - "
                      f"Failed with status {response.status_code}")
                print(response.text)
        except Exception as exc:
            print(f"Query {i+1}: '{query}' - Error: {exc}")

    if latencies:
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print("\nResults:")
        print(f"Total Queries: {len(QUERIES)}")
        print(f"Successful: {success_count}")
        print(f"Average Latency: {avg_latency:.2f}s")
        print(f"Min Latency: {min_latency:.2f}s")
        print(f"Max Latency: {max_latency:.2f}s")
    else:
        print("\nNo successful queries.")

if __name__ == "__main__":
    benchmark()
