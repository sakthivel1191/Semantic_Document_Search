
import time
import numpy as np

from src.api import MODEL, CHUNKS, VECTORS, perform_search

def measure_latency():
    queries = [
        "What is the employee leave policy?",
        "What are the safety rules?",
        "What are the working hours?",
        "What is the attendance policy?",
        "What are employee benefits?",
        "What is the recruitment process?",
        "What are the security guidelines?",
        "What is the training policy?",
        "What is the financial report?",
        "What are the health guidelines?",
        "What is the data protection policy?",
        "What are the customer service rules?",
        "What is the remote working policy?",
        "What are the quality standards?",
        "What is the environmental policy?",
        "What are the complaint procedures?",
        "What are the document management rules?",
        "What are the expense policies?",
        "What are the performance guidelines?",
        "What are the reporting requirements?"
    ]

    latencies = []

    for query in queries:
        start = time.perf_counter()

        perform_search(
            query,
            5
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

        print(query, round(elapsed, 2), "ms")

    print("\nLatency Results")

    print(
        "p50:",
        round(float(np.percentile(latencies, 50)), 2),
        "ms"
    )

    print(
        "p95:",
        round(float(np.percentile(latencies, 95)), 2),
        "ms"
    )

if __name__ == "__main__":
    import src.api as api

    api.MODEL = api.SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    api.load_index()

    MODEL = api.MODEL
    CHUNKS = api.CHUNKS
    VECTORS = api.VECTORS

    measure_latency()
