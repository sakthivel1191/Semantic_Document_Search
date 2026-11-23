
import os
import json
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from src.chunk import create_chunks, generate_chunk_id
from src.extract import extract_text

DATA_FOLDER = "data"
EVALUATION_FILE = "evaluation_queries.json"

def load_queries():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def build_configuration(chunk_size, overlap, model):
    chunks = []

    for filename in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, filename)

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension not in [".pdf", ".docx", ".txt"]:
            continue

        pages = extract_text(path)

        for page in pages:
            page_chunks = create_chunks(
                page["text"],
                chunk_size,
                overlap
            )

            for number, text in enumerate(page_chunks, start=1):
                chunks.append({
                    "chunk_id": generate_chunk_id(
                        filename,
                        page["page"],
                        number,
                        text
                    ),
                    "source": filename,
                    "doc_type": extension,
                    "page": page["page"],
                    "section": page.get("section", ""),
                    "text": text
                })

    if not chunks:
        raise ValueError("No documents found")

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return chunks, np.asarray(vectors)

def calculate_metrics(queries, chunks, vectors, model):
    hit1 = 0
    hit3 = 0
    hit5 = 0
    reciprocal_ranks = []
    details = []

    for item in queries:
        query_vector = model.encode(
            item["query"],
            normalize_embeddings=True
        )

        scores = np.dot(vectors, query_vector)

        ranked_indexes = np.argsort(scores)[::-1]

        relevant_rank = None

        for rank, index in enumerate(ranked_indexes, start=1):
            chunk = chunks[index]

            if (
                chunk["source"] == item["expected_source"]
                and chunk["page"] == item["expected_page"]
            ):
                relevant_rank = rank
                break

        if relevant_rank is None:
            reciprocal_rank = 0
        else:
            reciprocal_rank = 1 / relevant_rank

        if relevant_rank is not None and relevant_rank <= 1:
            hit1 += 1

        if relevant_rank is not None and relevant_rank <= 3:
            hit3 += 1

        if relevant_rank is not None and relevant_rank <= 5:
            hit5 += 1

        reciprocal_ranks.append(reciprocal_rank)

        details.append({
            "query": item["query"],
            "expected_source": item["expected_source"],
            "expected_page": item["expected_page"],
            "rank": relevant_rank,
            "reciprocal_rank": reciprocal_rank
        })

    total = len(queries)

    metrics = {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "hit@5": hit5 / total,
        "MRR": sum(reciprocal_ranks) / total
    }

    return metrics, details

def run_evaluation():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    queries = load_queries()

    configurations = [
        {
            "name": "Config A",
            "chunk_size": 500,
            "overlap": 75
        },
        {
            "name": "Config B",
            "chunk_size": 400,
            "overlap": 60
        }
    ]

    results = []

    for config in configurations:
        print("\nRunning:", config["name"])

        chunks, vectors = build_configuration(
            config["chunk_size"],
            config["overlap"],
            model
        )

        metrics, details = calculate_metrics(
            queries,
            chunks,
            vectors,
            model
        )

        result = {
            "configuration": config["name"],
            "chunk_size": config["chunk_size"],
            "overlap": config["overlap"],
            "total_chunks": len(chunks),
            "metrics": metrics,
            "details": details
        }

        results.append(result)

        print("Total chunks:", len(chunks))
        print("Hit@1:", round(metrics["hit@1"], 3))
        print("Hit@3:", round(metrics["hit@3"], 3))
        print("Hit@5:", round(metrics["hit@5"], 3))
        print("MRR:", round(metrics["MRR"], 3))

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nExperiment Comparison")

    for result in results:
        print(
            result["configuration"],
            result["metrics"]
        )

    print("\nThree Worst Queries")

    worst_queries = sorted(
        results[0]["details"],
        key=lambda item: item["reciprocal_rank"]
    )[:3]

    for item in worst_queries:
        print("\nQuery:", item["query"])
        print("Expected:", item["expected_source"])
        print("Expected page:", item["expected_page"])
        print("Rank:", item["rank"])

if __name__ == "__main__":
    run_evaluation()
