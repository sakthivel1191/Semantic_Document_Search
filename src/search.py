
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(path="chunks.jsonl"):
    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks

def search(query, top_k=5, filters=None, min_score=0.0,
           model=None, chunks=None, embeddings=None):

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    if chunks is None:
        chunks = load_chunks()

    if embeddings is None:
        embeddings = np.load("embeddings/all_embeddings.npy")

    if len(chunks) != len(embeddings):
        raise ValueError("Embeddings and chunks count mismatch")

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = np.dot(embeddings, query_vector)

    results = []

    for index, score in enumerate(scores):
        chunk = chunks[index]

        if filters:
            matched = all(
                chunk.get(key) == value
                for key, value in filters.items()
                if value is not None
            )

            if not matched:
                continue

        if float(score) < min_score:
            continue

        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "section": chunk.get("section", ""),
            "score": float(score),
            "text": chunk["text"]
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]

if __name__ == "__main__":
    while True:
        query = input("\nEnter your search (or exit): ")

        if query.lower() == "exit":
            break

        results = search(query)

        for result in results:
            print("\nSource:", result["source"])
            print("Page:", result["page"])
            print("Score:", round(result["score"], 4))
            print("Snippet:", result["text"][:300])
 