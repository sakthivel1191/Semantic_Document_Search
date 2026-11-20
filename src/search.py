import json
import numpy as np
from sentence_transformers import SentenceTransformer
def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks
def search(query, top_k=3, filters=None):
    chunks = load_chunks("chunks.jsonl")
    embeddings = np.load(
        "embeddings/all_embeddings.npy"
    )
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of embeddings does not match chunks"
        )
    model = SentenceTransformer("all-MiniLM-L6-v2")
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
            )
            if not matched:
                continue
        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score),
            "text": chunk["text"]
        })
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )
    for result in results[:top_k]:
        print("\nSource:", result["source"])
        print("Page:", result["page"])
        print("Score:", round(result["score"], 4))
        print("Snippet:", result["text"][:300])
    return results[:top_k]
if __name__ == "__main__":
    while True:
        query = input("\nEnter your search (or exit): ")
        if query.lower() == "exit":
            break
        search(query, top_k=3)
