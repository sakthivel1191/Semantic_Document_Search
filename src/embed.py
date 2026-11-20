import os
import json
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks
def generate_embeddings(chunks, batch_size=64):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    os.makedirs("embeddings", exist_ok=True)
    embeddings = []
    cache_hits = 0
    failed_batches = []
    start_time = time.time()
    for batch_start in tqdm(
        range(0, len(chunks), batch_size)
    ):
        batch = chunks[batch_start:batch_start + batch_size]
        batch_embeddings = []
        missing = []
        for chunk in batch:
            cache_path = os.path.join(
                "embeddings",
                chunk["chunk_id"] + ".npy"
            )
            if os.path.exists(cache_path):
                vector = np.load(cache_path)
                batch_embeddings.append(vector)
                cache_hits += 1
            else:
                batch_embeddings.append(None)
                missing.append(
                    (len(batch_embeddings) - 1, chunk, cache_path)
                )
        if missing:
            texts = [item[1]["text"] for item in missing]
            for attempt in range(3):
                try:
                    vectors = model.encode(
                        texts,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    for index, vector in zip(missing, vectors):
                        position, chunk, cache_path = index
                        np.save(cache_path, vector)
                        batch_embeddings[position] = vector
                    break
                except Exception as error:
                    if attempt == 2:
                        failed_batches.append({
                            "batch": batch_start,
                            "error": str(error)
                        })
        embeddings.extend(batch_embeddings)
    successful = [
        vector for vector in embeddings
        if vector is not None
    ]
    if successful:
        np.save(
            "embeddings/all_embeddings.npy",
            np.array(successful)
        )
    with open("embeddings/failed_batches.json", "w") as file:
        json.dump(failed_batches, file, indent=4)
    elapsed = time.time() - start_time
    print(f"Total chunks: {len(chunks)}")
    print(f"Successful embeddings: {len(successful)}")
    print(f"Cache hits: {cache_hits}")
    print(f"Failed batches: {len(failed_batches)}")
    print(f"Total time: {elapsed:.2f} seconds")
    print("Local model API cost: $0")
    return successful
if __name__ == "__main__":
    chunks = load_chunks("chunks.jsonl")
    generate_embeddings(chunks)
