import os
import json
import hashlib
from src.extract import extract_text
def create_chunks(text, chunk_size=500, overlap=75):
    words = text.split()
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
    return chunks
def generate_chunk_id(filename, page, chunk_number, text):
    content = f"{filename}-{page}-{chunk_number}-{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
def process_documents(folder):
    all_chunks = []
    supported = (".pdf", ".docx", ".txt")
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        if not filename.lower().endswith(supported):
            continue
        pages = extract_text(path)
        document_chunks = []
        for page in pages:
            chunks = create_chunks(page["text"])
            for number, chunk_text in enumerate(chunks, start=1):
                chunk_id = generate_chunk_id(
                    filename,
                    page["page"],
                    number,
                    chunk_text
                )
                record = {
                    "chunk_id": chunk_id,
                    "source": filename,
                    "doc_type": os.path.splitext(filename)[1].lower(),
                    "page": page["page"],
                    "chunk_number": number,
                    "text": chunk_text
                }
                all_chunks.append(record)
                document_chunks.append(chunk_text)
        print(f"{filename}: {len(document_chunks)} chunks")
    return all_chunks
def save_chunks(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk) + "\n")
def print_statistics(chunks):
    if not chunks:
        print("No chunks available")
        return
    sizes = [
        len(chunk["text"].split())
        for chunk in chunks
    ]
    average = sum(sizes) / len(sizes)
    print("\nChunk Statistics")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average words: {average:.2f}")
    print(f"Smallest chunk: {min(sizes)} words")
    print(f"Largest chunk: {max(sizes)} words")
if __name__ == "__main__":
    chunks = process_documents("data")
    save_chunks(chunks, "chunks.jsonl")
    print_statistics(chunks)
