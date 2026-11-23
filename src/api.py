import os
import json
import time
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from src.extract import extract_text
from src.chunk import create_chunks, generate_chunk_id
DATA_FOLDER = "data"
EMBEDDING_FOLDER = "embeddings"
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings/all_embeddings.npy"
API_KEY = os.getenv("API_KEY", "development-key")
MODEL = None
CHUNKS = []
VECTORS = None
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = None
    source: str | None = None
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
class BatchSearchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)
def save_index():
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        for chunk in CHUNKS:
            file.write(json.dumps(chunk) + "\n")
    if VECTORS is None:
        np.save(EMBEDDINGS_FILE, np.empty((0, 384)))
    else:
        np.save(EMBEDDINGS_FILE, VECTORS)
def load_index():
    global CHUNKS, VECTORS
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            CHUNKS = [
                json.loads(line)
                for line in file
                if line.strip()
            ]
    if os.path.exists(EMBEDDINGS_FILE):
        VECTORS = np.load(EMBEDDINGS_FILE)
    else:
        VECTORS = np.empty((0, 384))
    if len(CHUNKS) != len(VECTORS):
        raise ValueError("Chunks and embeddings count mismatch")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = await asyncio.to_thread(
        SentenceTransformer,
        "all-MiniLM-L6-v2"
    )
    load_index()
    yield
app = FastAPI(
    title="Semantic Document Search API",
    description="Search PDF, DOCX and TXT documents using semantic search",
    version="1.0",
    lifespan=lifespan
)
def verify_api_key(x_api_key: str = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
def perform_search(query, top_k, doc_type=None,
                   source=None, min_score=0.0):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not ready"
        )
    if VECTORS is None or len(CHUNKS) != len(VECTORS):
        raise HTTPException(
            status_code=503,
            detail="Search index is unavailable"
        )
    query_vector = MODEL.encode(
        query,
        normalize_embeddings=True
    )
    scores = np.dot(VECTORS, query_vector)
    results = []
    for index, score in enumerate(scores):
        chunk = CHUNKS[index]
        if doc_type and chunk["doc_type"] != doc_type:
            continue
        if source and chunk["source"] != source:
            continue
        if float(score) < min_score:
            continue
        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "section": chunk.get("section", ""),
            "score": round(float(score), 4),
            "text": chunk["text"]
        })
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )
    return results[:top_k]
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "total_chunks": len(CHUNKS),
        "total_vectors": 0 if VECTORS is None else len(VECTORS)
    }
@app.get("/documents")
async def list_documents(api_key: str = Depends(verify_api_key)):
    documents = {}
    for chunk in CHUNKS:
        source = chunk["source"]
        if source not in documents:
            documents[source] = {
                "source": source,
                "doc_type": chunk["doc_type"],
                "chunks": 0
            }
        documents[source]["chunks"] += 1
    return {
        "total_documents": len(documents),
        "documents": list(documents.values())
    }
@app.post("/search")
async def search_documents(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    start = time.perf_counter()
    results = await asyncio.to_thread(
        perform_search,
        request.query,
        request.top_k,
        request.doc_type,
        request.source,
        request.min_score
    )
    latency = (time.perf_counter() - start) * 1000
    return {
        "query": request.query,
        "results": results,
        "latency_ms": round(latency, 2)
    }
@app.post("/search/batch")
async def batch_search(
    request: BatchSearchRequest,
    api_key: str = Depends(verify_api_key)
):
    output = []
    for query in request.queries:
        results = await asyncio.to_thread(
            perform_search,
            query,
            request.top_k
        )
        output.append({
            "query": query,
            "results": results
        })
    return {"results": output}
@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    global CHUNKS, VECTORS
    filename = os.path.basename(file.filename or "")
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    extension = os.path.splitext(filename)[1].lower()
    if extension not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=415,
            detail="Only PDF, DOCX and TXT files are supported"
        )
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )
    path = os.path.join(DATA_FOLDER, filename)
    try:
        with open(path, "wb") as output_file:
            output_file.write(content)
        pages = await asyncio.to_thread(extract_text, path)
        if not pages:
            os.remove(path)
            raise HTTPException(
                status_code=400,
                detail="No readable text found in document"
            )

        new_chunks = []

        for page in pages:
            page_chunks = create_chunks(page["text"])

            for number, text in enumerate(page_chunks, start=1):
                new_chunks.append({
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
                    "chunk_number": number,
                    "text": text
                })

        if not new_chunks:
            os.remove(path)

            raise HTTPException(
                status_code=400,
                detail="Document does not contain extractable chunks"
            )

        new_vectors = await asyncio.to_thread(
            MODEL.encode,
            [chunk["text"] for chunk in new_chunks],
            normalize_embeddings=True
        )

        new_vectors = np.asarray(new_vectors)

        old_chunks = CHUNKS.copy()
        old_vectors = VECTORS.copy()

        keep_indexes = [
            index for index, chunk in enumerate(CHUNKS)
            if chunk["source"] != filename
        ]

        CHUNKS = [CHUNKS[index] for index in keep_indexes]
        VECTORS = VECTORS[keep_indexes]

        CHUNKS.extend(new_chunks)
        VECTORS = np.vstack((VECTORS, new_vectors))

        try:
            save_index()
        except Exception:
            CHUNKS = old_chunks
            VECTORS = old_vectors
            raise

        return {
            "message": "Document indexed successfully",
            "source": filename,
            "chunks": len(new_chunks),
            "total_chunks": len(CHUNKS)
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Document indexing failed: {str(error)}"
        )

@app.delete("/documents/{source}")
async def delete_document(
    source: str,
    api_key: str = Depends(verify_api_key)
):
    global CHUNKS, VECTORS

    matching_indexes = [
        index for index, chunk in enumerate(CHUNKS)
        if chunk["source"] == source
    ]

    if not matching_indexes:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    old_chunks = CHUNKS.copy()
    old_vectors = VECTORS.copy()
    try:
        keep_indexes = [
            index for index, chunk in enumerate(CHUNKS)
            if chunk["source"] != source
        ]
        CHUNKS = [CHUNKS[index] for index in keep_indexes]
        VECTORS = VECTORS[keep_indexes]
        save_index()
        path = os.path.join(DATA_FOLDER, source)
        if os.path.exists(path):
            os.remove(path)
        return {
            "message": "Document deleted successfully",
            "source": source,
            "deleted_chunks": len(matching_indexes)
        }
    except Exception as error:
        CHUNKS = old_chunks
        VECTORS = old_vectors
        raise HTTPException(
            status_code=503,
            detail=f"Document deletion failed: {str(error)}"
        )
