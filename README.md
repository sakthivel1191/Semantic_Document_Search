
 Semantic Document Search

 Project Description

This project is a semantic document search system built using Python.

It extracts text from PDF, DOCX and TXT documents, divides the text into chunks, generates embeddings and searches documents based on the meaning of a query.

 Technologies Used

- FastAPI
- Sentence Transformers
- NumPy
- PyMuPDF
- python-docx
- Pydantic
- Pytest

 Architecture

text
             Documents
                 |
                 v
          Text Extraction
                 |
                 v
             Chunking
                 |
                 v
          Embedding Model
                 |
                 v
       Local Embedding Files
                 |
                 v
          Semantic Search
                 |
                 v
             FastAPI
                 |
                 v
              Results
```

 Project Structure

```text
data/
embeddings/
src/
tests/
chunks.jsonl
evaluation_queries.json
evaluation_results.json
README.md
```

 Installation

```bash
python -m venv venv
venv\Scripts\activate
```

 Initial Document Indexing

```bash
python -m src.extract
python -m src.chunk
python -m src.embed
```
 Start API

Set API key:

```powershell
$env:API_KEY="my-secret-key"
```

Start server:

```bash
uvicorn src.api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Check API |
| GET | /documents | List documents |
| POST | /search | Search documents |
| POST | /documents | Upload document |
| DELETE | /documents/{source} | Delete document |
| POST | /search/batch | Search multiple queries |
 Search Example

```json
{
    "query": "What is the leave policy?",
    "top_k": 5
}
```

 Evaluation

The evaluation program tests 20 queries.

Metrics:

- Hit Rate@1
- Hit Rate@3
- Hit Rate@5
- Mean Reciprocal Rank

Two chunk configurations are compared.

Run:

```bash
python -m src.evaluation
```

API Testing

```bash
python -m pytest -v
```

## Latency Testing

```bash
python measure_latency.py
```

 Limitations

- Scanned PDFs require OCR.
- Search quality depends on the embedding model.
- Local files are used for storage.
- Semantic similarity does not guarantee that a result answers the question.
- Document extraction may lose some table and formatting information.
- This implementation uses in-memory vectors and is intended for a small project corpus.

 Future Improvements

- Add OCR support.
- Improve chunking.
- Try different embedding models.
- Add a simple web interface.
- Improve search relevance.
