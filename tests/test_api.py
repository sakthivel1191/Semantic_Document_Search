
import io
import os
import numpy as np
import pytest

from fastapi.testclient import TestClient
from unittest.mock import patch

import src.api as api


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return np.array([1.0] + [0.0] * 383)

        return np.array([
            [1.0] + [0.0] * 383
            for text in texts
        ])


@pytest.fixture
def client(monkeypatch, tmp_path):

    monkeypatch.setattr(
        api,
        "DATA_FOLDER",
        str(tmp_path / "data")
    )

    monkeypatch.setattr(
        api,
        "CHUNKS_FILE",
        str(tmp_path / "chunks.jsonl")
    )

    monkeypatch.setattr(
        api,
        "EMBEDDINGS_FILE",
        str(tmp_path / "embeddings" / "all_embeddings.npy")
    )

    monkeypatch.setattr(
        api,
        "EMBEDDING_FOLDER",
        str(tmp_path / "embeddings")
    )

    monkeypatch.setattr(api, "API_KEY", "test-key")

    os.makedirs(api.DATA_FOLDER, exist_ok=True)
    os.makedirs(api.EMBEDDING_FOLDER, exist_ok=True)

    monkeypatch.setattr(api, "MODEL", FakeModel())

    monkeypatch.setattr(
        api,
        "CHUNKS",
        [
            {
                "chunk_id": "1",
                "source": "policy.pdf",
                "doc_type": ".pdf",
                "page": 1,
                "section": "",
                "chunk_number": 1,
                "text": "Employee leave policy information"
            }
        ]
    )

    monkeypatch.setattr(
        api,
        "VECTORS",
        np.array([[1.0] + [0.0] * 383])
    )

    with patch.object(api, "load_index"):
        with TestClient(api.app) as test_client:
            yield test_client


@pytest.fixture
def headers():
    return {
        "X-API-Key": "test-key"
    }


def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_search(client, headers):

    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": "employee leave policy",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) == 1

    assert data["results"][0]["source"] == "policy.pdf"

    assert data["results"][0]["page"] == 1


def test_search_validation(client, headers):

    response = client.post(
        "/search",
        headers=headers,
        json={
            "query": "policy",
            "top_k": 25
        }
    )

    assert response.status_code == 422


def test_api_key_protection(client):

    response = client.get("/documents")

    assert response.status_code == 401


def test_documents(client, headers):

    response = client.get(
        "/documents",
        headers=headers
    )

    assert response.status_code == 200

    assert response.json()["total_documents"] == 1


def test_unsupported_upload(client, headers):

    response = client.post(
        "/documents",
        headers=headers,
        files={
            "file": (
                "sample.exe",
                io.BytesIO(b"test"),
                "application/octet-stream"
            )
        }
    )

    assert response.status_code == 415
