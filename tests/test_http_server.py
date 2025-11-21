from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from src.servers.mcp_app import api as rest_module


@pytest.fixture(name="client")
def client_fixture():
    return TestClient(rest_module.rest_api)


def test_upsert_document_returns_job_id(monkeypatch, client):
    captured: Dict[str, Any] = {}

    def fake_enqueue(payload, job_type):
        captured["payload"] = payload
        captured["job_type"] = job_type
        return "job-123"

    monkeypatch.setattr(rest_module.worker_mod, "enqueue_job", fake_enqueue)
    response = client.post(
        "/upsert_document",
        json={"uri": "docs/roadmap.md", "text": "Launch goals"},
    )
    assert response.status_code == 200
    assert response.json() == {"job_id": "job-123", "status": "queued"}
    assert captured["job_type"] == "upsert_document"
    assert captured["payload"]["uri"] == "docs/roadmap.md"


def test_index_path_queues_job(monkeypatch, client):
    captured: Dict[str, Any] = {}

    def fake_enqueue(payload, job_type):
        captured["payload"] = payload
        captured["job_type"] = job_type
        return "job-456"

    monkeypatch.setattr(rest_module.worker_mod, "enqueue_job", fake_enqueue)
    response = client.post(
        "/index_path",
        json={"path": "./docs", "glob": "**/*.md"},
    )
    assert response.status_code == 200
    assert response.json()["job_id"] == "job-456"
    assert captured["payload"] == {"path": "./docs", "glob": "**/*.md"}
    assert captured["job_type"] == "index_path"


def test_search_sync_returns_answer(monkeypatch, client):
    async def fake_run_sync(func, *args, **kwargs):  # pylint: disable=unused-argument
        return {"answer": "done", "sources": ["docs/roadmap.md"]}

    monkeypatch.setattr(
        rest_module.anyio.to_thread,
        "run_sync",
        lambda func, *args, **kwargs: fake_run_sync(func, *args, **kwargs),
    )
    response = client.post("/search", json={"query": "anything"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "done"
    assert "sources" in payload
