import pytest
from fastapi.testclient import TestClient
from http_server import app

client = TestClient(app)

def test_index_documents_tool():
    response = client.post("/mcp", json={
        "tool": "index_documents_tool",
        "args": {"path": "docs", "glob": "*.txt"}
    })
    assert response.status_code == 200
    data = response.json()
    assert "indexed" in data


def test_index_url_tool():
    response = client.post("/mcp", json={
        "tool": "index_url_tool",
        "args": {"url": "http://example.com/test.txt", "doc_id": "test_url_doc"}
    })
    assert response.status_code == 200
    data = response.json()
    assert "indexed" in data


def test_search_tool():
    response = client.post("/mcp", json={
        "tool": "search_tool",
        "args": {"query": "test", "top_k": 5}
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data or "choices" in data
