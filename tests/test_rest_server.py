import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

# Set environment variables before importing the app
os.environ["RAG_HOST"] = "127.0.0.1"
os.environ["RAG_PORT"] = "8001"
os.environ["RAG_PATH"] = "api"

# Mock heavy dependencies before importing rest_server
# This prevents the actual backend from initializing (which loads models, FAISS, etc.)
patcher_factory = patch("src.core.factory.get_rag_backend")
mock_get_backend = patcher_factory.start()
mock_backend_instance = MagicMock()
mock_get_backend.return_value = mock_backend_instance

patcher_auth = patch("src.core.google_auth.GoogleAuthManager")
mock_auth_class = patcher_auth.start()
mock_auth_instance = MagicMock()
mock_auth_class.return_value = mock_auth_instance

# Now import the app
from src.servers.rest_server import app

# Stop patchers
patcher_factory.stop()
patcher_auth.stop()

from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture
def mock_backend():
    # The backend in rest_server is already our mock_backend_instance
    # But we can configure it further here or use a new patch if we want to be safe
    # Since 'backend' is a global in rest_server, we can patch it directly there
    with patch("src.servers.rest_server.backend", mock_backend_instance) as mock:
        # Reset mocks
        mock.reset_mock()
        # Default return values
        mock.get_stats.return_value = {
            "status": "ok",
            "documents": 10,
            "vectors": 10,
            "memory_mb": 100,
            "total_size_bytes": 1000,
            "store_file_bytes": 500
        }
        mock.list_documents.return_value = [{"uri": "doc1.txt", "size": 100}]
        mock.search.return_value = {"answer": "test answer", "sources": []}
        yield mock

@pytest.fixture
def mock_auth_manager():
    with patch("src.servers.rest_server.auth_manager", mock_auth_instance) as mock:
        yield mock

def test_health_check(mock_backend):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["documents"] == 10

def test_get_documents(mock_backend):
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data["documents"]) == 1
    assert data["documents"][0]["uri"] == "doc1.txt"

def test_app_config_endpoints():
    # Patch _append_access_log to avoid file operations during this test
    # This is necessary because we mock 'open' below, which breaks the logging in middleware
    with patch("src.servers.rest_server._append_access_log"):
        # Test GET config (mocking file read)
        with patch("builtins.open", mock_open(read_data='{"model": "test-model"}')):
            with patch("pathlib.Path.exists", return_value=True):
                response = client.get("/api/config/app")
                assert response.status_code == 200
                assert response.json() == {"model": "test-model"}

        # Test POST config (mocking file write)
        with patch("builtins.open", mock_open()) as mock_file:
            config_data = {
                "apiEndpoint": "http://localhost:11434",
                "model": "llama3",
                "embeddingModel": "nomic-embed-text",
                "temperature": "0.7",
                "topP": "0.9",
                "topK": "40",
                "repeatPenalty": "1.1",
                "seed": "42",
                "numCtx": "4096",
                "mcpHost": "localhost",
                "mcpPort": "8000",
                "mcpPath": "/mcp",
                "ragHost": "localhost",
                "ragPort": "8001",
                "ragPath": "api"
            }
            response = client.post("/api/config/app", json=config_data)
            assert response.status_code == 200
            assert response.json() == {"status": "saved"}
            # Verify write was called
            mock_file.assert_called()

def test_logs_endpoint():
    # Mock log file reading
    log_content = "line 1\nline 2\nline 3"
    with patch("src.servers.rest_server._append_access_log"):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=log_content)):
                response = client.get("/api/logs/rest?lines=2")
                assert response.status_code == 200
                data = response.json()
                # The logic in rest_server uses f.readlines()
                # mock_open(read_data=...) makes readlines() return the lines
                # We requested 2 lines, so we should get 2
                assert len(data["lines"]) == 2
                assert data["lines"] == ["line 2", "line 3"] 

            
def test_search_endpoint(mock_backend):
    response = client.post("/api/search", json={"query": "test query"})
    assert response.status_code == 200
    assert response.json()["answer"] == "test answer"
    mock_backend.search.assert_called_with("test query")

def test_grounded_answer_endpoint(mock_backend):
    mock_backend.grounded_answer.return_value = {"answer": "grounded", "sources": []}
    response = client.post("/api/grounded_answer", json={"question": "test?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "grounded"
    mock_backend.grounded_answer.assert_called()
