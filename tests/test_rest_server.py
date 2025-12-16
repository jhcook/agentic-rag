import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open, AsyncMock

# Set environment variables before importing the app
os.environ["RAG_HOST"] = "127.0.0.1"
os.environ["RAG_PORT"] = "8001"
os.environ["RAG_PATH"] = "api"

# Mock heavy dependencies before importing rest_server
# This prevents the actual backend from initializing (which loads models, etc.)
patcher_factory = patch("src.core.factory.get_rag_backend")
mock_get_backend = patcher_factory.start()
mock_backend_instance = MagicMock()
mock_get_backend.return_value = mock_backend_instance

patcher_auth = patch("src.core.google_auth.GoogleAuthManager")
mock_auth_class = patcher_auth.start()
mock_auth_instance = MagicMock()
mock_auth_class.return_value = mock_auth_instance

# Now import the app
from src.servers.rest_server import app, _proxy_to_mcp

# Stop patchers
patcher_factory.stop()
patcher_auth.stop()

from fastapi.testclient import TestClient
from pathlib import Path

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


@pytest.mark.parametrize(
    "payload",
    [
        {"content": "hello"},
        {"answer": "hello"},
        {"response": "hello"},
        {"grounded_answer": "hello"},
    ],
)
def test_chat_persists_assistant_text_across_backends(tmp_path: Path, mock_backend, payload):
    from src.core.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.db")
    with patch("src.servers.rest_server.chat_store", store):
        mock_backend.chat.return_value = payload
        resp = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi", "display_content": "hi"}],
            },
        )
        assert resp.status_code == 200
        session_id = resp.json().get("session_id")
        assert session_id

        msgs = client.get(f"/api/chat/history/{session_id}").json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "hi"
        assert msgs[0].get("display_content") == "hi"
        assert msgs[1]["content"] == "hello"


def test_chat_does_not_persist_error_payload_as_assistant(tmp_path: Path, mock_backend):
    from src.core.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.db")
    with patch("src.servers.rest_server.chat_store", store):
        mock_backend.chat.return_value = {"error": "boom"}
        resp = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi", "display_content": "hi"}],
            },
        )
        assert resp.status_code == 200
        session_id = resp.json().get("session_id")
        assert session_id

        msgs = client.get(f"/api/chat/history/{session_id}").json()
        assert [m["role"] for m in msgs] == ["user"]


def test_grounded_answer_persists_into_chat_history(tmp_path: Path, mock_backend):
    from src.core.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.db")
    with patch("src.servers.rest_server.chat_store", store):
        mock_backend.grounded_answer.return_value = {
            "grounded_answer": "A",
            "citations": ["u1"],
        }
        resp = client.post("/api/grounded_answer", json={"question": "Q?"})
        assert resp.status_code == 200
        session_id = resp.json().get("session_id")
        assert session_id

        msgs = client.get(f"/api/chat/history/{session_id}").json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "Q?"
        assert msgs[1]["content"] == "A"
        assert msgs[1].get("kind") == "assistant_grounded"
        assert msgs[1].get("sources") == ["u1"]


def test_delete_message_persists_in_history(tmp_path: Path, mock_backend):
    from src.core.chat_store import ChatStore

    store = ChatStore(tmp_path / "chat.db")
    with patch("src.servers.rest_server.chat_store", store):
        mock_backend.chat.return_value = {"content": "hello"}
        resp = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi", "display_content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        session_id = body.get("session_id")
        assert session_id

        msgs = client.get(f"/api/chat/history/{session_id}").json()
        assert len(msgs) == 2
        assistant_id = msgs[1]["id"]

        del_resp = client.delete(f"/api/chat/history/{session_id}/messages/{assistant_id}")
        assert del_resp.status_code == 200

        msgs2 = client.get(f"/api/chat/history/{session_id}").json()
        assert [m["role"] for m in msgs2] == ["user"]

def test_flush_cache(mock_backend):
    mock_backend.flush_cache.return_value = {"status": "flushed", "db_removed": True, "documents": 0}
    response = client.post("/api/flush_cache")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "flushed"
    assert data["db_removed"] is True
    mock_backend.flush_cache.assert_called_once()

@pytest.mark.asyncio
async def test_api_upsert_async_delegation():
    """Test that the async api_upsert endpoint correctly awaits the proxy."""
    with patch("src.servers.rest_server._proxy_to_mcp", new_callable=AsyncMock) as mock_proxy:
        mock_proxy.return_value = {"job_id": "test-job-123", "status": "queued"}
        
        client = TestClient(app)
        response = client.post("/api/upsert_document", json={
            "uri": "test.txt",
            "text": "hello world"
        })
        
        assert response.status_code == 200
        assert response.json() == {"job_id": "test-job-123", "status": "queued"}
        mock_proxy.assert_called_with("POST", "/rest/upsert_document", {"uri": "test.txt", "text": "hello world", "binary_base64": None})

@pytest.mark.asyncio
async def test_api_index_url_async_delegation():
    """Test that the async api_index_url endpoint correctly awaits the proxy."""
    with patch("src.servers.rest_server._proxy_to_mcp", new_callable=AsyncMock) as mock_proxy:
        mock_proxy.return_value = {"job_id": "url-job-456", "status": "queued"}
        
        client = TestClient(app)
        response = client.post("/api/index_url", json={
            "url": "http://example.com/doc.pdf"
        })
        
        assert response.status_code == 200
        assert response.json() == {"job_id": "url-job-456", "status": "queued"}
        mock_proxy.assert_called_with("POST", "/rest/index_url", {"url": "http://example.com/doc.pdf", "doc_id": None})

@pytest.mark.asyncio
async def test_api_cancel_job_async_delegation():
    """Test that the async api_cancel_job endpoint correctly awaits the proxy."""
    with patch("src.servers.rest_server._proxy_to_mcp", new_callable=AsyncMock) as mock_proxy:
        job_id = "cancel-job-789"
        mock_proxy.return_value = {"status": "canceled", "id": job_id}
        
        client = TestClient(app)
        response = client.post(f"/api/jobs/{job_id}/cancel")
        
        assert response.status_code == 200
        assert response.json() == {"status": "canceled", "id": job_id}
        mock_proxy.assert_called_with("POST", f"/rest/jobs/{job_id}/cancel")

@pytest.mark.asyncio
async def test_proxy_to_mcp_httpx_usage():
    # We want to mock httpx.AsyncClient to verify arguments passed to its constructor
    with patch("httpx.AsyncClient") as MockClient:
        # Setup the mock client instance
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        
        # Setup the request method return value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_instance.request.return_value = mock_response

        # Mock get_ca_bundle_path
        with patch("src.servers.rest_server.get_ca_bundle_path", return_value="/path/to/cert"):
             # Call the function under test
            await _proxy_to_mcp("GET", "/test")

        # Verify AsyncClient was initialized with verify argument
        MockClient.assert_called_once()
        call_kwargs = MockClient.call_args.kwargs
        assert "verify" in call_kwargs, "AsyncClient expected to be called with 'verify' kwarg"
        assert call_kwargs["verify"] == "/path/to/cert"

        # Verify request() was called WITHOUT verify argument
        mock_instance.request.assert_called()
        request_kwargs = mock_instance.request.call_args.kwargs
        assert "verify" not in request_kwargs, "request() should NOT be called with 'verify' kwarg"
