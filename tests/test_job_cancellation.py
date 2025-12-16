import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from src.servers.mcp_app import worker
from src.servers.rest_server import app as rest_app

# --- Worker Tests ---

def test_worker_cancel_all_jobs():
    """Test the core logic of cancelling all jobs in the worker module."""
    # Reset worker state
    worker.JOBS = {}
    worker.CANCELED = set()
    worker.JOB_QUEUE = MagicMock()
    worker.RESULT_QUEUE = MagicMock()
    
    # Setup some dummy jobs
    worker.JOBS = {
        "job1": {"id": "job1", "status": "queued"},
        "job2": {"id": "job2", "status": "running"},
        "job3": {"id": "job3", "status": "completed"},
        "job4": {"id": "job4", "status": "failed"},
    }
    
    # Mock JOB_QUEUE.get_nowait to simulate draining
    # First call returns a job, second call raises Empty
    import queue
    mock_job = {"id": "job5", "status": "queued"}
    worker.JOB_QUEUE.get_nowait.side_effect = [mock_job, queue.Empty]
    
    count = worker.cancel_all_jobs()
    
    # job1 (queued) -> canceled
    # job2 (running) -> canceled
    # job3 (completed) -> unchanged
    # job4 (failed) -> unchanged
    # job5 (in queue) -> canceled (handled by draining logic)
    
    # Note: job5 is added to JOBS and CANCELED during draining
    
    assert count == 2  # Only counts jobs already in JOBS that were active
    assert worker.JOBS["job1"]["status"] == "canceled"
    assert worker.JOBS["job2"]["status"] == "canceled"
    assert worker.JOBS["job3"]["status"] == "completed"
    assert worker.JOBS["job4"]["status"] == "failed"
    
    # Check if job5 was processed and canceled
    assert "job5" in worker.JOBS
    assert worker.JOBS["job5"]["status"] == "canceled"
    assert "job5" in worker.CANCELED
    
    assert "job1" in worker.CANCELED
    assert "job2" in worker.CANCELED

# --- MCP API Tests ---

def test_mcp_api_cancel_all_jobs():
    """Test the MCP API endpoint for cancelling all jobs."""
    from src.servers.mcp_app.api import rest_api
    client = TestClient(rest_api)
    
    with patch("src.servers.mcp_app.worker.cancel_all_jobs") as mock_cancel:
        mock_cancel.return_value = 5
        response = client.post("/jobs/cancel_all")
        assert response.status_code == 200
        assert response.json() == {"status": "canceled", "count": 5}
        mock_cancel.assert_called_once()

# --- REST Server Tests ---

@pytest.mark.asyncio
async def test_rest_api_cancel_all_jobs():
    """Test the REST server endpoint delegating to MCP."""
    # We need to mock _proxy_to_mcp since it's now async
    with patch("src.servers.rest_server._proxy_to_mcp", new_callable=AsyncMock) as mock_proxy:
        mock_proxy.return_value = {"status": "canceled", "count": 10}
        
        # We need to use AsyncClient for async endpoints if we were calling them directly,
        # but TestClient handles async app endpoints by running them in a loop.
        # However, _proxy_to_mcp is internal.
        
        client = TestClient(rest_app)
        response = client.post("/api/jobs/cancel_all")
        
        assert response.status_code == 200
        assert response.json() == {"status": "canceled", "count": 10}
        mock_proxy.assert_called_with("POST", "/rest/jobs/cancel_all")

@pytest.mark.asyncio
async def test_proxy_to_mcp_async():
    """Test the asynchronous _proxy_to_mcp function."""
    from src.servers.rest_server import _proxy_to_mcp
    
    # Mock httpx.AsyncClient
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_client.request.return_value = mock_response
        
        result = await _proxy_to_mcp("GET", "/test/path")
        assert result == {"key": "value"}
        
        # Verify call args
        # Note: _proxy_to_mcp constructs the URL. We assume defaults here.
        # It loops through configured URLs.
        args, kwargs = mock_client.request.call_args
        assert args[0] == "GET"
        assert "/test/path" in args[1] # URL should contain the path
        
# --- Integration Check: Async API Endpoints ---
# Verify that other endpoints converted to async still work with the async proxy

@pytest.mark.asyncio
async def test_api_jobs_async_delegation():
    """Test that the async api_jobs endpoint correctly awaits the proxy."""
    with patch("src.servers.rest_server._proxy_to_mcp", new_callable=AsyncMock) as mock_proxy:
        mock_proxy.return_value = {"jobs": []}
        
        client = TestClient(rest_app)
        response = client.get("/api/jobs")
        
        assert response.status_code == 200
        assert response.json() == {"jobs": []}
        mock_proxy.assert_called_with("GET", "/rest/jobs")
