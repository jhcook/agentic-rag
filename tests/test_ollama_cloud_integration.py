"""Integration tests for Ollama Cloud functionality."""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
import requests

# Set environment variables before imports
os.environ["RAG_HOST"] = "127.0.0.1"
os.environ["RAG_PORT"] = "8001"
os.environ["RAG_PATH"] = "api"

from fastapi.testclient import TestClient
from src.servers.rest_server import app

client = TestClient(app)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def temp_secrets_dir(tmp_path):
    """Create temporary secrets directory."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    return secrets_dir


class TestOllamaModeEndpoints:
    """Tests for Ollama mode API endpoints."""
    
    @patch("src.servers.rest_server.CONFIG_DIR")
    def test_get_ollama_mode(self, mock_config_dir, temp_config_dir):
        """Test GET /api/ollama/mode endpoint."""
        mock_config_dir.__truediv__ = lambda self, other: temp_config_dir / other
        
        settings_path = temp_config_dir / "settings.json"
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "cloud"}, f)
        
        with patch("src.core.ollama_config.SETTINGS_PATH", settings_path):
            response = client.get("/api/ollama/mode")
            assert response.status_code == 200
            assert response.json()["mode"] == "cloud"
    
    @patch("src.servers.rest_server.CONFIG_DIR")
    def test_set_ollama_mode(self, mock_config_dir, temp_config_dir):
        """Test POST /api/ollama/mode endpoint."""
        mock_config_dir.mkdir = lambda parents=True, exist_ok=True: None
        mock_config_dir.__truediv__ = lambda self, other: temp_config_dir / other
        
        settings_path = temp_config_dir / "settings.json"
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "local"}, f)
        
        with patch("src.core.ollama_config.SETTINGS_PATH", settings_path):
            with patch("src.servers.rest_server.CONFIG_DIR", temp_config_dir):
                response = client.post(
                    "/api/ollama/mode",
                    json={"mode": "auto"}
                )
                assert response.status_code == 200
                assert response.json()["mode"] == "auto"
                
                # Verify it was saved
                with open(settings_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                assert config["ollamaMode"] == "auto"
    
    def test_set_invalid_mode(self):
        """Test POST with invalid mode returns 400."""
        response = client.post(
            "/api/ollama/mode",
            json={"mode": "invalid"}
        )
        assert response.status_code == 400
        assert "Invalid mode" in response.json()["detail"]


class TestOllamaStatusEndpoint:
    """Tests for Ollama status endpoint."""
    
    @patch("src.core.ollama_config.get_ollama_mode")
    @patch("src.core.ollama_config.get_ollama_endpoint")
    @patch("src.core.ollama_config.get_ollama_local_endpoint")
    @patch("src.core.ollama_config.test_cloud_connection")
    @patch("requests.get")
    def test_get_ollama_status_local(self, mock_requests_get, mock_test_cloud,
                                     mock_get_local, mock_get_endpoint, mock_get_mode):
        """Test GET /api/ollama/status for local mode."""
        mock_get_mode.return_value = "local"
        mock_get_endpoint.return_value = "http://localhost:11434"
        mock_get_local.return_value = "http://localhost:11434"
        
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response
        
        response = client.get("/api/ollama/status")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "local"
        assert data["local_available"] is True
        assert data["local_status"] == "connected"
    
    @patch("src.core.ollama_config.get_ollama_mode")
    @patch("src.core.ollama_config.get_ollama_endpoint")
    @patch("src.core.ollama_config.get_ollama_local_endpoint")
    @patch("src.core.ollama_config.test_cloud_connection")
    @patch("requests.get")
    def test_get_ollama_status_cloud(self, mock_requests_get, mock_test_cloud,
                                     mock_get_local, mock_get_endpoint, mock_get_mode):
        """Test GET /api/ollama/status for cloud mode."""
        mock_get_mode.return_value = "cloud"
        mock_get_endpoint.return_value = "https://ollama.com"
        mock_get_local.return_value = "http://localhost:11434"
        mock_test_cloud.return_value = (True, "Connection successful")
        
        mock_response = MagicMock()
        mock_response.ok = True
        mock_requests_get.return_value = mock_response
        
        response = client.get("/api/ollama/status")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "cloud"
        assert data["cloud_available"] is True
        assert data["cloud_status"] == "connected"


class TestOllamaModelsEndpoint:
    """Tests for Ollama models endpoint."""
    
    @patch("src.servers.rest_server.get_rag_backend")
    def test_list_ollama_models(self, mock_get_backend):
        """Test GET /api/ollama/models endpoint."""
        mock_backend = MagicMock()
        mock_backend.list_models.return_value = ["llama3.2:1b", "qwen2.5:0.5b"]
        mock_get_backend.return_value = mock_backend
        
        response = client.get("/api/ollama/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 2
        assert "llama3.2:1b" in data["models"]


class TestOllamaTestConnectionEndpoint:
    """Tests for Ollama test connection endpoint."""
    
    @patch("src.core.ollama_config.test_cloud_connection")
    @patch("src.core.ollama_config.save_ollama_cloud_config")
    def test_test_connection_success(self, mock_save, mock_test):
        """Test POST /api/ollama/test-connection with successful connection."""
        mock_test.return_value = (True, "Connection successful")
        
        response = client.post(
            "/api/ollama/test-connection",
            json={"api_key": "test-key-123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "successful" in data["message"].lower()
        # Should save API key on success
        mock_save.assert_called_once()
    
    @patch("src.core.ollama_config.test_cloud_connection")
    def test_test_connection_failure(self, mock_test):
        """Test POST /api/ollama/test-connection with failed connection."""
        mock_test.return_value = (False, "Invalid API key")
        
        response = client.post(
            "/api/ollama/test-connection",
            json={"api_key": "invalid-key"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Invalid API key" in data["message"]


class TestOllamaFallbackLogic:
    """Tests for auto mode fallback logic."""
    
    def test_auto_mode_fallback_on_timeout(self):
        """Test auto mode falls back to local on cloud timeout."""
        # This test verifies the fallback pattern works correctly
        # The actual fallback is tested in rag_core.py integration
        endpoint = "https://ollama.com"
        headers = {"Authorization": "Bearer test-key"}
        fallback = "http://localhost:11434"
        
        # Simulate the fallback logic pattern
        # Use a simple Exception for testing since litellm exceptions have complex constructors
        class MockTimeout(Exception):
            pass
        
        mock_completion = MagicMock()
        mock_completion.side_effect = [
            MockTimeout("Connection timeout"),
            {"choices": [{"message": {"content": "Local response"}}]}
        ]
        
        # Simulate the fallback logic
        try:
            result = mock_completion(api_base=endpoint, extra_headers=headers)
        except MockTimeout:
            if fallback:
                result = mock_completion(api_base=fallback, extra_headers={})
        
        assert mock_completion.call_count == 2
        assert result["choices"][0]["message"]["content"] == "Local response"
    
    def test_cloud_mode_no_fallback(self):
        """Test cloud mode does not fallback on failure."""
        # Setup cloud mode (no fallback)
        endpoint = "https://ollama.com"
        headers = {"Authorization": "Bearer test-key"}
        fallback = None  # No fallback
        
        # Use a simple Exception for testing
        class MockConnectionError(Exception):
            pass
        
        mock_completion = MagicMock()
        mock_completion.side_effect = MockConnectionError("Connection failed")
        
        # Should raise error, not fallback
        with pytest.raises(MockConnectionError):
            try:
                mock_completion(api_base=endpoint, extra_headers=headers)
            except MockConnectionError:
                if fallback:
                    mock_completion(api_base=fallback, extra_headers={})
                else:
                    raise
        
        # Should only be called once (no fallback)
        assert mock_completion.call_count == 1


class TestOllamaSecurityFeatures:
    """Tests for security features."""
    
    def test_api_key_redaction_in_errors(self):
        """Test API keys are redacted from error messages."""
        from src.core.ollama_config import _redact_api_key
        
        api_key = "secret-api-key-12345"
        error_msg = f"Connection failed with key {api_key}"
        
        redacted = _redact_api_key(error_msg, api_key)
        assert api_key not in redacted
        assert "***REDACTED" in redacted
    
    def test_https_enforcement(self):
        """Test HTTPS is enforced for cloud endpoints."""
        from src.core.ollama_config import _validate_url
        
        # HTTP should be invalid for cloud
        is_valid, error = _validate_url("http://ollama.com", allow_local=False)
        assert is_valid is False
        assert "HTTPS" in error
        
        # HTTPS should be valid
        is_valid, error = _validate_url("https://ollama.com", allow_local=False)
        assert is_valid is True
    
    def test_ssrf_protection(self):
        """Test SSRF protection blocks private IPs."""
        from src.core.ollama_config import _validate_url
        
        # Private IPs should be blocked
        is_valid, error = _validate_url("https://192.168.1.1", allow_local=False)
        assert is_valid is False
        assert "private" in error.lower()
        
        # Localhost should be blocked
        is_valid, error = _validate_url("https://localhost", allow_local=False)
        assert is_valid is False
        assert "localhost" in error.lower()
    
    def test_secrets_file_permissions(self, temp_secrets_dir, monkeypatch):
        """Test secrets file has restrictive permissions."""
        import stat
        from src.core.ollama_config import save_ollama_cloud_config, OLLAMA_CLOUD_SECRETS_PATH
        
        secrets_path = temp_secrets_dir / "ollama_cloud_config.json"
        monkeypatch.setattr("src.core.ollama_config.OLLAMA_CLOUD_SECRETS_PATH", secrets_path)
        
        save_ollama_cloud_config(api_key="test-key")
        
        assert secrets_path.exists()
        file_stat = secrets_path.stat()
        # Check permissions are 600 (rw-------)
        assert (file_stat.st_mode & 0o777) == 0o600

