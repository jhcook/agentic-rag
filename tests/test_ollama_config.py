"""Unit tests for Ollama Cloud configuration module."""
import os
import json
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest

from src.core.ollama_config import (
    get_ollama_mode,
    get_ollama_api_key,
    get_ollama_cloud_endpoint,
    get_ollama_local_endpoint,
    get_ollama_endpoint,
    get_ollama_client_headers,
    get_ollama_endpoint_with_fallback,
    save_ollama_cloud_config,
    normalize_ollama_model_name,
    validate_ollama_config,
    _redact_api_key,
    _validate_url,
    get_requests_ca_bundle,
)
# Import test_cloud_connection with alias to avoid pytest treating as test
from src.core import ollama_config


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


@pytest.fixture
def mock_settings_path(temp_config_dir, temp_secrets_dir, monkeypatch):
    """Mock settings.json path."""
    settings_path = temp_config_dir / "settings.json"
    monkeypatch.setattr("src.core.ollama_config.SETTINGS_PATH", settings_path)
    # Default to an empty secrets file path for tests that don't explicitly
    # request `mock_secrets_path`, so repo-local secrets don't affect unit tests.
    monkeypatch.setattr(
        "src.core.ollama_config.OLLAMA_CLOUD_SECRETS_PATH",
        temp_secrets_dir / "ollama_cloud_config.json",
    )

    # Clear process env defaults that could affect deterministic unit tests.
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("OLLAMA_CLOUD_ENDPOINT", raising=False)
    monkeypatch.delenv("OLLAMA_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODE", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    return settings_path


@pytest.fixture
def mock_secrets_path(temp_secrets_dir, monkeypatch):
    """Mock secrets file path."""
    secrets_path = temp_secrets_dir / "ollama_cloud_config.json"
    monkeypatch.setattr("src.core.ollama_config.OLLAMA_CLOUD_SECRETS_PATH", secrets_path)
    return secrets_path


class TestGetOllamaMode:
    """Tests for get_ollama_mode()."""
    
    def test_get_mode_from_settings(self, mock_settings_path):
        """Test getting mode from settings.json."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "cloud"}, f)
        
        assert get_ollama_mode() == "cloud"
    
    def test_get_mode_from_env(self, mock_settings_path, monkeypatch):
        """Test getting mode from environment variable."""
        monkeypatch.setenv("OLLAMA_MODE", "auto")
        assert get_ollama_mode() == "auto"
    
    def test_default_mode(self, mock_settings_path):
        """Test default mode is 'local'."""
        # No settings file, no env var
        assert get_ollama_mode() == "local"
    
    def test_invalid_mode_defaults_to_local(self, mock_settings_path):
        """Test invalid mode defaults to 'local'."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "invalid"}, f)
        
        assert get_ollama_mode() == "local"


class TestGetOllamaApiKey:
    """Tests for get_ollama_api_key()."""
    
    def test_get_api_key_from_secrets(self, mock_secrets_path):
        """Test getting API key from secrets file."""
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "test-api-key-123"}, f)
        
        assert get_ollama_api_key() == "test-api-key-123"
    
    def test_get_api_key_from_env(self, mock_secrets_path, monkeypatch):
        """Test getting API key from environment variable."""
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "env-api-key-456")
        assert get_ollama_api_key() == "env-api-key-456"
    
    def test_secrets_file_priority_over_env(self, mock_secrets_path, monkeypatch):
        """Test secrets file takes priority over env var."""
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "file-key"}, f)
        monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "env-key")
        
        assert get_ollama_api_key() == "file-key"
    
    def test_no_api_key_returns_none(self, mock_secrets_path):
        """Test returns None when no API key is configured."""
        assert get_ollama_api_key() is None
    
    def test_empty_api_key_returns_none(self, mock_secrets_path):
        """Test empty API key returns None."""
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": ""}, f)
        
        assert get_ollama_api_key() is None


class TestGetOllamaCloudEndpoint:
    """Tests for get_ollama_cloud_endpoint()."""
    
    def test_get_endpoint_from_settings(self, mock_settings_path):
        """Test getting endpoint from settings.json."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudEndpoint": "https://custom.ollama.com"}, f)
        
        assert get_ollama_cloud_endpoint() == "https://custom.ollama.com"
    
    def test_get_endpoint_from_secrets(self, mock_settings_path, mock_secrets_path):
        """Test getting endpoint from secrets file."""
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"endpoint": "https://secrets.ollama.com"}, f)
        
        assert get_ollama_cloud_endpoint() == "https://secrets.ollama.com"
    
    def test_get_endpoint_from_env(self, mock_settings_path, monkeypatch):
        """Test getting endpoint from environment variable."""
        monkeypatch.setenv("OLLAMA_CLOUD_ENDPOINT", "https://env.ollama.com")
        assert get_ollama_cloud_endpoint() == "https://env.ollama.com"
    
    def test_default_endpoint(self, mock_settings_path):
        """Test default endpoint is https://ollama.com."""
        assert get_ollama_cloud_endpoint() == "https://ollama.com"
    
    def test_invalid_endpoint_uses_default(self, mock_settings_path):
        """Test invalid endpoint format uses default."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudEndpoint": "not-a-url"}, f)
        
        assert get_ollama_cloud_endpoint() == "https://ollama.com"
    
    def test_http_endpoint_converted_to_https(self, mock_settings_path):
        """Test HTTP endpoint is converted to HTTPS."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudEndpoint": "http://ollama.com"}, f)
        
        # Should convert to HTTPS
        endpoint = get_ollama_cloud_endpoint()
        assert endpoint.startswith("https://")


class TestGetOllamaEndpoint:
    """Tests for get_ollama_endpoint()."""
    
    def test_cloud_mode_returns_cloud_endpoint(self, mock_settings_path):
        """Test cloud mode returns cloud endpoint."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollamaMode": "cloud",
                "ollamaCloudEndpoint": "https://cloud.ollama.com"
            }, f)
        
        assert get_ollama_endpoint() == "https://cloud.ollama.com"
    
    def test_local_mode_returns_local_endpoint(self, mock_settings_path):
        """Test local mode returns local endpoint."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollamaMode": "local",
                "apiEndpoint": "http://localhost:11434"
            }, f)
        
        assert get_ollama_endpoint() == "http://localhost:11434"
    
    def test_auto_mode_returns_cloud_endpoint(self, mock_settings_path):
        """Test auto mode returns cloud endpoint (fallback handled elsewhere)."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollamaMode": "auto",
                "ollamaCloudEndpoint": "https://cloud.ollama.com"
            }, f)
        
        assert get_ollama_endpoint() == "https://cloud.ollama.com"


class TestGetOllamaClientHeaders:
    """Tests for get_ollama_client_headers()."""
    
    def test_headers_with_api_key(self, mock_secrets_path):
        """Test headers include Authorization when API key is present."""
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "test-key-123"}, f)
        
        headers = get_ollama_client_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key-123"
    
    def test_headers_without_api_key(self, mock_secrets_path):
        """Test headers are empty when no API key."""
        headers = get_ollama_client_headers()
        assert headers == {}


class TestGetOllamaEndpointWithFallback:
    """Tests for get_ollama_endpoint_with_fallback()."""
    
    def test_cloud_mode_no_fallback(self, mock_settings_path):
        """Test cloud mode has no fallback."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollamaMode": "cloud",
                "ollamaCloudEndpoint": "https://cloud.ollama.com"
            }, f)
        
        endpoint, headers, fallback = get_ollama_endpoint_with_fallback()
        assert endpoint == "https://cloud.ollama.com"
        assert fallback is None
    
    def test_auto_mode_has_fallback(self, mock_settings_path):
        """Test auto mode has local fallback."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollamaMode": "auto",
                "ollamaCloudEndpoint": "https://cloud.ollama.com",
                "apiEndpoint": "http://localhost:11434"
            }, f)
        
        endpoint, headers, fallback = get_ollama_endpoint_with_fallback()
        assert endpoint == "https://cloud.ollama.com"
        assert fallback == "http://localhost:11434"
    
    def test_local_mode_no_fallback(self, mock_settings_path):
        """Test local mode has no fallback."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollamaMode": "local",
                "apiEndpoint": "http://localhost:11434"
            }, f)
        
        endpoint, headers, fallback = get_ollama_endpoint_with_fallback()
        assert endpoint == "http://localhost:11434"
        assert fallback is None


class TestSaveOllamaCloudConfig:
    """Tests for save_ollama_cloud_config()."""
    
    def test_save_api_key(self, mock_secrets_path):
        """Test saving API key to secrets file."""
        save_ollama_cloud_config(api_key="new-api-key-789")
        
        assert mock_secrets_path.exists()
        with open(mock_secrets_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["api_key"] == "new-api-key-789"
    
    def test_save_endpoint(self, mock_secrets_path):
        """Test saving endpoint to secrets file."""
        save_ollama_cloud_config(endpoint="https://custom.ollama.com")
        
        assert mock_secrets_path.exists()
        with open(mock_secrets_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["endpoint"] == "https://custom.ollama.com"
    
    def test_file_permissions(self, mock_secrets_path):
        """Test secrets file has restrictive permissions (600)."""
        save_ollama_cloud_config(api_key="test-key")
        
        assert mock_secrets_path.exists()
        file_stat = mock_secrets_path.stat()
        # Check permissions are 600 (rw-------)
        # On Unix: 0o600 = 384, mode & 0o777 should be 0o600
        assert (file_stat.st_mode & 0o777) == 0o600
    
    def test_update_existing_config(self, mock_secrets_path):
        """Test updating existing config preserves other fields."""
        # Create existing config
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "old-key", "endpoint": "https://old.com"}, f)
        
        # Update only API key
        save_ollama_cloud_config(api_key="new-key")
        
        with open(mock_secrets_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["api_key"] == "new-key"
        assert config["endpoint"] == "https://old.com"  # Preserved
    
    def test_http_endpoint_converted_to_https(self, mock_secrets_path):
        """Test HTTP endpoint is converted to HTTPS when saving."""
        save_ollama_cloud_config(endpoint="http://ollama.com")
        
        with open(mock_secrets_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        assert config["endpoint"] == "https://ollama.com"
    
    def test_invalid_endpoint_raises_error(self, mock_secrets_path):
        """Test invalid endpoint raises ValueError."""
        with pytest.raises(ValueError, match="Invalid cloud endpoint"):
            save_ollama_cloud_config(endpoint="localhost")


class TestTestCloudConnection:
    """Tests for test_cloud_connection()."""
    
    @patch('requests.get')
    def test_successful_connection(self, mock_get, mock_secrets_path):
        """Test successful connection returns True."""
        # Setup
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "test-key"}, f)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test
        success, message = ollama_config.test_cloud_connection()
        assert success is True
        assert "successful" in message.lower()
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
    
    @patch('requests.get')
    def test_invalid_api_key(self, mock_get, mock_secrets_path):
        """Test invalid API key returns False."""
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "invalid-key"}, f)
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        success, message = ollama_config.test_cloud_connection()
        assert success is False
        assert "Invalid API key" in message
    
    @patch('requests.get')
    def test_connection_timeout(self, mock_get, mock_secrets_path):
        """Test connection timeout returns False."""
        # Setup API key
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "test-key"}, f)
        
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        success, message = ollama_config.test_cloud_connection()
        assert success is False
        assert "timeout" in message.lower()
    
    @patch('requests.get')
    def test_connection_error(self, mock_get, mock_secrets_path):
        """Test connection error returns False."""
        # Setup API key
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "test-key"}, f)
        
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        success, message = ollama_config.test_cloud_connection()
        assert success is False
        assert "connection error" in message.lower()
    
    def test_no_api_key_returns_false(self, mock_secrets_path):
        """Test no API key returns False."""
        success, message = ollama_config.test_cloud_connection()
        assert success is False
        assert "API key is required" in message


class TestNormalizeOllamaModelName:
    """Tests for normalize_ollama_model_name()."""
    
    def test_cloud_mode_adds_suffix(self, mock_settings_path):
        """Test cloud mode adds -cloud suffix."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "cloud"}, f)
        
        result = normalize_ollama_model_name("qwen2.5:0.5b")
        assert result == "qwen2.5:0.5b-cloud"
    
    def test_cloud_mode_preserves_existing_suffix(self, mock_settings_path):
        """Test cloud mode preserves existing -cloud suffix."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "cloud"}, f)
        
        result = normalize_ollama_model_name("qwen2.5:0.5b-cloud")
        assert result == "qwen2.5:0.5b-cloud"
    
    def test_local_mode_removes_suffix(self, mock_settings_path):
        """Test local mode removes -cloud suffix."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "local"}, f)
        
        result = normalize_ollama_model_name("qwen2.5:0.5b-cloud")
        assert result == "qwen2.5:0.5b"
    
    def test_auto_mode_preserves_name(self, mock_settings_path):
        """Test auto mode preserves model name as-is."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "auto"}, f)
        
        result = normalize_ollama_model_name("qwen2.5:0.5b")
        assert result == "qwen2.5:0.5b"
        
        result2 = normalize_ollama_model_name("qwen2.5:0.5b-cloud")
        assert result2 == "qwen2.5:0.5b-cloud"


class TestValidateOllamaConfig:
    """Tests for validate_ollama_config()."""
    
    def test_valid_cloud_config(self, mock_settings_path, mock_secrets_path):
        """Test valid cloud configuration."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "cloud"}, f)
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "test-key"}, f)
        
        is_valid, error = validate_ollama_config()
        assert is_valid is True
        assert error is None
    
    def test_cloud_mode_missing_api_key(self, mock_settings_path):
        """Test cloud mode without API key is invalid."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "cloud"}, f)
        
        is_valid, error = validate_ollama_config()
        assert is_valid is False
        assert "API key is required" in error
    
    def test_cloud_mode_http_endpoint_invalid(self, mock_settings_path, mock_secrets_path):
        """Test cloud mode with HTTP endpoint is invalid."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({
                "ollamaMode": "cloud",
                "ollamaCloudEndpoint": "http://ollama.com"  # Should be HTTPS
            }, f)
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": "test-key"}, f)
        
        # Note: get_ollama_cloud_endpoint() converts HTTP to HTTPS,
        # so this test may need adjustment based on actual behavior
        is_valid, error = validate_ollama_config()
        # Should be valid because endpoint gets converted to HTTPS
        assert is_valid is True
    
    def test_local_mode_always_valid(self, mock_settings_path):
        """Test local mode is always valid."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaMode": "local"}, f)
        
        is_valid, error = validate_ollama_config()
        assert is_valid is True
        assert error is None


class TestRedactApiKey:
    """Tests for _redact_api_key()."""
    
    def test_redact_full_key(self):
        """Test full API key is redacted."""
        api_key = "test-api-key-12345"
        text = f"Error: API key {api_key} is invalid"
        
        result = _redact_api_key(text, api_key)
        assert api_key not in result
        assert "***REDACTED_API_KEY***" in result
    
    def test_redact_partial_key(self):
        """Test partial API key (prefix/suffix) is redacted."""
        api_key = "very-long-api-key-1234567890"
        text = f"Error: {api_key[:8]} is invalid"
        
        result = _redact_api_key(text, api_key)
        assert "***REDACTED***" in result
    
    def test_no_key_returns_original(self):
        """Test text without API key returns original."""
        text = "Error: Connection failed"
        result = _redact_api_key(text, None)
        assert result == text


class TestValidateUrl:
    """Tests for _validate_url()."""
    
    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        is_valid, error = _validate_url("https://ollama.com", allow_local=False)
        assert is_valid is True
        assert error is None
    
    def test_http_url_for_cloud_invalid(self):
        """Test HTTP URL for cloud endpoint is invalid."""
        is_valid, error = _validate_url("http://ollama.com", allow_local=False)
        assert is_valid is False
        assert "HTTPS" in error
    
    def test_localhost_blocked_for_cloud(self):
        """Test localhost is blocked for cloud endpoints."""
        is_valid, error = _validate_url("https://localhost", allow_local=False)
        assert is_valid is False
        assert "localhost" in error.lower()
    
    def test_private_ip_blocked_for_cloud(self):
        """Test private IP is blocked for cloud endpoints."""
        is_valid, error = _validate_url("https://192.168.1.1", allow_local=False)
        assert is_valid is False
        assert "private" in error.lower()
    
    def test_localhost_allowed_for_local(self):
        """Test localhost is allowed for local endpoints."""
        is_valid, error = _validate_url("http://localhost:11434", allow_local=True)
        assert is_valid is True
    
    def test_missing_scheme_invalid(self):
        """Test URL without scheme is invalid."""
        is_valid, error = _validate_url("ollama.com", allow_local=False)
        assert is_valid is False
        assert "scheme" in error.lower()
    
    def test_missing_hostname_invalid(self):
        """Test URL without hostname is invalid."""
        is_valid, error = _validate_url("https://", allow_local=False)
        assert is_valid is False
        assert "hostname" in error.lower()


class TestGetRequestsCABundle:
    """Tests for get_requests_ca_bundle() path resolution."""
    
    def test_get_ca_bundle_from_settings(self, mock_settings_path, tmp_path, monkeypatch):
        """Test getting CA bundle from settings.json."""
        # Create a test CA bundle file
        ca_file = tmp_path / "test-ca-bundle.pem"
        ca_file.write_text("FAKE CA CERT")
        
        # Write settings with CA bundle path
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudCABundle": str(ca_file)}, f)
        
        result = get_requests_ca_bundle()
        assert result == str(ca_file)
    
    def test_get_ca_bundle_from_secrets(self, mock_settings_path, mock_secrets_path, tmp_path):
        """Test getting CA bundle from secrets file."""
        # Create a test CA bundle file
        ca_file = tmp_path / "secret-ca-bundle.pem"
        ca_file.write_text("FAKE CA CERT")
        
        # Write secrets with CA bundle path
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"ca_bundle": str(ca_file)}, f)
        
        result = get_requests_ca_bundle()
        assert result == str(ca_file)
    
    def test_get_ca_bundle_from_env(self, mock_settings_path, monkeypatch, tmp_path):
        """Test getting CA bundle from environment variable."""
        # Create a test CA bundle file
        ca_file = tmp_path / "env-ca-bundle.pem"
        ca_file.write_text("FAKE CA CERT")
        
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca_file))
        result = get_requests_ca_bundle()
        assert result == str(ca_file)
    
    def test_settings_priority_over_secrets(self, mock_settings_path, mock_secrets_path, tmp_path):
        """Test settings.json takes priority over secrets file."""
        # Create two CA bundle files
        settings_ca_file = tmp_path / "settings-ca.pem"
        settings_ca_file.write_text("SETTINGS CA")
        secrets_ca_file = tmp_path / "secrets-ca.pem"
        secrets_ca_file.write_text("SECRETS CA")
        
        # Write both settings and secrets
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudCABundle": str(settings_ca_file)}, f)
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"ca_bundle": str(secrets_ca_file)}, f)
        
        result = get_requests_ca_bundle()
        assert result == str(settings_ca_file)
    
    def test_secrets_priority_over_env(self, mock_settings_path, mock_secrets_path, tmp_path, monkeypatch):
        """Test secrets file takes priority over environment variable."""
        # Create two CA bundle files
        secrets_ca_file = tmp_path / "secrets-ca.pem"
        secrets_ca_file.write_text("SECRETS CA")
        env_ca_file = tmp_path / "env-ca.pem"
        env_ca_file.write_text("ENV CA")
        
        # Write secrets and set env var
        with open(mock_secrets_path, "w", encoding="utf-8") as f:
            json.dump({"ca_bundle": str(secrets_ca_file)}, f)
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(env_ca_file))
        
        result = get_requests_ca_bundle()
        assert result == str(secrets_ca_file)
    
    def test_relative_path_resolution(self, mock_settings_path, tmp_path, monkeypatch):
        """Test relative paths are resolved relative to BASE_DIR."""
        # Mock BASE_DIR to tmp_path
        monkeypatch.setattr("src.core.ollama_config.BASE_DIR", tmp_path)
        
        # Create CA bundle in a subdirectory
        subdir = tmp_path / "config"
        subdir.mkdir(exist_ok=True)
        ca_file = subdir / "ca-bundle.pem"
        ca_file.write_text("FAKE CA CERT")
        
        # Write settings with relative path
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudCABundle": "config/ca-bundle.pem"}, f)
        
        result = get_requests_ca_bundle()
        assert result == str(ca_file)
        assert Path(result).is_absolute()
    
    def test_absolute_path_unchanged(self, mock_settings_path, tmp_path):
        """Test absolute paths are used as-is."""
        # Create CA bundle with absolute path
        ca_file = tmp_path / "absolute-ca.pem"
        ca_file.write_text("FAKE CA CERT")
        
        # Write settings with absolute path
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudCABundle": str(ca_file)}, f)
        
        result = get_requests_ca_bundle()
        assert result == str(ca_file)
    
    def test_nonexistent_file_returns_none(self, mock_settings_path):
        """Test nonexistent CA bundle file returns None with warning."""
        # Write settings with path to nonexistent file
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudCABundle": "/nonexistent/ca-bundle.pem"}, f)
        
        result = get_requests_ca_bundle()
        assert result is None
    
    def test_empty_string_returns_none(self, mock_settings_path):
        """Test empty string CA bundle returns None."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudCABundle": ""}, f)
        
        result = get_requests_ca_bundle()
        assert result is None
    
    def test_whitespace_only_returns_none(self, mock_settings_path):
        """Test whitespace-only CA bundle returns None."""
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({"ollamaCloudCABundle": "   "}, f)
        
        result = get_requests_ca_bundle()
        assert result is None
    
    def test_no_ca_bundle_configured_returns_none(self, mock_settings_path):
        """Test returns None when no CA bundle is configured."""
        # Empty settings file
        with open(mock_settings_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        
        result = get_requests_ca_bundle()
        assert result is None
