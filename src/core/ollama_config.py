"""Ollama configuration management for local and cloud modes."""
import json
import logging
import os
import stat
from pathlib import Path
from typing import Literal, Optional, Tuple
from urllib.parse import urlparse

from src.core.config_paths import SETTINGS_PATH, BASE_DIR

logger = logging.getLogger(__name__)

OllamaMode = Literal["local", "cloud", "auto"]

# Path to Ollama Cloud secrets file
OLLAMA_CLOUD_SECRETS_PATH = BASE_DIR / "secrets" / "ollama_cloud_config.json"

# Private IP ranges for SSRF protection
_PRIVATE_IP_RANGES = [
    ("127.0.0.0", "127.255.255.255"),  # localhost
    ("10.0.0.0", "10.255.255.255"),  # private
    ("172.16.0.0", "172.31.255.255"),  # private
    ("192.168.0.0", "192.168.255.255"),  # private
    ("169.254.0.0", "169.254.255.255"),  # link-local
    ("::1", "::1"),  # IPv6 localhost
    ("fc00::", "fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"),  # IPv6 private
]


def _redact_api_key(text: str, api_key: Optional[str]) -> str:
    """
    Redact API key from text to prevent exposure in logs or error messages.
    
    Args:
        text: Text that may contain API key
        api_key: API key to redact (if None, returns text as-is)
    
    Returns:
        Text with API key redacted
    """
    if not api_key or not text:
        return text
    
    # Redact full API key
    if api_key in text:
        text = text.replace(api_key, "***REDACTED_API_KEY***")
    
    # Also redact partial matches (first/last 8 chars)
    if len(api_key) > 16:
        prefix = api_key[:8]
        suffix = api_key[-8:]
        if prefix in text:
            text = text.replace(prefix, "***REDACTED***")
        if suffix in text:
            text = text.replace(suffix, "***REDACTED***")
    
    return text


def _validate_url(url: str, allow_local: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate URL format and check for SSRF vulnerabilities.
    
    Args:
        url: URL to validate
        allow_local: If True, allow localhost/private IPs (for local mode)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)
        
        # Must have scheme
        if not parsed.scheme:
            return False, "URL must include scheme (http:// or https://)"
        
        # Cloud endpoints must use HTTPS
        if not allow_local and parsed.scheme != "https":
            return False, "Cloud endpoints must use HTTPS"
        
        # Must have netloc (hostname)
        if not parsed.netloc:
            return False, "URL must include hostname"
        
        # Extract hostname (remove port if present)
        hostname = parsed.netloc.split(":")[0]
        
        # SSRF protection: block private IPs for cloud endpoints
        if not allow_local:
            # Check for localhost variants
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                return False, "Cloud endpoints cannot point to localhost"
            
            # Check for private IP ranges (basic check)
            # Note: This is a simplified check. For production, use ipaddress module
            if hostname.startswith(("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                                   "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                                   "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                                   "172.30.", "172.31.", "192.168.", "169.254.")):
                return False, "Cloud endpoints cannot point to private IP addresses"
        
        return True, None
        
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"Invalid URL format: {str(exc)}"


def _read_settings_file() -> dict:
    """Read settings from config/settings.json if it exists."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load settings.json: %s", e)
        return {}


def _read_ollama_cloud_secrets() -> dict:
    """Read Ollama Cloud configuration from secrets file."""
    if not OLLAMA_CLOUD_SECRETS_PATH.exists():
        return {}
    try:
        with open(OLLAMA_CLOUD_SECRETS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load ollama_cloud_config.json: %s", e)
        return {}


def get_ollama_mode() -> OllamaMode:
    """
    Get Ollama mode from settings.json or environment variable.
    
    Returns:
        "local", "cloud", or "auto". Defaults to "local" for backward compatibility.
    """
    settings = _read_settings_file()
    mode = settings.get("ollamaMode") or os.getenv("OLLAMA_MODE", "local")
    
    if mode not in ["local", "cloud", "auto"]:
        logger.warning("Invalid ollamaMode: %s, defaulting to 'local'", mode)
        return "local"
    
    return mode


def get_ollama_api_key() -> Optional[str]:
    """
    Get Ollama Cloud API key from secrets file or environment variable.
    
    Priority:
    1. secrets/ollama_cloud_config.json (api_key field)
    2. OLLAMA_CLOUD_API_KEY environment variable
    
    Returns:
        API key string if available, None otherwise.
    """
    # Try loading from secrets file first
    secrets = _read_ollama_cloud_secrets()
    api_key = secrets.get("api_key")
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.getenv("OLLAMA_CLOUD_API_KEY")
    
    # Return None if empty string
    if api_key and api_key.strip():
        return api_key.strip()
    
    return None


def get_ollama_cloud_proxy() -> Optional[str]:
    """
    Get proxy URL for Ollama Cloud connections.
    
    Priority:
    1. settings.json (ollamaCloudProxy)
    2. secrets/ollama_cloud_config.json (proxy)
    3. HTTPS_PROXY or HTTP_PROXY environment variables
    """
    settings = _read_settings_file()
    proxy = settings.get("ollamaCloudProxy")
    
    if not proxy:
        secrets = _read_ollama_cloud_secrets()
        proxy = secrets.get("proxy")
    
    if not proxy:
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    
    proxy = proxy.strip() if proxy else None
    return proxy or None


def save_ollama_cloud_proxy(proxy: Optional[str]) -> None:
    """
    Persist Ollama Cloud proxy setting to config/settings.json.
    
    Args:
        proxy: Proxy URL to save (None or empty string will clear it)
    """
    config: dict = _read_settings_file()
    if proxy:
        config["ollamaCloudProxy"] = proxy.strip()
    else:
        # Remove if empty/None
        config.pop("ollamaCloudProxy", None)
    
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_requests_ca_bundle() -> Optional[str]:
    """
    Get CA bundle path for verifying HTTPS connections.
    
    Priority:
    1. settings.json (ollamaCloudCABundle)
    2. secrets/ollama_cloud_config.json (ca_bundle)
    3. REQUESTS_CA_BUNDLE environment variable
    """
    settings = _read_settings_file()
    ca_bundle = settings.get("ollamaCloudCABundle")
    
    if not ca_bundle:
        secrets = _read_ollama_cloud_secrets()
        ca_bundle = secrets.get("ca_bundle")
    
    if not ca_bundle:
        ca_bundle = os.getenv("REQUESTS_CA_BUNDLE")
    
    ca_bundle = ca_bundle.strip() if ca_bundle else None
    return ca_bundle or None


def get_ollama_cloud_endpoint() -> str:
    """
    Get Ollama Cloud endpoint from settings, secrets file, or environment variable.
    
    Priority:
    1. settings.json (ollamaCloudEndpoint)
    2. secrets/ollama_cloud_config.json (endpoint field)
    3. OLLAMA_CLOUD_ENDPOINT environment variable
    4. Default: "https://ollama.com"
    
    Returns:
        Cloud endpoint URL. Defaults to "https://ollama.com".
        Always returns HTTPS endpoint (enforced).
    """
    settings = _read_settings_file()
    endpoint = settings.get("ollamaCloudEndpoint")
    
    # Try secrets file if not in settings
    if not endpoint:
        secrets = _read_ollama_cloud_secrets()
        endpoint = secrets.get("endpoint")
    
    # Fall back to environment variable or default
    if not endpoint:
        endpoint = os.getenv("OLLAMA_CLOUD_ENDPOINT", "https://ollama.com")
    
    # Validate URL format and enforce HTTPS
    is_valid, error_msg = _validate_url(endpoint, allow_local=False)
    if not is_valid:
        logger.warning(
            "Invalid cloud endpoint: %s. Error: %s. Using default.",
            _redact_api_key(endpoint, None),  # Endpoint itself shouldn't contain API key
            error_msg
        )
        return "https://ollama.com"
    
    # Ensure HTTPS (enforce even if validation passed)
    if not endpoint.startswith("https://"):
        logger.warning(
            "Cloud endpoint must use HTTPS. Converting %s to HTTPS.",
            _redact_api_key(endpoint, None)
        )
        # Replace http:// with https://
        endpoint = endpoint.replace("http://", "https://", 1)
    
    return endpoint


def get_ollama_local_endpoint() -> str:
    """
    Get local Ollama endpoint from settings or environment variable.
    
    Returns:
        Local endpoint URL. Defaults to "http://127.0.0.1:11434".
    """
    settings = _read_settings_file()
    endpoint = (
        settings.get("apiEndpoint")
        or os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
    )
    
    return endpoint


def get_ollama_endpoint() -> str:
    """
    Get appropriate Ollama endpoint based on current mode.
    
    Returns:
        Cloud endpoint if mode is "cloud", local endpoint if "local".
        For "auto" mode, returns cloud endpoint (fallback to local handled in API calls).
    """
    mode = get_ollama_mode()
    
    if mode == "cloud":
        return get_ollama_cloud_endpoint()
    elif mode == "auto":
        # For auto mode, prefer cloud but will fallback to local on failure
        return get_ollama_cloud_endpoint()
    
    # Local mode
    return get_ollama_local_endpoint()


def get_ollama_endpoint_with_fallback() -> Tuple[str, dict[str, str], str]:
    """
    Get Ollama endpoint with fallback support for auto mode.
    
    Returns:
        Tuple of (endpoint, headers, fallback_endpoint)
        - endpoint: Primary endpoint to try
        - headers: Headers for primary endpoint
        - fallback_endpoint: Fallback endpoint (local) if mode is "auto", None otherwise
    """
    mode = get_ollama_mode()
    
    if mode == "cloud":
        return (
            get_ollama_cloud_endpoint(),
            get_ollama_client_headers(),
            None  # No fallback for cloud mode
        )
    elif mode == "auto":
        # Try cloud first, fallback to local
        cloud_endpoint = get_ollama_cloud_endpoint()
        cloud_headers = get_ollama_client_headers()
        local_endpoint = get_ollama_local_endpoint()
        return (cloud_endpoint, cloud_headers, local_endpoint)
    else:
        # Local mode
        local_endpoint = get_ollama_local_endpoint()
        return (local_endpoint, {}, None)  # No headers for local, no fallback


def get_ollama_client_headers() -> dict[str, str]:
    """
    Get headers for Ollama client (includes auth for cloud).
    
    Returns:
        Dictionary with Authorization header if API key is available.
    """
    headers: dict[str, str] = {}
    api_key = get_ollama_api_key()
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    return headers


def save_ollama_cloud_config(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    ca_bundle: Optional[str] = None
) -> None:
    """
    Save Ollama Cloud configuration to secrets file.
    
    Args:
        api_key: API key to save (optional, only updates if provided)
        endpoint: Endpoint URL to save (optional, only updates if provided)
        ca_bundle: Path to CA bundle PEM file (optional)
    
    Raises:
        ValueError: If endpoint is invalid (not HTTPS or SSRF risk)
        IOError: If file cannot be written
    """
    # Ensure secrets directory exists
    secrets_dir = OLLAMA_CLOUD_SECRETS_PATH.parent
    secrets_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate endpoint if provided
    if endpoint is not None:
        # Convert HTTP to HTTPS before validation
        if endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = endpoint.replace("http://", "https://", 1)
        
        is_valid, error_msg = _validate_url(endpoint, allow_local=False)
        if not is_valid:
            raise ValueError(f"Invalid cloud endpoint: {error_msg}")
    
    # Load existing config if it exists
    existing_config = _read_ollama_cloud_secrets()
    
    # Update only provided fields
    if api_key is not None:
        existing_config["api_key"] = api_key.strip() if api_key else ""
    if endpoint is not None:
        existing_config["endpoint"] = endpoint.strip() if endpoint else ""
    if ca_bundle is not None:
        existing_config["ca_bundle"] = ca_bundle.strip() if ca_bundle else ""
    
    # Write to file
    try:
        with open(OLLAMA_CLOUD_SECRETS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_config, f, indent=2)
        
        # Set restrictive file permissions (rw-------)
        os.chmod(OLLAMA_CLOUD_SECRETS_PATH, stat.S_IRUSR | stat.S_IWUSR)
        
        logger.info("Saved Ollama Cloud configuration to %s", OLLAMA_CLOUD_SECRETS_PATH)
    except IOError as e:
        logger.error("Failed to save Ollama Cloud configuration: %s", e)
        raise


def test_cloud_connection(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    proxy: Optional[str] = None,
    ca_bundle: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Test connection to Ollama Cloud.
    
    Args:
        api_key: API key to test (optional, uses configured key if not provided)
        endpoint: Endpoint to test (optional, uses configured endpoint if not provided)
        proxy: HTTPS proxy to use for the test (optional)
        ca_bundle: Path to CA bundle PEM file for TLS verification (optional)
    
    Returns:
        Tuple of (success: bool, message: str)
        All error messages are sanitized to prevent API key exposure.
    """
    import requests
    
    # Use provided values or fall back to configured values
    test_api_key = api_key or get_ollama_api_key()
    test_endpoint = endpoint or get_ollama_cloud_endpoint()
    test_proxy = proxy or get_ollama_cloud_proxy()
    test_ca_bundle = ca_bundle or get_requests_ca_bundle()
    
    if not test_api_key:
        return False, "API key is required for cloud connection"
    
    # Validate endpoint before making request
    is_valid, error_msg = _validate_url(test_endpoint, allow_local=False)
    if not is_valid:
        return False, f"Invalid endpoint: {error_msg}"
    
    verify_arg: bool | str = True
    if test_ca_bundle:
        if not Path(test_ca_bundle).is_file():
            return False, f"CA bundle not found: {test_ca_bundle}"
        verify_arg = test_ca_bundle
    
    proxies = None
    if test_proxy:
        proxies = {"http": test_proxy, "https": test_proxy}
    
    try:
        headers = {"Authorization": f"Bearer {test_api_key}"}
        response = requests.get(
            f"{test_endpoint}/api/tags",
            headers=headers,
            timeout=5,
            verify=verify_arg,  # Enforce SSL certificate verification
            proxies=proxies,
        )
        
        if response.status_code == 200:
            return True, "Connection successful"
        elif response.status_code == 401:
            return False, "Invalid API key"
        elif response.status_code == 403:
            return False, "API key lacks required permissions"
        else:
            return False, f"Connection failed: HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error: Unable to reach endpoint"
    except requests.exceptions.SSLError as exc:
        # Redact any potential API key from SSL error
        error_msg = _redact_api_key(str(exc), test_api_key)
        return False, f"SSL error: {error_msg}"
    except requests.exceptions.RequestException as exc:
        # Redact API key from error message
        error_msg = _redact_api_key(str(exc), test_api_key)
        return False, f"Connection error: {error_msg}"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Redact API key from unexpected errors
        error_msg = _redact_api_key(str(exc), test_api_key)
        logger.error("Unexpected error testing cloud connection: %s", error_msg)
        return False, "Unexpected error occurred"


def normalize_ollama_model_name(model: str, mode: Optional[OllamaMode] = None) -> str:
    """
    Normalize Ollama model name based on mode.
    
    Examples:
    - "qwen2.5:0.5b" + mode="cloud" -> "qwen2.5:0.5b-cloud"
    - "qwen2.5:0.5b-cloud" + mode="local" -> "qwen2.5:0.5b"
    - "qwen2.5:0.5b-cloud" + mode="cloud" -> "qwen2.5:0.5b-cloud"
    - "qwen2.5:0.5b" + mode="auto" -> "qwen2.5:0.5b" (preserve as-is)
    
    Args:
        model: Model name (may or may not have -cloud suffix)
        mode: Ollama mode (optional, uses current mode if not provided)
    
    Returns:
        Normalized model name
    """
    if mode is None:
        mode = get_ollama_mode()
    
    # Remove -cloud suffix if present
    base_model = model.replace("-cloud", "")
    
    if mode == "cloud":
        # Add -cloud suffix if not already present
        if not model.endswith("-cloud"):
            return f"{base_model}-cloud"
        return model
    elif mode == "local":
        # Remove -cloud suffix if present
        return base_model
    else:  # auto mode
        # Preserve as-is
        return model


def validate_ollama_config() -> Tuple[bool, Optional[str]]:
    """
    Validate Ollama configuration.
    
    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is None.
    """
    mode = get_ollama_mode()
    
    if mode == "cloud":
        api_key = get_ollama_api_key()
        if not api_key:
            return False, "API key is required for cloud mode"
        
        endpoint = get_ollama_cloud_endpoint()
        
        # Validate URL format and security
        is_valid, error_msg = _validate_url(endpoint, allow_local=False)
        if not is_valid:
            return False, f"Invalid cloud endpoint: {error_msg}"
        
        # Double-check HTTPS (should already be enforced by get_ollama_cloud_endpoint)
        if not endpoint.startswith("https://"):
            return False, "Cloud endpoint must use HTTPS"
    
    return True, None
