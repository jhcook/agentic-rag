import os
import json
from typing import Union
from pathlib import Path

def _get_ca_bundle_from_settings() -> Union[str, None]:
    """Read CA bundle path from settings.json."""
    try:
        # Resolve path relative to this file: src/core/ssl_utils.py -> config/settings.json
        base_dir = Path(__file__).resolve().parent.parent.parent
        settings_path = base_dir / "config" / "settings.json"
        
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                return settings.get("ollamaCloudCABundle")
    except Exception:
        pass
    return None

def configure_ssl_environment() -> None:
    """
    Configure SSL environment variables based on CA_BUNDLE or settings.
    This ensures that libraries like httpx, requests, and grpc use the correct CA bundle.
    """
    # Check settings first (highest priority)
    ca_bundle = _get_ca_bundle_from_settings()
    
    # Fallback to environment variable
    if not ca_bundle:
        ca_bundle = os.environ.get("CA_BUNDLE")
        
    if ca_bundle:
        # Propagate to standard environment variables
        # We overwrite here because settings/env var should take precedence
        os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
        os.environ["SSL_CERT_FILE"] = ca_bundle
        os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ca_bundle
        # Also set for huggingface_hub
        os.environ["CURL_CA_BUNDLE"] = ca_bundle

def get_ssl_verify() -> Union[bool, str]:
    """
    Get the SSL verification setting for requests.
    
    Returns:
        Union[bool, str]: The path to the CA bundle if configured,
                         otherwise True (default SSL verification).
    """
    # Ensure environment is configured
    configure_ssl_environment()
    
    # Check settings first
    ca_bundle = _get_ca_bundle_from_settings()
    
    # Fallback to environment variable
    if not ca_bundle:
        ca_bundle = os.environ.get("CA_BUNDLE")
        
    if ca_bundle:
        return ca_bundle
    
    # Default to True (verify with standard CAs or REQUESTS_CA_BUNDLE if set)
    return True
