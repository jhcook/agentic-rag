"""
Shared filesystem paths for configuration artifacts.
"""
import os
import json
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_PATH = CONFIG_DIR / "settings.json"
VERTEX_CONFIG_PATH = CONFIG_DIR / "vertex_config.json"

def get_ca_bundle_path() -> Optional[str]:
    """
    Get CA bundle path from environment or settings.
    Returns None if not configured (uses system default).
    """
    # 1. Check environment variables
    ca_path = os.getenv("CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if ca_path:
        return ca_path

    # 2. Check settings.json
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
                return settings.get("caBundlePath")
        except Exception:
            pass
            
    return None

__all__ = [
    "BASE_DIR",
    "CONFIG_DIR",
    "SETTINGS_PATH",
    "VERTEX_CONFIG_PATH",
    "get_ca_bundle_path",
]



