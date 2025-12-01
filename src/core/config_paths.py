"""
Shared filesystem paths for configuration artifacts.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_PATH = CONFIG_DIR / "settings.json"
VERTEX_CONFIG_PATH = CONFIG_DIR / "vertex_config.json"
LEGACY_VERTEX_CONFIG_PATH = BASE_DIR / "vertex_config.json"

__all__ = [
    "BASE_DIR",
    "CONFIG_DIR",
    "SETTINGS_PATH",
    "VERTEX_CONFIG_PATH",
    "LEGACY_VERTEX_CONFIG_PATH",
]


