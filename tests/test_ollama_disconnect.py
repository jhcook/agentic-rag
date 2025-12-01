import json
import sys
import tempfile
import types
from pathlib import Path

class _FakeStoreModel:
    """Minimal stand-in for rag_core.Store."""

    def __init__(self):
        self.docs = {}


_FAKE_STORE = _FakeStoreModel()


def _noop(*_, **__):
    return {}


def _get_store():
    return _FAKE_STORE


def _get_faiss_globals():
    return (types.SimpleNamespace(ntotal=0), None, None)


_fake_rag_core = types.SimpleNamespace(
    OLLAMA_API_BASE="http://127.0.0.1:11434",
    resolve_input_path=lambda path: Path(path),
    upsert_document=_noop,
    search=_noop,
    load_store=lambda: True,
    save_store=lambda: True,
    ensure_store_synced=lambda: True,
    get_store=_get_store,
    rebuild_faiss_index=lambda: None,
    rerank=_noop,
    verify_grounding=_noop,
    get_faiss_globals=_get_faiss_globals,
    Store=_FakeStoreModel,
    DB_PATH=str(Path(tempfile.gettempdir()) / "rag_store.jsonl"),
    should_skip_uri=lambda *_args, **_kwargs: False,
)

sys.modules.setdefault("src.core.rag_core", _fake_rag_core)

from src.core import factory


class _DummyOllamaBackend:
    """Lightweight stand-in so HybridBackend can initialize without heavy deps."""

    configured = True


def test_logout_ollama_clears_settings():
    """
    Ensure that calling logout(provider='ollama'):
    - clears provider-specific fields in settings
    - marks ollamaConfigured as False
    - keeps existing REST connection fields intact
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dir = Path(tmp_dir) / "config"
        config_dir.mkdir()
        settings_path = config_dir / "settings.json"

        initial_config = {
            "apiEndpoint": "http://localhost:11434",
            "model": "qwen2.5:7b",
            "embeddingModel": "Snowflake/arctic-embed-xs",
            "temperature": "0.7",
            "topP": "0.9",
            "topK": "40",
            "repeatPenalty": "1.1",
            "seed": "-1",
            "numCtx": "2048",
            "mcpHost": "127.0.0.1",
            "mcpPort": "8000",
            "mcpPath": "/mcp",
            "ragHost": "10.0.0.5",
            "ragPort": "8100",
            "ragPath": "api",
            "debugMode": False,
        }

        with settings_path.open("w", encoding="utf-8") as handle:
            json.dump(initial_config, handle)

        original_config_dir = factory.CONFIG_DIR
        original_settings_path = factory.SETTINGS_PATH
        original_has_ollama = factory.HAS_OLLAMA_CORE
        original_backend_cls = factory.OllamaBackend

        try:
            factory.CONFIG_DIR = config_dir
            factory.SETTINGS_PATH = settings_path
            factory.HAS_OLLAMA_CORE = True
            factory.OllamaBackend = lambda: _DummyOllamaBackend()

            backend = factory.HybridBackend(initial_mode="ollama")
            assert backend.ollama_configured is True

            backend.logout(provider="ollama")

            with settings_path.open("r", encoding="utf-8") as handle:
                updated = json.load(handle)

            assert updated["ollamaConfigured"] is False
            assert updated["ragMode"] == "none"
            assert updated["ragHost"] == "10.0.0.5"  # preserved
            assert backend.ollama_configured is False
        finally:
            factory.CONFIG_DIR = original_config_dir
            factory.SETTINGS_PATH = original_settings_path
            factory.HAS_OLLAMA_CORE = original_has_ollama
            factory.OllamaBackend = original_backend_cls


if __name__ == "__main__":
    test_logout_ollama_clears_settings()
    print("✓ Ollama disconnect test passed")
