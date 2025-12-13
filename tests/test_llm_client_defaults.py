import importlib
import json


def test_llm_client_prefers_settings_model_over_env(monkeypatch, tmp_path):
    # Arrange: create a temporary settings.json
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"model": "gemini-3-pro-preview"}), encoding="utf-8")

    # Patch SETTINGS_PATH to our temp settings
    from src.core import config_paths

    monkeypatch.setattr(config_paths, "SETTINGS_PATH", settings_path)

    # Set env model to a different value; settings should win
    monkeypatch.setenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")
    monkeypatch.delenv("ASYNC_LLM_MODEL_NAME", raising=False)

    # Reload module to ensure any import-time defaults do not leak into the assertion
    from src.core import llm_client

    importlib.reload(llm_client)

    # Act
    cfg = llm_client.get_llm_config()

    # Assert
    assert cfg["model"] == "gemini-3-pro-preview"
    assert cfg["async_model"] == "gemini-3-pro-preview"
