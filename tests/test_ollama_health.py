"""
Health check for the local Ollama HTTP service.

Purpose:
    Detect situations where the Ollama REST endpoint is unresponsive, which
    manifests in the application log as repeated LiteLLM timeouts.

Usage:
    By default this test pings /api/tags on OLLAMA_API_BASE and asserts it
    responds quickly. Set SKIP_OLLAMA_HEALTHCHECK=1 to skip in environments
    that do not run Ollama.
"""

from __future__ import annotations

import os
import time

import pytest
import requests


@pytest.mark.timeout(15)
def test_ollama_api_tags_is_reachable():
    """Ensure the Ollama REST API responds within a small window."""
    if os.getenv("SKIP_OLLAMA_HEALTHCHECK") == "1":
        pytest.skip("Ollama healthcheck skipped via SKIP_OLLAMA_HEALTHCHECK=1")

    base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434").rstrip("/")
    url = f"{base}/api/tags"
    soft_timeout = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "5"))

    start = time.monotonic()
    response = requests.get(url, timeout=soft_timeout)
    duration = time.monotonic() - start

    assert response.ok, f"Ollama /api/tags returned {response.status_code}"
    assert (
        duration <= soft_timeout
    ), f"Ollama /api/tags took {duration:.2f}s (> {soft_timeout}s); service may be hung"
