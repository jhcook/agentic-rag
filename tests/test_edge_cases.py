from pathlib import Path

import pytest

from src.core.rag_core import (
    Store,
    get_store,
    reset_store,
    upsert_document,
    _extract_text_from_file,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    """Ensure each test starts with a fresh in-memory store."""
    db_file = tmp_path / "edge-store.jsonl"
    reset_store(Store(), db_path=str(db_file))
    yield
    if db_file.exists():
        db_file.unlink()


def test_empty_doc_upsert():
    upsert_document("empty.txt", "")
    assert get_store().docs["empty.txt"] == ""


def test_nonexistent_file_extraction():
    text = _extract_text_from_file(Path("/nonexistent/file.txt"))
    assert text == ""


def test_bad_url_extraction(monkeypatch):
    class DummyError(Exception):
        """Raised by the dummy HTTP client."""

    def dummy_get(url, timeout=30):  # pylint: disable=unused-argument
        raise DummyError("Connection error")

    monkeypatch.setattr(
        "src.core.extractors.requests.get",
        dummy_get,
        raising=False,
    )
    text = _extract_text_from_file("http://badurl.test")
    assert "CONNECTION ERROR" in text or "ERROR" in text
