import pytest
from src.core.rag_core import _extract_text_from_file, upsert_document, get_store
import pathlib

def test_empty_doc_upsert():
    upsert_document("empty.txt", "")
    assert get_store().docs["empty.txt"] == ""

def test_nonexistent_file_extraction():
    text = _extract_text_from_file(pathlib.Path("/nonexistent/file.txt"))
    assert text == ""

def test_bad_url_extraction(monkeypatch):
    def dummy_get(url, timeout=30):
        raise Exception("Connection error")
    import src.core.extractors as extractors
    monkeypatch.setattr(extractors.requests, "get", dummy_get)
    text = _extract_text_from_file("http://badurl.test")
    assert "CONNECTION ERROR" in text or "ERROR" in text
