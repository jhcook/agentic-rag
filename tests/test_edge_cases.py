import pytest
from src.core import document_repo
from src.core.rag_core import _extract_text_from_file, upsert_document
import pathlib


@pytest.fixture
def temp_indexed_dir(tmp_path, monkeypatch):
    indexed = tmp_path / "indexed"
    monkeypatch.setattr("src.core.document_repo.INDEXED_DIR", str(indexed))
    indexed.mkdir(parents=True, exist_ok=True)
    return indexed

def test_empty_doc_upsert(require_pgvector, temp_indexed_dir):
    upsert_document("empty.txt", "")
    artifact_path = document_repo.artifact_path_for_uri("empty.txt")
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == ""

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
