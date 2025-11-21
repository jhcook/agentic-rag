from pathlib import Path

import pytest

from src.core.rag_core import Store, _extract_text_from_file, reset_store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    """Reset the store so extraction tests do not pollute shared state."""
    db_file = tmp_path / "extraction-store.jsonl"
    reset_store(Store(), db_path=str(db_file))
    yield
    if db_file.exists():
        db_file.unlink()


def test_txt_extraction(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world! This is a test.")
    text = _extract_text_from_file(txt_file)
    assert "Hello world!" in text


def test_html_extraction(tmp_path):
    html_file = tmp_path / "test.html"
    html_file.write_text("<html><body><h1>Title</h1><p>Paragraph</p></body></html>")
    text = _extract_text_from_file(html_file)
    assert "Title" in text and "Paragraph" in text


@pytest.mark.skip(reason="DOCX extraction requires python-docx and sample files")
def test_docx_extraction():
    """Placeholder to signal DOCX extraction is not covered in unit tests."""


@pytest.mark.skip(reason="PDF extraction requires pypdf and sample files")
def test_pdf_extraction():
    """Placeholder to signal PDF extraction is not covered in unit tests."""


def test_url_extraction(monkeypatch):
    class DummyResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    def dummy_get(url, timeout=30):  # pylint: disable=unused-argument
        return DummyResponse(b"Web page text")

    monkeypatch.setattr(
        "src.core.extractors.requests.get",
        dummy_get,
        raising=False,
    )
    text = _extract_text_from_file("http://example.com/test.txt")
    assert "Web page text" in text


def test_ssl_error(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            raise RuntimeError("SSL error")

    def dummy_get(url, timeout=30):  # pylint: disable=unused-argument
        raise RuntimeError("SSL error")

    monkeypatch.setattr(
        "src.core.extractors.requests.get",
        dummy_get,
        raising=False,
    )
    text = _extract_text_from_file("https://example.com/test.txt")
    assert "SSL ERROR" in text or "ERROR" in text


def test_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    text = _extract_text_from_file(empty_file)
    assert text == ""


def test_bad_path():
    text = _extract_text_from_file(Path("/nonexistent/file.txt"))
    assert text == ""
