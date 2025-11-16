import pytest
import pathlib
from rag_core import _extract_text_from_file

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

def test_docx_extraction():
    # Skipped: requires a sample docx file and python-docx
    pass

def test_pdf_extraction():
    # Skipped: requires a sample pdf file and pypdf
    pass

def test_url_extraction(monkeypatch):
    class DummyResponse:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            pass
    def dummy_get(url, timeout=30):
        return DummyResponse(b"Web page text")
    monkeypatch.setattr("rag_core.requests.get", dummy_get)
    text = _extract_text_from_file("http://example.com/test.txt")
    assert "Web page text" in text

def test_ssl_error(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            raise Exception("SSL error")
    def dummy_get(url, timeout=30):
        raise Exception("SSL error")
    monkeypatch.setattr("rag_core.requests.get", dummy_get)
    text = _extract_text_from_file("https://example.com/test.txt")
    assert "SSL ERROR" in text or "ERROR" in text

def test_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    text = _extract_text_from_file(empty_file)
    assert text == ""

def test_bad_path():
    text = _extract_text_from_file(pathlib.Path("/nonexistent/file.txt"))
    assert text == ""
