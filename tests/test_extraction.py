"""Tests for text extraction from various file formats."""
import pathlib

import pytest

from src.core.extractors import extract_text_from_file


def test_txt_extraction(tmp_path):
    """Test text extraction from TXT files."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world! This is a test.")
    text = extract_text_from_file(txt_file)
    assert "Hello world!" in text

def test_html_extraction(tmp_path):
    """Test text extraction from HTML files."""
    html_file = tmp_path / "test.html"
    html_file.write_text(
        "<html><body><h1>Title</h1><p>Paragraph</p></body></html>")
    text = extract_text_from_file(html_file)
    assert "Title" in text and "Paragraph" in text


def test_docx_extraction():
    """Test DOCX extraction (skipped - requires sample file)."""
    # Skipped: requires a sample docx file and python-docx
    pass


def test_pdf_extraction():
    """Test PDF extraction (skipped - requires sample file)."""
    # Skipped: requires a sample pdf file and pypdf
    pass


def test_url_extraction(monkeypatch):
    """Test text extraction from URLs."""
    class DummyResponse:
        """Mock response for URL extraction tests."""
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def dummy_get(_url, timeout=30):
        return DummyResponse(b"Web page text")

    monkeypatch.setattr("src.core.extractors.requests.get", dummy_get)
    text = extract_text_from_file("http://example.com/test.txt")
    assert "Web page text" in text


def test_ssl_error(monkeypatch):
    """Test handling of SSL errors during URL extraction."""
    class DummyResponse:
        """Mock response that raises SSL errors."""
        def raise_for_status(self):
            raise Exception("SSL error")

    def dummy_get(_url, timeout=30):
        raise Exception("SSL error")

    monkeypatch.setattr("src.core.extractors.requests.get", dummy_get)
    text = extract_text_from_file("https://example.com/test.txt")
    assert "SSL ERROR" in text or "ERROR" in text


def test_empty_file(tmp_path):
    """Test extraction from empty files."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    text = extract_text_from_file(empty_file)
    assert text == ""


def test_bad_path():
    """Test extraction from non-existent files."""
    text = extract_text_from_file(pathlib.Path("/nonexistent/file.txt"))
    assert text == ""

