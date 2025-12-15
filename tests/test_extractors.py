import io
import types

import pytest

from src.core import extractors


def test_markdown_extraction():
    content = b"# Title\nHello\n"
    assert "Hello" in extractors.extract_text_from_bytes(content, "note.md")


def test_csv_extraction():
    content = b"a,b,c\n1,2,3\n"
    text = extractors.extract_text_from_bytes(content, "data.csv")
    assert "a,b,c" in text
    assert "1,2,3" in text


def test_html_sniff_without_extension(monkeypatch):
    html = b"<html><body><h1>Hello</h1><script>bad()</script></body></html>"
    soup = types.SimpleNamespace
    # Use real BeautifulSoup if available, otherwise skip
    if extractors.BeautifulSoup is None:
        pytest.skip("BeautifulSoup not installed")
    text = extractors.extract_text_from_bytes(html, "noext")
    assert "Hello" in text
    assert "bad" not in text


def test_html_extension_extraction():
    html = b"<html><body><p>Test</p></body></html>"
    if extractors.BeautifulSoup is None:
        pytest.skip("BeautifulSoup not installed")
    text = extractors.extract_text_from_bytes(html, "page.html")
    assert "Test" in text


def test_markdown_fallback_utf8():
    # ensure no crash on non-utf8
    bad = "caf\u00e9".encode("latin-1")
    text = extractors.extract_text_from_bytes(bad, "note.md")
    assert "caf" in text


def test_supported_filename_detection():
    assert extractors.is_supported_filename("file.pdf")
    assert extractors.is_supported_filename("file.html")
    assert extractors.is_supported_filename("file.pptx")
    assert extractors.is_supported_filename("noext")
    assert not extractors.is_supported_filename("file.exe")
