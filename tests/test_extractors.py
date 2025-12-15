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


def test_sniff_rejects_exe_magic():
    # MZ header for PE/EXE should be rejected despite no extension
    exe_bytes = b"MZ" + b"\x00" * 10
    assert extractors._sniff_supported_content(exe_bytes, "file") is False


def test_sniff_allows_pdf_magic():
    pdf_bytes = b"%PDF-1.4 something"
    assert extractors._sniff_supported_content(pdf_bytes, "file") is True


def test_sniff_allows_zip_office():
    zip_bytes = b"PK\x03\x04" + b"\x00" * 10
    assert extractors._sniff_supported_content(zip_bytes, "file") is True


def test_sniff_allows_html_without_ext():
    html = b"<!doctype html><html><body>Hi</body></html>"
    assert extractors._sniff_supported_content(html, "noext") is True


def test_sniff_allows_utf8_text():
    txt = b"hello world " * 10
    assert extractors._sniff_supported_content(txt, "note") is True


def test_svg_supported_and_parsed():
    svg = b"<?xml version='1.0'?><svg><text>Hi</text></svg>"
    text = extractors.extract_text_from_bytes(svg, "image.svg")
    assert "Hi" in text
