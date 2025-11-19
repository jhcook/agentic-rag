import io
import logging
import pathlib
from typing import Optional

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    PdfReader = None

try:
    from docx import Document  # type: ignore
except ImportError:
    Document = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None

try:
    import requests  # type: ignore
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


def _download_from_url(url: str) -> bytes:
    """Download content from a URL."""
    if requests is None:
        logger.warning("URL support not available. Install requests: pip install requests")
        return b""
    try:
        logger.info("Downloading from %s", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.SSLError as ssl_exc:
        logger.error("SSL error downloading %s: %s", url, ssl_exc)
        return b"[SSL ERROR: Could not connect to %s]" % url.encode()
    except requests.exceptions.ConnectionError as conn_exc:
        logger.error("Connection error downloading %s: %s", url, conn_exc)
        return b"[CONNECTION ERROR: Could not connect to %s]" % url.encode()
    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return b"[ERROR: Could not download %s: %s]" % (url.encode(), str(exc).encode())


def _extract_text_from_file(file_path: pathlib.Path) -> str:
    """Extract text from various file types (txt, pdf, docx, html) or URLs."""
    # Check if it's a URL and normalize single-slash URLs
    file_str = str(file_path)
    if file_path.name in {".DS_Store"}:
        return ""

    def _read_head(path: pathlib.Path, size: int = 512) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read(size)
        except Exception:
            return b""

    def _is_probably_text_bytes(buf: bytes) -> bool:
        if not buf:
            return False
        printable = sum((chr(b).isprintable() or chr(b).isspace()) for b in buf)
        density = printable / max(1, len(buf))
        return density >= 0.8

    head = None
    if not file_str.startswith(("http://", "https://")):
        head = _read_head(file_path, 512)

    def _normalize_suffix_from_head(default_suffix: str, head_bytes: Optional[bytes]) -> str:
        if head_bytes is None:
            return default_suffix
        if head_bytes.startswith(b"%PDF-"):
            return ".pdf"
        if head_bytes.startswith(b"PK"):
            # could be zip-based formats: docx, pages, etc.
            if default_suffix.lower() in (".docx", ".pages"):
                return default_suffix.lower()
            return ".zip"
        return default_suffix

    if file_str.startswith(('http://', 'https://')):
        content = _download_from_url(file_str)
        if not content:
            return ""
        # Determine file type from URL or content
        if file_str.endswith('.pdf'):
            suffix = '.pdf'
        elif file_str.endswith(('.docx', '.doc')):
            suffix = '.docx'
        elif file_str.endswith(('.html', '.htm')):
            suffix = '.html'
        else:
            # Try to detect HTML
            try:
                decoded = content.decode('utf-8', errors='ignore')
                if decoded.strip().startswith(('<html', '<!DOCTYPE', '<!doctype')):
                    suffix = '.html'
                else:
                    suffix = '.txt'
            except Exception:
                suffix = '.txt'
    else:
        content = None
        suffix = _normalize_suffix_from_head(file_path.suffix.lower(), head)

    if suffix == '.pdf':
        if PdfReader is None:
            logger.warning("PDF support not available. Install pypdf: pip install pypdf")
            return ""
        try:
            if content:
                # PDF from URL
                reader = PdfReader(io.BytesIO(content))
            else:
                # PDF from file
                reader = PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
            joined = "\n".join(text_parts).strip()
            if not joined:
                logger.warning("No text extracted from PDF %s", file_path)
                return ""
            return joined
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", file_path, exc)
            return ""

    elif suffix in ['.docx', '.doc']:
        if Document is None:
            logger.warning("DOCX support not available. Install python-docx: pip install python-docx")
            return ""
        try:
            if content:
                doc = Document(io.BytesIO(content))
            else:
                doc = Document(str(file_path))
            text_parts = [paragraph.text for paragraph in doc.paragraphs]
            return "\n".join(text_parts)
        except Exception as exc:
            logger.error("Failed to read DOCX %s: %s", file_path, exc)
            return ""

    elif suffix in ['.html', '.htm']:
        if BeautifulSoup is None:
            logger.warning("HTML support not available. Install beautifulsoup4: pip install beautifulsoup4")
            return ""
        try:
            if content:
                html_content = content.decode('utf-8', errors='ignore')
            else:
                html_content = file_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as exc:
            logger.error("Failed to read HTML %s: %s", file_path, exc)
            return ""

    else:
        try:
            if content:
                text = content.decode('utf-8', errors='ignore')
            else:
                if head is not None and not _is_probably_text_bytes(head):
                    return ""
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            lower_text = text[:8]
            if lower_text.startswith("%PDF-") or lower_text.startswith("PK"):
                return ""
            return text
        except Exception as exc:
            logger.error("Failed to read file %s: %s", file_path, exc)
            return ""
