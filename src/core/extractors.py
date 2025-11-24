"""Text extraction from various file formats."""
import io
import logging
import pathlib

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


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """Extract text from file content (bytes) based on filename extension."""
    suffix = pathlib.Path(filename).suffix.lower()

    if suffix == '.pdf':
        if PdfReader is None:
            logger.warning("PDF support not available. Install pypdf: pip install pypdf")
            return ""
        try:
            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
            joined = "\n".join(text_parts).strip()
            if not joined:
                logger.warning("No text extracted from PDF %s", filename)
                return ""
            return joined
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", filename, exc)
            return ""

    elif suffix in ['.docx', '.doc']:
        if Document is None:
            logger.warning("DOCX support not available. Install python-docx: pip install python-docx")
            return ""
        try:
            doc = Document(io.BytesIO(content))
            text_parts = [paragraph.text for paragraph in doc.paragraphs]
            return "\n".join(text_parts)
        except Exception as exc:
            logger.error("Failed to read DOCX %s: %s", filename, exc)
            return ""

    elif suffix in ['.html', '.htm']:
        if BeautifulSoup is None:
            logger.warning(
                "HTML support not available. Install beautifulsoup4: "
                "pip install beautifulsoup4")
            return ""
        try:
            html_content = content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines
                      for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as exc:
            logger.error("Failed to read HTML %s: %s", filename, exc)
            return ""

    else:
        try:
            text = content.decode('utf-8', errors='ignore')
            lower_text = text[:8]
            if lower_text.startswith("%PDF-") or lower_text.startswith("PK"):
                return ""
            return text
        except Exception as exc:
            logger.error("Failed to read file %s: %s", filename, exc)
            return ""


def _extract_text_from_file(file_path: pathlib.Path) -> str:
    """Extract text from various file types (txt, pdf, docx, html) or URLs."""
    # Check if it's a URL and normalize single-slash URLs
    file_str = str(file_path)
    if file_path.name in {".DS_Store"}:
        return ""

    if file_str.startswith(('http://', 'https://')):
        content = _download_from_url(file_str)
        if not content:
            return ""
        return extract_text_from_bytes(content, file_str)

    # Local file
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        return extract_text_from_bytes(content, file_path.name)
    except Exception as exc:
        logger.error("Failed to read file %s: %s", file_path, exc)
        return ""
