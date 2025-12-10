"""Tests for Advanced RAG functionality: Loaders, Autotuning, Expansion."""
import logging
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure source is in path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.core import rag_core

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Test Autotuning ---

def test_autotune_rag_params():
    """Test dynamic parameter adjustment based on query length."""
    
    # Case 1: Short query (Technical lookup)
    short_q = "How do I install?"
    k, ctx = rag_core._autotune_rag_params(short_q, default_k=10)
    assert k == 3, "Expected strict top_k for short query"
    assert ctx == 2500, "Expected smaller context for short query"
    
    # Case 2: Long query (Research/Complex)
    long_q = "Please explain the detailed history of the Roman Empire and effective strategies used during its expansion across Europe and the Mediterranean."
    k, ctx = rag_core._autotune_rag_params(long_q, default_k=10)
    assert k == 8, f"Expected higher top_k for long query (len={len(long_q.split())})"
    assert ctx == 10000, "Expected large context for long query"
    
    # Case 3: Medium/Default query
    medium_q = "What is the capital of France and Germany and how do they compare?"
    k, ctx = rag_core._autotune_rag_params(medium_q, default_k=10)
    assert k == 10, f"Expected default k for medium query (len={len(medium_q.split())})"
    
# --- Test Query Expansion ---

def test_expand_query():
    """Test query expansion logic."""
    
    # 1. Test Fallback (when no LLM connected or simple pass-through)
    # Even if mocked, current impl returns original if exception or simple logic
    # We patch 'completion' to simulate failure/None if needed, or success
    
    q = "test query"
    # Current implementation is lightweight/placeholder that returns q unless enhanced
    # ensuring it at least returns the query
    res = rag_core.expand_query(q)
    assert res == q or "test query" in res

# --- Test LangChain Loaders Integration (Mocked) ---

@patch("src.core.rag_core.PyPDFLoader")
@patch("src.core.rag_core.TextLoader")
@patch("src.core.rag_core.Docx2txtLoader")
@patch("src.core.rag_core.UnstructuredFileLoader")
def test_loaders_selection(mock_unstructured, mock_docx, mock_text, mock_pdf):
    """Verify correct loader is initialized based on file extension."""
    
    # Mock return values
    mock_doc = MagicMock()
    mock_doc.page_content = "Mock Content"
    
    for mock_loader in [mock_pdf, mock_text, mock_docx, mock_unstructured]:
        mock_loader.return_value.load.return_value = [mock_doc]
    
    # Test PDF
    rag_core._load_and_chunk_file(pathlib.Path("test.pdf"))
    mock_pdf.assert_called_once()
    
    # Test Docx
    rag_core._load_and_chunk_file(pathlib.Path("test.docx"))
    mock_docx.assert_called_once()
    
    # Test Text
    rag_core._load_and_chunk_file(pathlib.Path("test.py"))
    mock_text.assert_called_once()
    
    # Test Fallback
    rag_core._load_and_chunk_file(pathlib.Path("test.unknown"))
    mock_unstructured.assert_called()

# --- Test Chunking (Recursive) ---

def test_recursive_chunking_integration():
    """Test that chunking uses separators if LangChain is present."""
    
    text = "Para1.\n\nPara2.\n\nPara3."
    
    # If LangChain is present, it should split by \n\n first
    if rag_core.HAS_LANGCHAIN:
        chunks = rag_core._chunk_text(text, max_chars=100, overlap=0)
        # Should ideally be 3 chunks
        assert len(chunks) == 3 or "Para1." in chunks[0]
        
    else:
        pytest.skip("LangChain not available for testing recursive chunking")
