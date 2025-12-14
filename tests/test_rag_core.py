"""Tests for RAG core functionality."""
import logging
from pathlib import Path

import pytest

from src.core import document_repo, pgvector_store
from src.core.rag_core import (
    upsert_document,
    index_path,
    search,
    rerank,
    synthesize_answer,
    grounded_answer,
    verify_grounding_simple,
    verify_grounding,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def temp_indexed_dir(tmp_path, monkeypatch):
    indexed = tmp_path / "indexed"
    monkeypatch.setattr("src.core.document_repo.INDEXED_DIR", str(indexed))
    indexed.mkdir(parents=True, exist_ok=True)
    return indexed

def test_upsert_document(temp_indexed_dir, require_pgvector):
    """Upsert writes canonical artifact and updates pgvector."""

    test_uri = "test_doc.txt"
    test_content = "Test content for upsert"
    result = upsert_document(test_uri, test_content)
    assert result["upserted"] is True

    artifact_path = document_repo.artifact_path_for_uri(test_uri)
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == test_content

    # Verify pgvector sees the document
    docs = pgvector_store.list_documents()
    assert any(d.get("uri") == test_uri for d in docs)


def test_index_path(tmp_path, temp_indexed_dir, require_pgvector):
    """Test indexing a directory path."""
    # Create temporary directory with test files
    (tmp_path / "test1.txt").write_text("Content 1")
    (tmp_path / "test2.txt").write_text("Content 2")
    
    result = index_path(str(tmp_path))
    assert result["indexed"] == 2

def test_search(monkeypatch):
    """Test search functionality."""
    # Avoid external dependencies by mocking retrieval + LLM completion.
    def mock_build(*_args, **_kwargs):
        candidates = [{"uri": "doc1.txt", "text": "This is a passage about artificial intelligence.", "score": 1.0}]
        return ("context", ["doc1.txt"], candidates)

    monkeypatch.setattr("src.core.rag_core._build_rag_context", mock_build)
    monkeypatch.setattr(
        "src.core.rag_core.completion",
        lambda **_kwargs: {"choices": [{"message": {"content": "Answer"}}]},
    )

    results = search("artificial intelligence", top_k=5)
    assert isinstance(results, dict)
    assert "answer" in results or "choices" in results or "error" in results


def test_rerank():
    """Test reranking functionality."""
    passages = [
        {"text": "AI and machine learning", "score": 0.5},
        {"text": "Database systems", "score": 0.3}
    ]
    results = rerank("AI", passages)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_synthesize_answer():
    """Test answer synthesis."""
    passages = [
        {"text": "AI is a field of computer science", "uri": "doc1.txt"},
        {"text": "Machine learning is a subset of AI", "uri": "doc2.txt"}
    ]
    result = synthesize_answer("What is AI?", passages)
    assert "answer" in result
    assert "citations" in result


def test_grounded_answer(monkeypatch, temp_indexed_dir):
    """Test grounded answer generation."""
    # Provide an indexed artifact for grounding verification.
    document_repo.write_indexed_text(uri="ai_doc.txt", text="AI is artificial intelligence")

    monkeypatch.setattr(
        "src.core.rag_core._vector_search",
        lambda *_args, **_kwargs: [{
            "uri": "ai_doc.txt",
            "text": "AI is artificial intelligence and widely used in software.",
            "score": 1.0,
        }],
    )
    monkeypatch.setattr(
        "src.core.rag_core.completion",
        lambda **_kwargs: {"choices": [{"message": {"content": "AI is artificial intelligence.\n\nSources:\n[1] ai_doc.txt"}}]},
    )

    result = grounded_answer("What is AI?", k=1)
    assert "answer" in result
    assert "sources" in result


def test_verify_grounding_simple():
    """Test simple grounding verification."""
    passages = [
        {"text": "AI is artificial intelligence", "uri": "doc1.txt"},
        {"text": "Machine learning is part of AI", "uri": "doc2.txt"}
    ]
    result = verify_grounding_simple(
        "What is AI?",
        "AI is artificial intelligence [1]. It includes machine learning [2].",
        passages
    )
    assert "answer_conf" in result
    assert "citation_coverage" in result
    assert "missing_facts" in result


def test_verify_grounding():
    """Test grounding verification."""
    document_repo.write_indexed_text(uri="doc1.txt", text="AI is artificial intelligence")
    result = verify_grounding(
        "What is AI?",
        "AI is artificial intelligence [1].",
        ["doc1.txt"]
    )
    assert "answer_conf" in result
    assert "citation_coverage" in result


from unittest.mock import MagicMock

def test_grounded_answer_with_config(monkeypatch):
    """Test grounded answer with custom configuration."""
    import src.core.rag_core as rag_core
    
    # Mock completion
    mock_completion = MagicMock()
    mock_completion.return_value = {"choices": [{"message": {"content": "Test answer"}}]}
    monkeypatch.setattr("src.core.rag_core.completion", mock_completion)
    
    # Ensure artifact exists for any grounding calls.
    document_repo.write_indexed_text(uri="doc1.txt", text="Content")
    
    # Mock vector search to return hits so we don't hit "I don't know" early return
    # Text must be long enough to pass _is_low_signal filter (>= 12 words)
    long_text = "This is a sample document content that is long enough to pass the low signal filter check in the grounded answer function."
    def mock_search(*args, **kwargs):
        return [{"uri": "doc1.txt", "text": long_text, "score": 1.0}]
    monkeypatch.setattr("src.core.rag_core._vector_search", mock_search)
    
    # Call with config
    rag_core.grounded_answer(
        "Question", 
        model="custom-model", 
        temperature=0.7,
        num_ctx=2048
    )
    
    # Verify call args
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "custom-model"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["num_ctx"] == 2048
