import json
import logging
from pathlib import Path

import pytest

from src.core.rag_core import (
    Store,
    get_store,
    reset_store,
    save_store,
    load_store,
    rerank,
    synthesize_answer,
)

logging.basicConfig(level=logging.INFO)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    """Use a temporary DB path and reset the in-memory store before each test."""
    db_file = tmp_path / "store.jsonl"
    reset_store(Store(), db_path=str(db_file))
    yield
    if db_file.exists():
        db_file.unlink()


def test_store_add_and_persist():
    store = get_store()
    store.add("doc1.txt", "This is a sample document.")
    store.add("doc2.txt", "Another document.")
    save_store()

    # Reset and reload
    reset_store(Store())
    load_store()
    store = get_store()
    assert "doc1.txt" in store.docs
    assert "doc2.txt" in store.docs


def test_save_store_creates_jsonl(tmp_path):
    store = get_store()
    store.add("hello.txt", "hello world")
    save_store()
    db_path = Path(tmp_path) / "store.jsonl"
    assert db_path.exists()
    data = db_path.read_text().strip().splitlines()
    assert len(data) == 1
    record = json.loads(data[0])
    assert record["uri"] == "hello.txt"


def test_rerank_orders_more_relevant_passages():
    passages = [
        {"text": "AI and machine learning", "score": 0.4},
        {"text": "Completely unrelated topic", "score": 0.9},
    ]
    ranked = rerank("machine learning", passages)
    assert ranked[0]["text"] == "AI and machine learning"


def test_synthesize_answer_collects_citations():
    passages = [
        {"text": "AI is intelligence", "uri": "doc1.txt"},
        {"text": "Machine learning is part of AI", "uri": "doc2.txt"},
    ]
    result = synthesize_answer("What is AI?", passages)
    assert "AI is intelligence" in result["answer"]
    assert result["citations"] == ["doc1.txt", "doc2.txt"]
