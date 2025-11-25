#!/usr/bin/env python3
"""Minimal test to identify what's causing the indexing hang."""

import os
import sys
import time
from pathlib import Path
from typing import Sequence

import sentence_transformers
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Snowflake/snowflake-arctic-embed-xs"
TEST_TEXT = "This is a test sentence."


def _fail(stage: str, exc: Exception) -> None:
    print(f"{stage} failed: {exc}")
    sys.exit(1)


def _test_imports() -> None:
    """Verify core imports are available."""
    try:
        print(f"✓ sentence_transformers imported (v{sentence_transformers.__version__})")
    except ImportError as exc:
        _fail("Import", exc)


def _load_model() -> SentenceTransformer:
    """Load the embedding model in CPU/offline mode."""
    try:
        print(f"Loading model: {MODEL_NAME}")
        os.environ['HF_HUB_OFFLINE'] = '1'
        embedder = SentenceTransformer(
            MODEL_NAME,
            device='cpu',
            model_kwargs={"low_cpu_mem_usage": False},
        )
        dimension = embedder.get_sentence_embedding_dimension()
        print(f"✓ Model loaded successfully. Dimension: {dimension}")
        return embedder
    except (OSError, RuntimeError, ValueError) as exc:
        _fail("Model loading", exc)
    raise RuntimeError("Unreachable")  # Safety for type checkers


def _test_embedding(embedder: SentenceTransformer, texts: Sequence[str]) -> None:
    """Embed sample text and report timing."""
    try:
        start_time = time.time()
        embedding = embedder.encode(list(texts))
        duration = time.time() - start_time
        shape = getattr(embedding, "shape", "unknown")
        print(f"✓ Embedding generated in {duration:.2f} seconds (shape={shape})")
    except (RuntimeError, ValueError) as exc:
        _fail("Embedding", exc)


def _test_files(embedder: SentenceTransformer) -> None:
    """Read local documents and embed the first non-empty example."""
    try:
        docs_dir = Path("documents")
        if not docs_dir.exists():
            print("documents directory not found")
            sys.exit(1)

        txt_files = list(docs_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} txt files")

        for txt_file in txt_files:
            content = txt_file.read_text(encoding='utf-8')
            print(f"✓ Read {txt_file.name}: {len(content)} characters")

            if content.strip():
                _test_embedding(embedder, [content[:1000]])  # First 1000 chars only
                break
    except (OSError, ValueError) as exc:
        _fail("File operations", exc)


def main() -> None:
    """Run the suite of debug checks."""
    print(f"Starting debug test at {time.time()}")
    _test_imports()
    embedder = _load_model()
    _test_embedding(embedder, [TEST_TEXT])
    _test_files(embedder)
    print(f"All tests passed! Completed at {time.time()}")


if __name__ == "__main__":
    main()
