# RAG Test Suite

This directory contains tests for the Agentic RAG system, covering extraction, indexing, search, error handling, and integration.

## Test Files

- **test_rag_core.py**: Core logic for document store, search, upsert, rerank, synthesis, grounding, and verification.
- **test_extraction.py**: Extraction from TXT, HTML, URL, and edge cases (empty, bad path, SSL errors). PDF/DOCX tests are scaffolded for future samples.
- **test_faiss.py**: FAISS index creation, vector shape, and reset logic.
- **test_http_server.py**: FastAPI tool endpoint integration for document indexing, URL indexing, and search.
- **test_edge_cases.py**: Edge cases for empty docs, nonexistent files, and bad URLs.

## Running Tests

Activate your virtual environment and run:

```bash
pytest tests/
```

## Adding More Tests
- For PDF/DOCX extraction, add sample files and implement the skipped tests in `test_extraction.py`.
- For more HTTP server scenarios, expand `test_http_server.py` with additional endpoint and error cases.
- For advanced FAISS or embedding logic, add more assertions in `test_faiss.py`.

## Notes
- Monkeypatching is used for URL/SSL error simulation.
- All tests are designed to run in isolation and not require external services.
- If you add new features, please add corresponding tests here.
