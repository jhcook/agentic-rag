# MCP Server Implementation

This document provides technical details about the MCP server implementation in the Agentic RAG system.

## Architecture Overview

The MCP server is built using the [MCP Python SDK](https://pypi.org/project/mcp/) and FastAPI. It exposes RAG capabilities as MCP tools that AI assistants can invoke.

## Server Structure

```
src/servers/mcp_server.py
├── MCP Tool Definitions
│   ├── index_documents_tool()
│   ├── index_url_tool()
│   ├── search_tool()
│   ├── rerank_tool()
│   ├── grounded_answer_tool()
│   ├── verify_grounding_tool()
│   └── list_indexed_documents_tool()
├── Core Integration
│   ├── RAG Core (src/core/rag_core.py)
│   ├── Indexed Artifacts (cache/indexed)
│   └── Embedding System
└── Server Configuration
    ├── Streamable HTTP Transport
    ├── Memory Management
    └── Error Handling
```

## Tool Implementation Details

### Document Indexing Tools

#### `index_documents_tool`

**Purpose**: Recursively indexes documents from a directory path.

**Implementation**:
- Resolves the input path to an absolute path
- Uses glob patterns to find matching files
- Extracts text from supported formats (PDF, DOCX, TXT, HTML)
- Stores documents in the vector database
- Returns indexing statistics

**Supported Formats**:
- PDF: Uses PyPDF for text extraction
- DOC/DOCX: Uses python-docx for content extraction
- TXT/Markdown: Direct UTF-8 decode (markdown left as-is)
- HTML: Uses BeautifulSoup to strip markup/script/style and return readable text
- CSV: UTF-8 decode of rows
- XLS/XLSX: openpyxl (first few sheets) to extract cell text
- PPT/PPTX: python-pptx to extract slide text
- RTF: striprtf to convert to plain text
- EPUB: ebooklib (+ BeautifulSoup if available) to extract document text
- Images (png/jpg/jpeg/tiff/bmp): OCR via Pillow + pytesseract if both the Python packages **and** the system `tesseract-ocr` binary are installed; otherwise these are rejected. Install the binary via your package manager (e.g., Homebrew: `brew install tesseract`, winget: `winget install tesseract`, apt: `sudo apt-get install tesseract-ocr`).
- Unsupported/binary files are rejected via extension allowlist plus content sniffing (magic headers/HTML/text heuristics); rejected paths are returned to the caller.

**Error Handling**:
- File not found errors
- Unsupported format warnings
- Text extraction failures (skips file, logs warning)

#### `index_url_tool`

**Purpose**: Indexes documents from HTTP/HTTPS URLs.

**Implementation**:
- Accepts URLs directly or extracts from query strings
- Downloads content using requests library
- Routes local paths to `index_documents_tool`
- Extracts text based on content-type
- Stores with URL as document ID

**URL Processing**:
- PDF URLs: Downloads and extracts text
- HTML pages: Parses and extracts readable content
- Plain text: Direct storage

### Search and Retrieval Tools

#### `search_tool`

**Purpose**: End-to-end search and answer generation.

**Implementation**:
- Performs vector similarity search
- Retrieves top-k relevant passages
- Uses configured LLM for answer synthesis
- Returns complete answer with model metadata

**Search Process**:
1. Embed query using sentence transformer
2. Search pgvector index for similar vectors
3. Retrieve full text passages
4. Generate answer using LiteLLM
5. Return formatted response

#### `rerank_tool`

**Purpose**: Improves search result ranking using heuristics.

**Implementation**:
- Applies lightweight reranking algorithm
- Considers query-passage relevance
- Returns reordered passage list

#### `grounded_answer_tool`

**Purpose**: Generates answers with citation support.

**Implementation**:
- Similar to `search_tool` but focuses on citation quality
- Ensures all claims are supported by sources
- Returns structured answer with references

#### `verify_grounding_tool`

**Purpose**: Validates answer quality and grounding.

**Implementation**:
- Compares answer claims against source documents
- Calculates confidence and citation coverage scores
- Identifies unsupported claims
- Returns quality metrics

### Management Tools

#### `list_indexed_documents_tool`

**Purpose**: Provides inventory of indexed documents.

**Implementation**:
- Accesses document store metadata
- Returns list of document URIs
- No content retrieval (metadata only)

### Server Configuration

### Initialization

The MCP server starts immediately. There is no JSONL/in-memory document store.

- Canonical extracted text is persisted as artifacts under `cache/indexed/`.
- Vector search lives in PostgreSQL + pgvector.
- Index rebuilds are triggered explicitly via the rebuild operation (or by re-indexing).

### Transport Layer

The server uses the "streamable-http" transport:

```python
mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp")
mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))
```

### Memory Management

- Implements memory limits to prevent OOM issues
- Monitors memory usage during operations
- Forces garbage collection after search operations
- Configurable via `MAX_MEMORY_MB` environment variable

### Error Handling

- Comprehensive logging with different levels
- Graceful degradation for non-critical errors
- Structured error responses for all tools
- Automatic cleanup on shutdown

## Integration Points

### RAG Core Integration

The MCP server integrates with the core RAG system:

- **Indexed artifacts**: Canonical extracted text under `cache/indexed/`
- **Vector index**: PostgreSQL + pgvector
- **Search Functions**: Calls `search()`, `rerank()`, `grounded_answer()`
- **Indexing**: Uses `upsert_document()` for storage
- **Embeddings**: Leverages configured embedding model

### Configuration System

- Environment variables for server settings
- No store lifecycle; configuration is loaded from settings/env
- Model configuration through LiteLLM
- Path resolution for document directories

## Performance Considerations

### Memory Usage
- Large document collections require significant RAM
- Embedding operations are memory-intensive
- Search operations trigger garbage collection
- Memory limits prevent system instability

### Search Optimization
- pgvector provides fast vector similarity search
- Configurable top-k limits result sizes
- Embedding caching reduces redundant computations
- Batch processing for multiple documents

### Concurrent Access
- PostgreSQL provides concurrency control for vector/index metadata.
- Artifact writes are atomic (write-then-rename) to keep text durable and consistent.
- Connection pooling for external services

## Development and Testing

### Local Development
```bash
# Run with debug logging
export MCP_LOG_LEVEL=DEBUG
python src/servers/mcp_server.py
```

### Testing Tools
- Unit tests in `tests/test_http_server.py`
- Integration tests for tool functionality
- Memory usage monitoring
- Performance benchmarking

### Debugging
- Comprehensive logging to `log/mcp_server.log`
- Request/response tracing
- Memory usage monitoring
- Error stack traces with context

## Deployment Considerations

### Production Setup
- Use production WSGI server (gunicorn/uvicorn)
- Configure proper logging and monitoring
- Set appropriate memory limits
- Secure server endpoints if exposed externally

### Scaling
- Horizontal scaling with multiple MCP servers
- Shared document store backend
- Load balancing for high-traffic scenarios
- Caching layers for frequently accessed documents

## API Compatibility

The server maintains compatibility with:
- MCP protocol specification
- FastAPI framework conventions
- Existing RAG core interfaces
- LiteLLM provider abstractions

## Future Enhancements

Potential improvements:
- Streaming responses for long operations
- Batch indexing operations
- Advanced reranking algorithms
- Multi-modal document support
- Real-time indexing updates
