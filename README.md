# Agentic RAG

🚧 Project Under Construction 🚧

## Overview

Agentic RAG (Retrieval-Augmented Generation) is a project designed to enhance the capabilities of natural language processing by integrating retrieval mechanisms with generative models. This project allows users to index documents, perform searches, and synthesize answers based on retrieved information.

## Project Structure

```
agentic-rag/
├── src/
│   ├── core/
│   │   └── rag_core.py          # Core RAG functionality and vector operations
│   ├── servers/
│   │   ├── mcp_server.py       # Model Context Protocol server
│   │   └── rest_server.py      # FastAPI REST API server
│   ├── clients/
│   │   └── cli_agent.py         # Command-line client for REST API
│   └── utils/
│       ├── debug_test.py        # Debugging utilities
│       ├── simple_indexer.py    # Basic file indexing (no embeddings)
│       └── metrics_dashboard.py # Terminal metrics dashboard + log tail
│       └── pylance_check.py     # Pylance validation tools
├── config/
│   └── mcp/
│       ├── mcp.yaml             # MCPHost configuration
│       └── planner_prompt.md    # AI assistant planner configuration
├── docs/                        # Project documentation and guides
├── documents/                   # Indexable source documents (PDF, DOCX, TXT, etc.)
├── tests/                       # Test suite
├── cache/                       # Vector store and cache files
├── log/                         # Log files
│   ├── mcp_server.log          # MCP server process logs
│   ├── mcp_server_access.log   # MCP server access logs
│   ├── rest_server.log         # REST API process logs
│   ├── rest_server_access.log  # REST API access logs
│   ├── ollama_server.log       # Ollama server logs
│   └── start.log                # Startup script output
└── scripts/                     # Startup and utility scripts
```

## Features

- **Document Indexing**: Index text documents for efficient retrieval using FAISS vector store
- **Search Functionality**: Perform semantic searches on indexed documents using natural language queries
- **Answer Synthesis**: Generate answers based on retrieved documents using LLM integration
- **Grounding Verification**: Verify the grounding of generated answers against the source documents
- **MCP Server**: Model Context Protocol server for integration with AI assistants
- **REST API**: RESTful API for document search and RAG operations
- **Debug Mode**: Skip expensive embedding operations for testing (`RAG_DEBUG_MODE=true`)
- **Automated Startup**: Shell scripts for easy service management

## Installation

To get started with the Agentic RAG project, clone the repository and install the required dependencies:

```bash
$ git clone https://github.com/yourusername/agentic-rag.git
...
$ cd agentic-rag
agentic-rag$ ./start.sh
...
```

### Prerequisites

- **Python 3.11+** (required)
- **Ollama** (for local LLM inference)
- **uv** (Python package installer, installed via requirements.txt)

Install Ollama:

```bash
$ brew install ollama
...
```

### Python Environment Setup

The project includes automated setup via `start.sh` which handles virtual environment creation and dependency installation. Alternatively, you can set up manually:

```bash
agentic-rag$ python3.11 -m venv .venv
agentic-rag$ source .venv/bin/activate
agentic-rag$ pip install -r requirements.txt
```

### Configuration

Copy the example environment file and customize as needed:

```bash
agentic-rag$ cp .env.example .env
```

Key configuration options in `.env`:

- `EMBED_MODEL_NAME`: Embedding model (default: Snowflake/arctic-embed-xs)
- `LLM_MODEL_NAME`: LLM model for completions (default: ollama/qwen2.5:0.5b)
- `OLLAMA_API_BASE`: Ollama server URL (default: 127.0.0.1:11434)
- `RAG_DB`: Path to document store (default: ./cache/rag_store.jsonl)
- `MCP_HOST`/`MCP_PORT`: MCP server binding (default: 127.0.0.1:8000)
- `REST_HOST`/`REST_PORT`: REST API binding (default: 127.0.0.1:8001)
- `MAX_MEMORY_MB`: Memory limit for HTTP server (default: 75% of system RAM)

## Quick Start

### Using the Startup Script (Recommended)

The easiest way to start all services:

```bash
agentic-rag$ ./start.sh
...
```

This will:

1. Create/activate Python virtual environment (`.venv`)
2. Install all dependencies from `requirements.txt`
3. Start Ollama server (if configured locally)
4. Start HTTP/MCP server on port 8000
5. Start REST API server on port 8001
6. Log all output to `log/start.log` with timestamps

All services run in the background with comprehensive error handling and automatic rollback on failure.

To stop all services:

```bash
agentic-rag$ ./stop.sh
...
```

#### Startup Script Options

```bash
./start.sh [OPTIONS]

Options:
  -h, --help              Show detailed help
  --env FILE              Use custom environment file (default: .env)
  --venv NAME             Use custom venv name (default: .venv)
  --python CMD            Python command to use (default: python3.11)
  --recreate-venv         Force recreation of virtual environment
```

Examples:

```bash
# Use production environment
agentic-rag$ ./start.sh --env .env.production

# Recreate virtual environment
agentic-rag$ ./start.sh --recreate-venv

# Use Python 3.12
agentic-rag$ ./start.sh --python python3.12
```

### Manual Startup

If you prefer to start services individually:

#### Terminal 1: Start Ollama

```bash
ollama serve
```

#### Terminal 2: Start HTTP/MCP Server

```bash
python -m src.servers.mcp_server
```

Expected output:

```text
2025-11-15 22:36:13,413 - rag_core - INFO - Rebuilding FAISS index from store documents
INFO:     Started server process [23579]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

#### Terminal 3: Start REST API Server

```bash
uvicorn src.servers.rest_server:app --host 127.0.0.1 --port 8001
```

Expected output:

```text
2025-11-15 20:17:00,154 - rest_server - INFO - REST server initialized with base path: /api
INFO:     Started server process [86841]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

## Usage

### Metrics & Monitoring

- **Prometheus endpoint**: The MCP server exposes metrics at `http://127.0.0.1:8000/metrics` (configurable via `MCP_HOST`/`MCP_PORT`). Metrics include indexed documents, embeddings, process memory, and Ollama model information.
- **Terminal dashboard + log tail**: Use the bundled curses dashboard to view key metrics (with a memory usage bar) in the top half of the terminal while tailing `log/mcp_server.log` in the bottom half:

```bash
python -m src.utils.metrics_dashboard \
  --url http://localhost:8000/metrics \
  --refresh 5 \
  --log-file log/mcp_server.log
```

The defaults already match the example above; override `--url`, `--refresh` (seconds), or `--log-file` as needed.

### Starting Services

#### MCP Server (Model Context Protocol)
```bash
python -m src.servers.mcp_server
```
- **Purpose**: AI assistant integration via MCP protocol
- **Endpoint**: `http://127.0.0.1:8000/mcp`
- **Features**: Document indexing, search, and RAG tools

#### REST API Server
```bash
uvicorn src.servers.rest_server:app --host 127.0.0.1 --port 8001
```
- **Purpose**: RESTful API for document operations
- **Endpoint**: `http://127.0.0.1:8001`
- **Features**: `/api/search`, `/api/index_path`, `/api/upsert_document`

#### CLI Client
```bash
python src/clients/cli_agent.py "your search query"
python src/clients/cli_agent.py "index documents"
```
- **Purpose**: Command-line interface to REST API
- **Features**: Search queries and document indexing

### REST API Examples

Search for documents:
```bash
curl -s http://127.0.0.1:8001/api/search \
  -H 'content-type: application/json' \
  -d '{"query":"your search query","k":5}' | jq
```

Index documents:
```bash
curl -s http://127.0.0.1:8001/api/index_path \
  -H 'content-type: application/json' \
  -d '{"path":"documents","glob":"**/*.txt"}' | jq
```

### Debug Mode

For testing without expensive embedding operations:
```bash
export RAG_DEBUG_MODE=true
python src/clients/cli_agent.py "index documents"  # Fast indexing without ML
```

### MCP Integration

The MCP server exposes document indexing and search tools that can be used by MCP-compatible AI assistants. See the [MCP documentation](docs/mcp/index.md) for complete setup and usage instructions.

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest -v --cov=rag_core tests/ -s
```

### Log Files

All services log to the `log/` directory:

- `log/start.log` - Startup script output (timestamped)
- `log/mcp_server.log` - MCP server process logs (application events, errors, debug info)
- `log/mcp_server_access.log` - MCP server access logs (HTTP requests in Apache format)
- `log/rest_server.log` - REST API process logs (application events, errors, debug info)
- `log/rest_server_access.log` - REST API access logs (HTTP requests in Apache format)
- `log/ollama_server.log` - Ollama server logs (when started locally)

Monitor logs in real-time:

```bash
tail -f log/*.log
```

Access logs follow Apache combined log format:
```
127.0.0.1 - - [17/Nov/2025:10:51:35 +1100] "GET /mcp HTTP/1.1" 200 1234 "-" "user-agent" 0.1234s
```

## Troubleshooting

### Services Won't Start

Check the startup log for detailed diagnostic information:

```bash
less log/start.log
```

The startup script provides:

- System information and environment variables
- Service status and port usage
- Recent log entries from failed services
- Automatic rollback and cleanup on failure

### Port Already in Use

Stop existing services:

```bash
./stop.sh
```

Or manually:

```bash
lsof -ti:8000 | xargs kill
lsof -ti:8001 | xargs kill
```

### Memory Issues

Adjust `MAX_MEMORY_MB` in `.env` file based on your system resources:

```bash
MAX_MEMORY_MB=32768  # 32GB limit
```

### Virtual Environment Issues

Force recreation of the virtual environment:

```bash
./start.sh --recreate-venv
```

## Test Coverage

The Agentic RAG project includes a comprehensive test suite covering:

- Document store logic, search, upsert, rerank, synthesis, grounding, and verification
- Extraction from TXT, HTML, URL, and edge cases (empty, bad path, SSL errors)
- FAISS index creation, vector shape, and reset logic
- FastAPI tool endpoint integration for document indexing, URL indexing, and search
- Edge cases for empty docs, nonexistent files, and bad URLs

Test files are located in the `tests/` directory:
- `test_rag_core.py`: Core logic
- `test_extraction.py`: Extraction and error handling
- `test_faiss.py`: FAISS index
- `test_http_server.py`: HTTP server integration
- `test_edge_cases.py`: Edge cases

To run all tests and view coverage:

```bash
pytest tests/
```

For more details, see `tests/README.md`.

## Resources

- [Ollama - llama3.2](https://ollama.com/library/llama3.2)
- [LiteLLM Documentation](https://docs.litellm.ai/docs/)
- [MCP Documentation](docs/mcp/index.md) - Complete MCP integration guide
- [MCP Python SDK](https://pypi.org/project/mcp/)
- [FastAPI](https://fastapi.tiangolo.com)
- [MCPHost](https://mcphub.tools/detail/mark3labs/mcphost)
