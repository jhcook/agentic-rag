# Developer Getting Started Guide

Welcome to the **Agentic RAG** project! This guide is designed to get you up and running as a developer on the team. It covers environment setup, architecture, and common development workflows.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+**: The core language for the backend.
  - *Note*: We support Python 3.13 on macOS ARM64 for performance.
- **Node.js 18+ & npm**: Required for the React UI.
- **Docker Desktop**: Required for the PostgreSQL (pgvector) container.
- **Git**: For version control.

### Recommended Tools
- **uv**: Our fast Python package installer (installed automatically by `start.sh`, but good to have).
- **VS Code**: Recommended IDE with the Python and React extensions.

## 2. Project Architecture

The system follows a **Hybrid Monolith** architecture with distinct services that can run together or independently:

### Core Services
1.  **Ollama (AI Backend)**: Runs local LLMs (e.g., `qwen2.5:3b`). managed by `ollama serve`.
2.  **PostgreSQL + pgvector**: Stores the **Vector Index** for RAG search. Runs in Docker.
3.  **REST API Server** (`src/servers/rest_server.py`): FastAPI backend handling search, chat, and config.
4.  **MCP Server** (`src/servers/mcp_server.py`): Model Context Protocol server for AI agent integration.
5.  **UI Server** (Vite): React frontend for user interaction.

### Data Storage
- **Knowledge Base**: Stored in PostgreSQL (vectors) and `cache/indexed/` (text artifacts).
- **Chat History**: Stored locally in SQLite (`data/chat_store.db`) for lightweight persistence.

## 3. Quick Start

The easiest way to start developing is using our unified startup script.

```bash
# Clone the repo
git clone <repository_url>
cd agentic-rag

# Start everything (Monolith mode)
./start.sh
```

**What `start.sh` does:**
1.  Checks/installs `uv`.
2.  Creates a virtual environment (`.venv`).
3.  Syncs dependencies from `pyproject.toml`.
4.  Starts PostgreSQL via Docker Compose.
5.  Launches all services (Ollama, API, UI).

### Development Modes

You can run specific subsets of the application depending on what you are working on:

**Backend Only (No UI)**
```bash
./start.sh --role server
```

**Frontend Only (Connects to existing backend)**
```bash
./start.sh --role client
```

**Skip Local Ollama (Use Cloud/Remote)**
```bash
./start.sh --skip-ollama
```

## 4. Workflows

### Dependency Management
We use **uv** and `pyproject.toml`. **Do not use pip or requirements.txt directly.**

- **Add a package**: `uv add <package_name>`
- **Add a dev package**: `uv add --dev <package_name>`
- **Sync environment**: `uv sync`

### Running Tests
We use `pytest` for backend testing.

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_rag_core.py
```

### Database Management
- **Chat DB**: Delete `data/chat_store.db` to reset chat history.
- **Vector DB**: Use the "Flush Cache" button in the UI settings or API endpoint `/api/flush_cache` to clear the index.

### Linting & Formatting
Ensure code quality before committing:

```bash
# Lint backend code
uv run pylint src/
```

## 5. Directory Structure

- `src/core`: Core logic (RAG, embeddings, LLM client).
- `src/servers`: Service entrypoints (REST, MCP).
- `ui/`: React frontend application.
- `config/`: Application configuration.
- `docs/`: Documentation (API specs, guides).
- `tests/`: Pytest suite.

## 6. Troubleshooting

- **Port Conflicts**: Ensure ports 11434 (Ollama), 8000 (MCP), 8001 (API), and 5173 (UI) are free.
- **Missing Dependencies**: Run `uv sync` to ensure your environment is up to date.
- **Ollama Models**: If a model is missing, run `ollama pull <model_name>` or let `start.sh` handle it.

---
*Happy Coding!*
