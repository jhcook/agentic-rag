# Agentic RAG

🚧 Project Under Construction 🚧

## Overview

Agentic RAG (Retrieval-Augmented Generation) is a project designed to enhance the capabilities of natural language processing by integrating retrieval mechanisms with generative models. This project allows users to index documents, perform searches, and synthesize answers based on retrieved information.

## Project Structure

```
agentic-rag/
├── src/
│   ├── core/
│   │   ├── factory.py           # Factory pattern for RAG backends (Local/Remote)
│   │   ├── interfaces.py        # RAGBackend protocol definition
│   │   ├── rag_core.py          # Core RAG functionality and vector operations
│   │   ├── embeddings.py        # Embedding model wrapper
│   │   ├── extractors.py        # File content extraction
│   │   ├── faiss_index.py       # FAISS vector store management
│   │   ├── llm_client.py        # LLM client wrapper
│   │   └── store.py             # Document store implementation
│   ├── servers/
│   │   ├── mcp_server.py        # Model Context Protocol server
│   │   ├── rest_server.py       # FastAPI REST API server
│   │   └── mcp_app/             # MCP server application logic
│   ├── clients/
│   │   └── cli_agent.py         # Command-line client for REST API
│   └── utils/
│       ├── debug_test.py        # Debugging utilities
│       ├── simple_indexer.py    # Basic file indexing (no embeddings)
│       └── metrics_dashboard.py # Terminal metrics dashboard + log tail
├── config/
│   └── mcp/
│       ├── mcp.yaml             # MCPHost configuration
│       └── planner_prompt.md    # AI assistant planner configuration
├── docs/                        # Project documentation and guides
├── documents/                   # Indexable source documents (PDF, DOCX, TXT, etc.)
├── tests/                       # Test suite
├── cache/                       # Vector store and cache files
├── log/                         # Log files
└── scripts/                     # Startup, utility, and build (build_electron.py) scripts
```

## Features

- **Multiple AI Backends**: Choose between Local (Ollama), OpenAI Assistants, or Google Vertex AI
  - **Local (Ollama)**: Free, private, offline-capable with local LLMs
  - **OpenAI Assistants**: GPT-4 quality with function calling bridge to local documents
  - **Google Vertex AI**: Cloud integration with Drive/Gmail and 2M token context
- **Hybrid Architecture**: Support for both Monolithic (Local) and Distributed (Remote) deployments
- **Role-Based Startup**: flexible `start.sh` script to launch specific components (`monolith`, `server`, `client`)
- **Document Indexing**: Index text documents for efficient retrieval using FAISS vector store
- **URL Indexing**: Index content directly from remote URLs via the MCP worker
- **Search Functionality**: Perform semantic searches on indexed documents using natural language queries
- **Answer Synthesis**: Generate answers based on retrieved documents using LLM integration
- **Automatic Citations**: Inline citations [1], [2] with source references
- **Grounding Verification**: Verify the grounding of generated answers against the source documents
- **MCP Server**: Model Context Protocol server for integration with AI assistants
- **REST API**: RESTful API for document search and RAG operations
- **Modern Web UI**: React-based interface for chat, file management, and settings
- **Debug Mode**: Toggle verbose logging via UI or env (`RAG_DEBUG_MODE=true`) to skip expensive operations
- **Automated Startup**: Shell scripts for easy service management

## AI Backend Options

Agentic RAG supports three AI backends, each with different trade-offs:

### 1. Local (Ollama) - Default
**Best for**: Privacy, offline use, no API costs

- 100% local and private
- Works offline
- Free to use
- Fast responses (local inference)
- Lower quality than GPT-4
- Requires local GPU/CPU resources

**Setup**: Install Ollama and pull a model
```bash
brew install ollama
ollama pull qwen2.5:3b
# Start the app, then go to Settings > AI Provider > Ollama to configure
```

Ollama is the default backend and activates automatically when services start.

### 2. OpenAI Assistants - **NEW!**
**Best for**: GPT-4 quality while keeping documents private

- Excellent reasoning and answer quality
- Documents stay local (only queries sent to API)
- Automatic function calling orchestration
- Built-in citation support
- No file upload needed
- Costs ~$0.01 per query
- Requires internet and API key

**Setup**: See [docs/openai_assistants.md](docs/openai_assistants.md)
```bash
# Configure via UI (Settings > AI Provider > OpenAI Assistants)
# 1. Enter API key and save (backend reloads automatically)
# 2. Test connection
# 3. Provider appears in Active Provider dropdown
# 4. Click "Use this Backend" button to switch
```

Configuration is saved to `secrets/openai_config.json`. The backend automatically reloads when you save configuration changes - no restart required!

### 3. Google Vertex AI
**Best for**: Drive/Gmail integration, massive context windows

- 2M token context window
- Native Drive and Gmail OAuth
- Automatic grounding with Agent Builder
- Multi-modal support (PDFs, images, video)
- Complex setup (GCP project required)
- Files uploaded to Google Cloud
- Costs per query + storage

**Setup**: See [docs/vertex_ai_setup.md](docs/vertex_ai_setup.md) and [docs/google_integration.md](docs/google_integration.md)

### Comparison Table

| Feature | Local (Ollama) | OpenAI Assistants | Google Vertex AI |
|---------|----------------|-------------------|------------------|
| **Cost** | Free | ~$0.01/query | ~$0.005/query + storage |
| **Privacy** | 100% local | Queries sent | Files + queries sent |
| **Quality** | Good | Excellent | Excellent |
| **Setup** | Easy | Medium | Complex |
| **Offline** | Yes | No | No |
| **Context** | 8K-128K | 128K | 1M-2M |
| **Cloud Storage** | No | No | Yes (Drive/Gmail) |

**Switch backends anytime** - your local FAISS index works with all three!

## Installation

To get started with the Agentic RAG project, clone the repository and install the required dependencies:

```bash
$ git clone https://github.com/yourusername/agentic-rag.git
...
$ cd agentic-rag
agentic-rag$ ./start.sh --role monolith
...
```

### Prerequisites

- **Python 3.11+** (required)
- **Ollama** (for local LLM inference)
- **libomp** (macOS only, required for FAISS/Torch - installed automatically by start.sh)
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

The system uses a dual-layer configuration approach:
1. **Environment Variables (`.env`)**: For secrets, infrastructure settings, and defaults.
2. **Runtime Configuration (`config/settings.json`)**: For dynamic application settings that can be changed via the UI.

#### Environment Setup (`.env`)

Copy the example environment file and customize as needed:

```bash
agentic-rag$ cp .env.example .env
```

Key configuration options in `.env`:

- `RAG_BACKEND_TYPE`: Backend type (`local` or `remote`)
- `RAG_REMOTE_URL`: URL for remote backend (if type is `remote`)
- `EMBED_MODEL_NAME`: Embedding model (default: Snowflake/arctic-embed-xs) - **overridden by settings.json**
- `LLM_MODEL_NAME`: LLM model for completions (default: ollama/llama3.2:1b) - **overridden by settings.json**
- `OLLAMA_API_BASE`: Ollama server URL (default: 127.0.0.1:11434) - **overridden by settings.json**
- `RAG_DB`: Path to document store (default: ./cache/rag_store.jsonl)
- `MCP_HOST`/`MCP_PORT`: MCP server binding (default: 127.0.0.1:8000)
- `REST_HOST`/`REST_PORT`: REST API binding (default: 127.0.0.1:8001)
- `MAX_MEMORY_MB`: Memory limit for HTTP server (default: 75% of system RAM)

**Priority**: `config/settings.json` values take precedence over `.env` variables for models and API endpoints.

#### Runtime Configuration (`config/settings.json`)

Application settings like model selection, temperature, and search parameters can be modified at runtime without restarting the server. These settings are stored in `config/settings.json` and are **automatically reloaded** when saved via the **Settings Dashboard** in the web UI.

- **Local backend toggle**: Save `"allowLocalBackend": false` in `config/settings.json` (or export `ALLOW_LOCAL_BACKEND=0`) to hide the Ollama/local option in the UI without disabling OpenAI/Google modes. Re-enable it later via the same flag or the UI toggle added to the Settings dashboard.

Available runtime settings:
- **Model Selection**: Switch between available Ollama models (auto-reloaded on save)
- **Temperature**: Adjust generation creativity
- **API Endpoints**: Configure Ollama, MCP, and REST server URLs
- **Search Parameters**: Configure top-k results and context window size
- **Embedding Models**: Select sentence transformer models for embeddings

**Note**: Changes to `config/settings.json` are picked up immediately for new requests - no server restart needed!

You can hide or redisplay the local (Ollama) backend via the `allowLocalBackend` flag in `config/settings.json` or by toggling "Allow Local Backend" in the Settings dashboard. Even when you run `start.sh --skip-ollama`, you can re-enable the backend at runtime by setting `allowLocalBackend: true` and saving the config—the server honors that value even if the skip flag was used earlier.

## Quick Start

### Cross-Platform Python Launcher (Windows/Mac/Linux)

The easiest way to start on any platform:

```bash
# Start all services
python start.py

# Stop all services
python stop.py
```

**Options:**
```bash
python start.py --help                 # Show all options
python start.py --skip-ollama          # Skip Ollama (OpenAI/Gemini only)
python start.py --skip-ui              # Skip UI
python start.py --create-venv          # Recreate virtual environment
python start.py --role server          # Server mode (no UI)
python start.py --role client          # Client mode (no Ollama/MCP)
```

### Using the Bash Script (Mac/Linux only)

The `start.sh` script supports different roles for flexible deployment:

1. **Monolith Mode** (Default): Runs everything (Ollama, REST API, MCP Server) locally.
   ```bash
   ./start.sh --role monolith
   ```

2. **Server Mode**: Runs only the REST API and Ollama (useful for a dedicated backend server).
   ```bash
   ./start.sh --role server
   ```

3. **Client Mode**: Runs only the MCP Server (connects to a remote REST API).
   ```bash
   ./start.sh --role client
   ```

**Additional Flags:**
- `--skip-ollama`: Skip Ollama (use OpenAI Assistants or Google Gemini only)
- `--skip-model-pull`: Skip automatic Ollama model pulling
- `--skip-rest`: Skip starting the REST API server
- `--skip-ui`: Skip starting the UI (if applicable)
- `--env FILE`: Use a custom environment file

`start.py`/`start.sh` install `torch` up front: `x86_64` systems get `torch==2.2.2` from PyPI, while other architectures download the latest compatible wheel directly from `https://download.pytorch.org/whl/cpu`. You don't need to manage `torch` via `requirements.txt` — just run the launcher and the right build is pulled automatically before the remaining packages install.

If you still see warnings about `_ARRAY_API`, the launcher exports `PYTORCH_ENABLE_NUMPY_ARRAY_API=1` before torch loads so the NumPy array API initializes cleanly (with `numpy>=1.26.4` from `requirements.txt`).

To stop all services:

```bash
agentic-rag$ ./stop.sh
...
```

#### Startup Script Options

```bash
./start.sh [OPTIONS]

Options:
  --role {monolith,server,client}  Startup role (default: monolith)
  --skip-ollama                    Skip Ollama (OpenAI/Gemini only)
  --skip-model-pull                Skip automatic model pulling
  --skip-rest                      Skip starting REST API
  --skip-ui                        Skip starting UI
  -h, --help                       Show detailed help
  --env FILE                       Use custom environment file (default: .env)
  --venv NAME                      Use custom venv name (default: .venv)
  --python CMD                     Python command to use (default: python3.11)
  --recreate-venv                  Force recreation of virtual environment
```

Examples:

```bash
# Start as a backend server
agentic-rag$ ./start.sh --role server

# Start as a client connecting to a remote backend
agentic-rag$ ./start.sh --role client

# Use OpenAI Assistants only (no Ollama)
agentic-rag$ ./start.sh --skip-ollama

# Use Google Gemini only (no Ollama)
agentic-rag$ ./start.sh --skip-ollama
```

### Manual Startup

If you prefer to start services individually:

#### Terminal 1: Start Ollama

```bash
ollama serve
```

#### Terminal 2: Start REST API Server (Backend)

```bash
uvicorn src.servers.rest_server:app --host 127.0.0.1 --port 8001
```

#### Terminal 3: Start MCP Server (Frontend/Client)

```bash
# Ensure RAG_BACKEND_TYPE=remote and RAG_REMOTE_URL are set in .env if connecting to REST API
python -m src.servers.mcp_server
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

The CLI agent provides full access to search, backend switching, and model selection:

```bash
# Basic usage
python src/clients/cli_agent.py "your search query"
python src/clients/cli_agent.py "index documents"

# Get help and see all options
python src/clients/cli_agent.py --help
```

**Backend Management:**
```bash
# Show current backend
python src/clients/cli_agent.py --show-backend

# List available backends
python src/clients/cli_agent.py --list-backends

# Switch backends
python src/clients/cli_agent.py --set-backend local
python src/clients/cli_agent.py --set-backend openai_assistants
python src/clients/cli_agent.py --set-backend google_gemini

# Query with backend switch
python src/clients/cli_agent.py "Fast query" --set-backend local
```

**Model Selection (Ollama only):**
```bash
# List available models
python src/clients/cli_agent.py --list-models

# Use specific model
python src/clients/cli_agent.py "Explain architecture" --model llama3.2:3b

# Adjust temperature
python src/clients/cli_agent.py "Suggest ideas" --temperature 0.9

# Combine parameters
python src/clients/cli_agent.py "Complex query" \
  --model qwen2.5:3b \
  --temperature 0.7 \
  --top-k 10 \
  --max-tokens 2048
```

**Async Mode:**
```bash
# Long-running query with polling
python src/clients/cli_agent.py "Detailed analysis" --async --timeout 600
```

See [docs/api_model_selection.md](docs/api_model_selection.md) for complete CLI reference.

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

Index a URL:
```bash
curl -s http://127.0.0.1:8001/api/index_url \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com"}' | jq
```

### Debug Mode

For testing without expensive embedding operations:
```bash
export RAG_DEBUG_MODE=true
python src/clients/cli_agent.py "index documents"  # Fast indexing without ML
```

### MCP Integration

The MCP server exposes document indexing and search tools that can be used by MCP-compatible AI assistants. See the [MCP documentation](docs/mcp/index.md) for complete setup and usage instructions.

## Mobile App

The project includes a React Native mobile application (using Expo) located in the `mobile/` directory.

### Setup

1.  Navigate to the mobile directory:
    ```bash
    cd mobile
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm start
    ```

### Configuration

The mobile app connects to the REST API server. By default, it expects the server to be running at `http://localhost:8001/api`.

**Important for Emulators/Devices:**
-   **Server Configuration**: To allow access from emulators or physical devices, you must configure the REST API to listen on all interfaces. Add `RAG_HOST=0.0.0.0` to your `.env` file and restart the server.
-   **iOS Simulator**: `http://localhost:8001/api` works out of the box.
-   **Android Emulator**: Change `API_BASE` in `mobile/App.js` to `http://10.0.2.2:8001/api`.
-   **Physical Device**: Change `API_BASE` in `mobile/App.js` to your computer's LAN IP (e.g., `http://192.168.1.50:8001/api`).

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

## Development Standards

This project follows strict coding standards enforced by:

- **PEP-8**: Python style guide compliance (enforced via `.pylintrc`)
- **Cursor Rules**: Project-specific rules in `.cursor/rules/` directory
  - Code quality and documentation standards
  - Testing requirements
  - UI/UX guidelines
  - See `.cursor/rules/README.md` for details

All code must:
- Pass pylint checks (minimum score: 7.0/10)
- Include comprehensive docstrings
- Have corresponding tests
- Follow import organization (standard → third-party → local)
- Use explicit encoding for file operations
- Include timeout arguments for HTTP requests

## Resources

- [Ollama - llama3.2](https://ollama.com/library/llama3.2)
- [LiteLLM Documentation](https://docs.litellm.ai/docs/)
- [MCP Documentation](docs/mcp/index.md) - Complete MCP integration guide
- [MCP Python SDK](https://pypi.org/project/mcp/)
- [FastAPI](https://fastapi.tiangolo.com)
- [MCPHost](https://mcphub.tools/detail/mark3labs/mcphost)
