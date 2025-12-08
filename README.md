# Agentic RAG

Agentic RAG (Retrieval-Augmented Generation) combines a document store, FAISS index, and multiple LLM adapters (Ollama, OpenAI Assistants, Google Vertex AI) so you can ask natural language questions over your own knowledge base.

## Features
- Local Ollama server + REST/MCP APIs for hybrid deployments
- **Ollama Cloud support** - Use cloud-hosted models with automatic local fallback
- Role-based startup (`monolith`, `server`, `client`) so you can choose which components to run
- Automatic grounding, citations, reranking, and verification powered by FAISS + Litellm
- React UI with file manager, logs, metrics, and backend switching
- Command-line helper (`cli_agent.py`) and Python APIs for scripting

## Quick Start

```bash
git clone https://github.com/yourorg/agentic-rag.git
cd agentic-rag
# start all services (Ollama, REST API, MCP server, UI)
start.sh
# stop everything
stop.sh
```

`start.sh`/`start.py` do the heavy lifting:
1. create/activate `.venv`
2. install dependencies (`pip install -r requirements.txt`)
3. install the correct `torch` wheel per-platform (Intel → `2.2.2`, others → PyTorch CPU wheel)
4. respect `--skip-ollama`, `--skip-ui`, and role flags when launching services

## Configuration & Docs

- Runtime settings (`config/settings.json`) override `.env`; edit them via the Settings UI to update Ollama endpoint, model, or logging behavior.
- **Ollama Cloud**: Configure cloud mode, API keys, and endpoints via Settings UI. See `docs/ollama-cloud-architecture.md` for details.
- Server-side details (MCP, REST, CLI) live in `docs/models-and-configuration.md`, `docs/openai_assistants.md`, and `docs/google_integration.md`.
- Desktop packaging notes are in `docs/electron-desktop.md`; API references are under `docs/api_reference.md` and `docs/api_model_selection.md`.
- Torch/NumPy handling, service logs, and advanced diagnostics belong in the same docs folder—refer there before editing the startup scripts.

## Testing & Development

```bash
# run the full pytest suite
.venv/bin/pytest tests/

# run linting
.venv/bin/pylint src/servers/rest_server.py src/servers/control_daemon.py src/core/factory.py
```

See `docs/` for more targeted guides (ops, testing, API calls, cloud integrations).
