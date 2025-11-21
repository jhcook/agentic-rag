# Getting Started with Agentic RAG

This guide walks you through installing dependencies, configuring `.env`, and using the system via the CLI/REST shim as a typical user.

## 1. Requirements

- macOS or Linux
- Python 3.11+
- Node.js 18+ (for building/running the UI under `ui/`)
- git
- [Ollama](https://ollama.com/download) (for local LLM execution)

Optional: `curl` and `npx redoc-cli` for inspecting the REST API.

## 2. Clone and Bootstrap

```bash
git clone https://github.com/your-org/agentic-rag.git
cd agentic-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (for development tooling)
pip install -r requirements-dev.txt
```

## 3. Configure `.env`

1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to suit your environment:
   - `EMBED_MODEL_NAME`: Hugging Face model id or local path.
   - `LLM_MODEL_NAME`: e.g., `ollama/llama3.2:1b`.
   - `OLLAMA_API_BASE`: usually `http://127.0.0.1:11434`.
   - `SYSTEM_PROMPT` / `GROUNDING_SYSTEM_PROMPT` if you want custom answer styles.
   - `RAG_DB`: location for the JSONL document store.
   - See `.env.example` for every variable and description.

## 4. Start the Stack

Run the provided script to launch the retrieval server, REST shim, and optional UI:

```bash
./start.sh
```

This will:

- Verify Ollama is running (or start it if `START_OLLAMA=true`).
- Pull the configured LLMs the first time.
- Launch the MCP HTTP server on `http://127.0.0.1:8000/mcp`.
- Launch the REST shim on `http://127.0.0.1:8000`.
- Start the UI (if `START_UI=true`) on `http://127.0.0.1:4173`.

Logs are captured under `log/` (see `log/start.log`, `log/mcp_server.log`, etc.).

## 5. Index Documents

Use the REST shim or scripts to feed documents:

```bash
curl -X POST http://127.0.0.1:8000/upsert_document \
  -H "Content-Type: application/json" \
  -d '{"uri": "docs/roadmap.md", "text": "Launch goals ..."}'
```

Or index entire directories:

```bash
curl -X POST http://127.0.0.1:8000/index_path \
  -H "Content-Type: application/json" \
  -d '{"path": "./docs", "glob": "**/*.md"}'
```

Monitor background jobs:

```bash
curl http://127.0.0.1:8000/jobs
```

## 6. Ask Questions

Synchronous search:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the Q4 launch plan"}'
```

Asynchronous search (returns a job id immediately):

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Describe Justin Cook\u2019s career", "async": true}'
```

Poll `/jobs` for completion.

## 7. Manage the Store

- List documents: `curl http://127.0.0.1:8000/documents`
- Delete specific docs:
  ```bash
  curl -X POST http://127.0.0.1:8000/documents/delete \
    -H "Content-Type: application/json" \
    -d '{"uris": ["docs/roadmap.md"]}'
  ```
- Flush everything (in-memory + JSONL):
  ```bash
  curl -X POST http://127.0.0.1:8000/flush_cache
  ```

## 8. UI & Clients

- Visit `http://127.0.0.1:4173` (default UI) to interact in a browser.
- Use the MCP endpoint (`http://127.0.0.1:8000/mcp`) with compatible MCP clients.
- See `docs/development.md` for developers integrating via the OpenAPI spec (`docs/openapi.yaml`).

## 9. Troubleshooting

- Check `log/start.log` for startup errors.
- Review `log/mcp_server.log` and `log/rest_server.log` for runtime issues.
- Ensure Ollama is running and accessible (`ollama ps`).
- Verify `.env` values match your network environment.
- Restart via `./stop.sh` followed by `./start.sh` after changes.

## 10. Next Steps

- Customize prompts and retrieval settings through `.env`.
- Add documents regularly via the REST shim or CLI scripts.
- Build automation/integration tests using the OpenAPI spec.

Need more detail? See:

- `docs/development.md` (developer integration guide)
- `docs/models-and-configuration.md` (prompting, model selection)
- `README.md` (project overview)
