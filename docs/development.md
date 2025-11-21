# Developer Guide: Integrating with the REST Shim

The REST shim (`src/servers/mcp_app/api.py`) exposes a JSON API for indexing, searching, and monitoring the Agentic RAG system. The full OpenAPI 3.1 specification lives in `docs/openapi.yaml`. This guide summarizes the essentials and shows example requests using `curl`.

## Getting Started

1. Start the platform (`./start.sh`) so the shim listens on `http://127.0.0.1:8000`.
2. Review `docs/openapi.yaml` to generate SDKs or client bindings.
3. Send JSON requests with standard HTTP tools (curl, Postman, etc.). Authentication is not required on the default deployment.

## Key Endpoints

| Method | Path                | Description                                  |
|--------|---------------------|----------------------------------------------|
| POST   | `/upsert_document`  | Queue an upsert job for a single document    |
| POST   | `/index_path`       | Recursively index files under a path         |
| POST   | `/search`           | Run synchronous or asynchronous search       |
| GET    | `/documents`        | List stored documents                        |
| POST   | `/documents/delete` | Remove specific documents                    |
| POST   | `/flush_cache`      | Clear the store and remove the JSONL cache   |
| GET    | `/health`           | Check vector/store counts and memory usage   |
| GET    | `/jobs`             | Inspect queued/active background jobs        |

Refer to the OpenAPI file for full schemas, error responses, and field descriptions.

## Examples

### Upsert a Document

```bash
curl -X POST http://127.0.0.1:8000/upsert_document \
  -H "Content-Type: application/json" \
  -d '{
        "uri": "docs/strategy.md",
        "text": "Strategy notes for Q4 launch..."
      }'
```

**Response**
```json
{
  "job_id": "f6421f56-1e04-43c0-9e97-115cf3932b6d",
  "status": "queued"
}
```

### Index an Entire Folder

```bash
curl -X POST http://127.0.0.1:8000/index_path \
  -H "Content-Type: application/json" \
  -d '{
        "path": "./docs",
        "glob": "**/*.md"
      }'
```

Returns a job id that you can track via `/jobs`.

### Search Synchronously

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize our Q4 launch goals"}'
```

**Response**
```json
{
  "answer": "The launch focuses on ...",
  "sources": [
    "docs/strategy.md",
    "docs/roadmap.md"
  ]
}
```

### Search Asynchronously

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Describe the Justin Cook career timeline",
        "async": true,
        "timeout_seconds": 120
      }'
```

The shim immediately returns `{ "job_id": "...", "status": "queued" }`. Poll `/jobs` until it reports `completed`, `failed`, or `timeout`.

### List Documents

```bash
curl http://127.0.0.1:8000/documents
```

### Delete Documents

```bash
curl -X POST http://127.0.0.1:8000/documents/delete \
  -H "Content-Type: application/json" \
  -d '{"uris": ["docs/strategy.md", "docs/roadmap.md"]}'
```

### Flush Cache

Useful when you want a clean slate and to remove `RAG_DB`.

```bash
curl -X POST http://127.0.0.1:8000/flush_cache
```

### Health & Jobs

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/jobs
```

## Using the OpenAPI Specification

The `docs/openapi.yaml` file can be fed into tooling such as:

- Swagger UI / Redoc (`npx redoc-cli serve docs/openapi.yaml`)
- OpenAPI Generator (`openapi-generator-cli generate ...`)
- Postman import / VS Code REST clients

This ensures clients stay synchronized with the server contract.

## Development Tips

1. Keep `SYSTEM_PROMPT` / `GROUNDING_SYSTEM_PROMPT` in `.env` aligned with how your clients expect answers.
2. Use `/health` in CI to verify a deployment before running integration tests.
3. To run integration tests locally, spin up the shim via `./start.sh`, then run `pytest tests/test_http_server.py`.
4. When adding new endpoints, update `docs/openapi.yaml` and this guide so other teams can integrate quickly.

For more details on configuration knobs or embedding behavior, see `docs/models-and-configuration.md`.
