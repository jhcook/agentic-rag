# REST API Reference

The Agentic RAG system provides a comprehensive REST API for indexing, searching, and managing documents. This API is built using FastAPI and supports OpenAPI 3.1.0 specifications.

## OpenAPI Specifications

You can download the full OpenAPI specifications here:

- [openapi.yaml](./openapi.yaml)
- [openapi.json](./openapi.json)

These files can be imported into tools like [Postman](https://www.postman.com/) or [Swagger UI](https://swagger.io/tools/swagger-ui/) to explore and test the API interactively.

## Base URL

By default, the API is served at:

```
http://localhost:8001
```

The base path for API endpoints is `/api`.

## Common Endpoints

### 1. Search

Search the retrieval store for relevant documents.

**Endpoint:** `POST /api/search`

**Request:**

```json
{
  "query": "What is the Model Context Protocol?",
  "async": false
}
```

**Example:**

```bash
curl -X POST "http://localhost:8001/api/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the Model Context Protocol?"}'
```

### 2. Upsert Document

Add or update a document in the store.

**Endpoint:** `POST /api/upsert_document`

**Request:**

```json
{
  "uri": "file:///path/to/doc.txt",
  "text": "Content of the document..."
}
```

**Example:**

```bash
curl -X POST "http://localhost:8001/api/upsert_document" \
     -H "Content-Type: application/json" \
     -d '{
           "uri": "manual:example",
           "text": "This is an example document indexed via the REST API."
         }'
```

### 3. Index Path

Index a local filesystem path.

**Endpoint:** `POST /api/index_path`

**Request:**

```json
{
  "path": "/Users/username/documents",
    "glob": "**/*.md"
  }
```

**Example:**

```bash
curl -X POST "http://localhost:8001/api/index_path" \
     -H "Content-Type: application/json" \
     -d '{
           "path": "./docs",
           "glob": "**/*.md"
         }'
```

### 4. Index URL

Index content from a remote URL via the MCP worker.

**Endpoint:** `POST /api/index_url`

**Request:**

```json
{
  "url": "https://example.com/article",
  "doc_id": "optional-custom-id",
  "query": "optional query for extraction"
}
```

**Example:**

```bash
curl -X POST "http://localhost:8001/api/index_url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'
```

### 5. Health Check

Check the status of the server and get basic statistics.

**Endpoint:** `GET /api/health`

**Example:**

```bash
curl "http://localhost:8001/api/health"
```

## Ollama Cloud Configuration

Manage credentials and endpoints for Ollama Cloud without exposing secrets.

### GET /api/ollama/cloud-config

Returns stored cloud configuration. API keys are **masked** with the placeholder `***MASKED***`; use the `has_api_key` boolean to detect if a key is present.

```json
{
  "api_key": "***MASKED***",
  "has_api_key": true,
  "endpoint": "https://ollama.com",
  "proxy": "http://proxy.local:8080",
  "ca_bundle": "/path/to/corp.pem"
}
```

### POST /api/ollama/cloud-config

Persists cloud settings. Send a real key to update it, or send the masking placeholder to keep the stored key unchanged.

```json
{
  "api_key": "sk-ollama-123",        // or "***MASKED***" to leave existing key untouched
  "endpoint": "https://ollama.com",
  "proxy": "http://proxy.local:8080",
  "ca_bundle": "/path/to/corp.pem"
}
```

## Authentication

Some endpoints, particularly those related to Google Drive integration, may require authentication. The API supports OAuth2 flow for Google services.

See [Google Integration](./google_integration.md) for more details.

## Secret Masking Behavior

- Secrets returned by configuration endpoints are masked with `***MASKED***`.
- A companion `has_api_key` flag indicates whether a secret is stored server-side.
- When updating configs, send the masking placeholder to keep the stored key; send an empty string to clear; send a new value to replace.

### Delete/Clear the Ollama Cloud API key

- To remove a stored Ollama Cloud key, call `POST /api/ollama/cloud-config` with `api_key` set to `""` (empty string) or `null`. Example:

  ```bash
  curl -X POST "http://localhost:8001/api/ollama/cloud-config" \
       -H "Content-Type: application/json" \
       -d '{"api_key": "", "endpoint": null, "proxy": null, "ca_bundle": null}'
  ```
- After clearing, `GET /api/ollama/cloud-config` will return `has_api_key: false`.

### Retention and storage for Ollama Cloud credentials

- Secrets are stored locally in `secrets/ollama_cloud_config.json` with restrictive file permissions.
- They persist until you overwrite them or clear them via the deletion call above; there is no automatic time-based deletion.
- Removing the key stops further transmission to Ollama Cloud; data already sent to Ollama Cloud remains subject to Ollama's own retention policy.
