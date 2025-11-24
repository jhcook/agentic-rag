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

### 4. Health Check

Check the status of the server and get basic statistics.

**Endpoint:** `GET /api/health`

**Example:**

```bash
curl "http://localhost:8001/api/health"
```

## Authentication

Some endpoints, particularly those related to Google Drive integration, may require authentication. The API supports OAuth2 flow for Google services.

See [Google Integration](./google_integration.md) for more details.
