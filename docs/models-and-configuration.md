# Models and Configuration in Agentic RAG

This document explains the different AI models used in the Agentic RAG system, their purposes, and how they are integrated into the codebase.

## Overview

The Agentic RAG system uses multiple AI models working together to provide retrieval-augmented generation capabilities:

1. **Embedding Model** - Converts text into numerical vectors for semantic search
2. **LLM Models** - Generate human-like responses based on retrieved context
3. **Ollama Integration** - Local model serving and API communication

## Backend Switching

The system supports multiple backend providers that can be switched at runtime **via the REST API or UI**, without restart:

### Available Backends

1. **`ollama`** - Ollama with local and cloud models (default)
   - Supports per-query model selection
   - **Local mode**: Fast, private, runs offline
   - **Cloud mode**: Use Ollama Cloud hosted models with API key authentication
   - **Auto mode**: Try cloud first, automatically fallback to local on failure
   - Models: qwen2.5:3b, llama3.2:3b, mistral:7b, etc. (local) or cloud variants
   - See [Ollama Cloud Architecture](ollama-cloud-architecture.md) for details

2. **`openai_assistants`** - OpenAI Assistants API
   - Uses GPT-4 Turbo with file search
   - Enterprise-grade reasoning
   - Requires OpenAI API key

3. **`google_gemini`** - Google Gemini with Drive API
   - Integrates with Google Drive for document retrieval
   - Uses Gemini 2.0 Flash
   - Requires Google Cloud credentials

4. **`vertex_ai_search`** - Google Vertex AI Search
   - Enterprise search with Google Cloud
   - Managed service with SLA
   - Requires GCP project and search engine

### Backend Management API

**GET `/api/config/mode`** - Get current backend
```bash
curl http://localhost:8001/api/config/mode
```

Response:
```json
{
  "mode": "ollama",
  "available_modes": ["ollama", "openai_assistants", "google_gemini", "vertex_ai_search"]
}
```

**POST `/api/config/mode`** - Switch backend
```bash
curl -X POST http://localhost:8001/api/config/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "openai_assistants"}'
```

Response:
```json
{
  "status": "ok",
  "mode": "openai_assistants"
}
```

### CLI Backend Management

```bash
# List available backends
python src/clients/cli_agent.py --list-backends

# Show current backend
python src/clients/cli_agent.py --show-backend

# Switch backends
python src/clients/cli_agent.py --set-backend ollama
python src/clients/cli_agent.py --set-backend openai_assistants
python src/clients/cli_agent.py --set-backend google_gemini
python src/clients/cli_agent.py --set-backend vertex_ai_search

# Query with backend switch
python src/clients/cli_agent.py "My question" --set-backend ollama
```

### Backend Configuration Files

Each backend can be configured via its own config file:

- **Ollama**: 
  - `config/settings.json` → `ollamaMode`, `LLM_MODEL_NAME`, `LLM_TEMPERATURE`, `ollamaCloudEndpoint`
  - `secrets/ollama_cloud_config.json` → `api_key` (for cloud mode)
  - See [Ollama Cloud Architecture](ollama-cloud-architecture.md) for details
- **OpenAI**: `secrets/openai_config.json` → `api_key`, `model`, `assistant_id`
- **Google Gemini**: `.env` → `GOOGLE_GROUNDING_MODE=google_gemini`, `client_secrets.json`
- **Vertex AI**: `.env` → `GCP_PROJECT_ID`, `GCP_SEARCH_ENGINE_ID`

See [OpenAI Assistants Setup](openai_assistants.md) and [Google Integration](google_integration.md) for detailed configuration.

## Model Configuration

The Agentic RAG system uses a hierarchical configuration system:

1. **Environment Variables (`.env`)**: Define infrastructure settings, secrets, and default values.
2. **Runtime Configuration (`config/settings.json`)**: Stores dynamic settings that can be modified via the UI or API without restarting.

### Environment Variables (`.env`)

These settings are loaded at startup and define the baseline configuration:

```dotenv
EMBED_MODEL_NAME=Snowflake/arctic-embed-xs
LLM_MODEL_NAME=ollama/qwen2.5:3b
ASYNC_LLM_MODEL_NAME=qwen2.5:3b
OLLAMA_API_BASE=http://127.0.0.1:11434
OLLAMA_MODE=local  # local | cloud | auto
OLLAMA_CLOUD_API_KEY=your_api_key_here  # Optional, for cloud mode
OLLAMA_CLOUD_ENDPOINT=https://ollama.com  # Optional, defaults to https://ollama.com
OLLAMA_KEEP_ALIVE=-1
LLM_TEMPERATURE=0.1
RAG_BACKEND_TYPE=ollama  # or 'remote'
GOOGLE_GROUNDING_MODE=google_gemini  # or 'vertex_ai_search'
```

### Runtime Configuration (`config/settings.json`)

The system supports dynamic configuration updates. Settings stored in `config/settings.json` override the environment defaults. This allows users to:

- Switch LLM models on the fly
- Adjust temperature and generation parameters
- Modify search settings (top-k, context window)
- Update system prompts
- Toggle **Debug Mode** (`debugMode: true`) for verbose logging

Disconnecting a provider from the **Settings → AI Provider** panel also updates this file.
Choosing **Disconnect** on the Ollama card clears the stored Ollama parameters, sets `ollamaConfigured: false`,
and hides Ollama from `available_modes` until you save a new configuration—matching how the OpenAI and Google disconnect flows behave.

**Key runtime fields**

- `ollamaMode`: `local`, `cloud`, or `auto` (auto = cloud-first with local fallback).
- `ollamaLocalModel` / `ollamaCloudModel`: distinct model names for local vs. cloud. The UI populates dropdowns by fetching local or cloud models independently.
- `proxy`: shared HTTPS proxy applied to outbound requests (including Ollama Cloud) where supported.
- `caBundlePath` / `ollamaCloudCABundle`: PEM paths used for TLS verification; relative paths resolve from the repo root.
- `CHAT_STORE_DB` (env): Optional SQLite path for chat history. Defaults to `data/chat_store.db` (created if missing).

**Managing Configuration:**
- **Web UI**: Use the "Settings" dashboard to view and modify configuration (including Debug toggle).
- **API**: The REST API exposes endpoints (`/api/config`) to read and update settings.

## Embedding Model (`EMBED_MODEL_NAME`)

### Purpose
The embedding model converts text documents into high-dimensional numerical vectors that capture semantic meaning. This enables semantic similarity search rather than simple keyword matching.

### Current Configuration
- **Model**: `Snowflake/arctic-embed-xs`
- **Dimensions**: 384-dimensional vectors
- **Provider**: Hugging Face via SentenceTransformers

### Code Integration

**Loading the Model** (`src/core/embeddings.py`):
```python
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", ...) -> Optional[SentenceTransformer]:
    # ...
    if _EMBEDDER is None:
        logger.info("Loading embedding model: %s", model_name)
        _EMBEDDER = SentenceTransformer(model_name, ...)
    return _EMBEDDER
```

**Usage in Vector Search**:
```python
def _vector_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    embedder = get_embedder()
    query_emb = embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    # Search pgvector index for similar vectors
```

**Document Indexing**:
```python
def rebuild_index():
    embedder = get_embedder()
    for uri, text in _STORE.docs.items():
        chunks = _chunk_text(text)
        embeddings = embedder.encode(chunks, normalize_embeddings=True)
        # Vectors are written to pgvector during indexing
```

## Primary LLM Model (`LLM_MODEL_NAME`)

### Purpose
The primary language model generates grounded answers by synthesizing information from retrieved documents. It ensures responses are based only on the provided context.

### Current Configuration
- **Model**: `ollama/qwen2.5:3b`
- **Provider**: Ollama (local)
- **Usage**: Synchronous completions via LiteLLM

### Code Integration

**Search Function** (`src/core/rag_core.py`):
```python
def search(query: str, top_k: int = 5, max_context_chars: int = 4000):
    # 1. Vector search to find relevant documents
    candidates = _vector_search(query, k=top_k)
    
    # 2. Build context from retrieved documents
    context = build_context_from_candidates(candidates)
    
    # 3. Generate response using LLM
    resp = completion(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"}
        ],
        api_base=OLLAMA_API_BASE,
        stream=False,
        timeout=120,
    )
    return resp
```

The system prompt enforces grounding: *"Answer the user's question using ONLY the documents provided. If the answer is not contained in the documents, reply exactly: 'I don't know.'"*

## Async LLM Model (`ASYNC_LLM_MODEL_NAME`)

### Purpose
Handles asynchronous operations and bulk text processing, particularly useful for sending large amounts of text to the model or when working with async contexts.

### Current Configuration
- **Model**: `qwen2.5:3b` (same as primary but without provider prefix)
- **Provider**: Ollama AsyncClient
- **Usage**: Asynchronous chat completions

### Code Integration

**Async Text Processing** (`src/core/rag_core.py`):
```python
async def send_to_llm(query: List[str]) -> Any:
    client = AsyncClient(host=OLLAMA_API_BASE)
    messages = [{"content": str(text), "role": "user"} for text in query]
    resp = await client.chat(
        model=ASYNC_LLM_MODEL_NAME,
        messages=messages
    )
    return resp
```

**Document Store Processing**:
```python
def send_store_to_llm() -> str:
    from src.core import rag_core, document_repo
    docs = rag_core.list_documents()
    texts = []
    for doc in docs:
        uri = doc.get("uri")
        if not uri:
            continue
        text = document_repo.read_indexed_text(uri)
        if text:
            texts.append(text)
    # Handle both sync and async execution contexts
    if running_loop and running_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(send_to_llm(texts), running_loop)
        resp = future.result(timeout=120)
    else:
        resp = asyncio.run(send_to_llm(texts))
    return resp
```

## Ollama Integration

Ollama supports three operational modes:
- **Local mode**: Uses local Ollama instance (default)
- **Cloud mode**: Uses Ollama Cloud hosted models
- **Auto mode**: Tries cloud first, falls back to local on failure

See [Ollama Cloud Architecture](ollama-cloud-architecture.md) for detailed documentation.

### API Base URL (`OLLAMA_API_BASE`)

**Purpose**: Specifies the endpoint where the Ollama service is running.

**Local Mode Configuration**: `http://127.0.0.1:11434` (default Ollama port)

**Cloud Mode Configuration**: `https://ollama.com` (or custom endpoint)

**Usage**: Used by both LiteLLM and Ollama's AsyncClient for API communication. The endpoint is automatically selected based on the `ollamaMode` setting in `config/settings.json`.

### Keep Alive (`OLLAMA_KEEP_ALIVE`)

**Purpose**: Controls how long Ollama keeps models loaded in memory to improve response times.

**Current Configuration**: `-1` (keep models loaded indefinitely)

**Usage**: This is an Ollama server configuration parameter that prevents model reloading between requests, significantly improving performance for the qwen2.5:3b model.

## Temperature Control (`LLM_TEMPERATURE`)

### Purpose
Controls the randomness and creativity of LLM responses. Lower values produce more consistent, deterministic answers while higher values introduce more variation.

### Current Configuration
- **Value**: `0.1` (low temperature for consistent, grounded responses)
- **Range**: 0.0 (most deterministic) to 2.0 (most creative)
- **Default**: 0.1

### Code Integration

**Configuration Loading** (`src/core/rag_core.py`):
```python
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # Low temperature for consistent, grounded responses
```

**Usage in Completions**:
```python
resp = completion(
    model=LLM_MODEL_NAME,
    messages=[{"role": "system", "content": system_msg},
              {"role": "user", "content": user_msg}],
    api_base=OLLAMA_API_BASE,
    temperature=LLM_TEMPERATURE,  # Controls response consistency
    stream=False,
    timeout=120,
)
```

### Temperature Guidelines
- **0.0-0.3**: Highly consistent, deterministic responses (best for RAG applications)
- **0.4-0.7**: Balanced creativity and consistency
- **0.8-1.2**: More creative, varied responses
- **1.3-2.0**: Highly creative, potentially unpredictable

For retrieval-augmented generation, low temperature (0.1) ensures responses stay grounded in the retrieved documents rather than generating speculative content.

## Model Workflow

### Document Indexing Pipeline

1. **Text Extraction**: Documents are parsed from various formats (PDF, DOCX, HTML, TXT)
2. **Chunking**: Text is split into overlapping chunks (800 chars, 120 char overlap)
3. **Embedding**: Each chunk is converted to a 384-dimensional vector using `EMBED_MODEL_NAME`
4. **Storage**: Vectors are stored in pgvector with metadata linking back to source documents

### Query Processing Pipeline

1. **Query Embedding**: User query is converted to vector using same embedding model
2. **Vector Search**: pgvector finds the most similar document chunks (top-k results)
3. **Context Assembly**: Retrieved chunks are assembled into context (max 4000 chars)
4. **Answer Generation**: `LLM_MODEL_NAME` generates response using only the retrieved context
5. **Response**: Grounded answer based on indexed documents

### Async Processing

- `ASYNC_LLM_MODEL_NAME` handles bulk operations like processing entire document stores
- Used when the main processing needs to be non-blocking
- Supports both synchronous and asynchronous execution contexts

## Configuration Best Practices

### Model Selection
- **Embedding Model**: Choose models optimized for semantic similarity (384-768 dimensions typical)
- **LLM**: Balance size vs. quality; qwen2.5:3b provides good quality with reasonable resource usage
- **Async Model**: Usually same as primary LLM but can be different for specialized tasks

### Performance Tuning
- **OLLAMA_KEEP_ALIVE=-1**: Prevents model reloading for better response times
- **Chunk Size**: 800 characters with 120 character overlap balances context and precision
- **Top-K**: 5-10 results typically provide sufficient context without overwhelming the LLM

### Resource Management
- **Memory**: Monitor usage with `MAX_MEMORY_MB` setting
- **Debug Mode**: Set `RAG_DEBUG_MODE=true` to skip embeddings for testing
- **Batch Processing**: Embeddings are processed in batches of 8 to control memory usage

## Troubleshooting

### Common Issues

**Embedding Model Fails to Load**:
- Check Hugging Face model identifier
- Ensure network connectivity for model download
- Verify sufficient disk space (models can be several GB)

**LLM Connection Errors**:
- Confirm Ollama is running: `ollama serve`
- Check `OLLAMA_API_BASE` URL is correct
- Verify model is available: `ollama list`

**Slow Responses**:
- Ensure `OLLAMA_KEEP_ALIVE=-1` is set
- Check system has sufficient RAM for model size
- Consider smaller models for resource-constrained environments

**Empty Search Results**:
- Verify documents have been indexed: check `cache/indexed/` contains extracted text artifacts
- Confirm pgvector is running and has vectors
- Check debug logs for embedding failures

## Google Backend Configuration

The system supports a Google Drive-backed RAG implementation that bypasses the local vector store.

### Configuration Variables

- **`GOOGLE_GROUNDING_MODE`**: 
  - `manual`: Uses Google Drive API to search files and Gemini to generate answers (Default).
  - `vertex_ai_search`: Uses Vertex AI Agent Builder with a Data Store.

- **`GOOGLE_MODEL_NAME`**:
  - Specifies the Gemini model to use.
  - Default: `models/gemini-2.0-flash` (Fast and cost-effective)
  - Options: `models/gemini-1.5-pro`, `models/gemini-3.0-pro` (if available to your account)

### Setup

1. **Credentials**: Ensure you have `client_secrets.json` or valid Google Cloud credentials.
2. **Dependencies**: Install `google-generativeai` and `google-api-python-client`.
3. **Environment**: Set the variables in your `.env` file.

```dotenv
GOOGLE_GROUNDING_MODE=google_gemini
GOOGLE_MODEL_NAME=models/gemini-3.0-pro
```

## Related Documentation

- [README.md](../README.md) - Project overview and setup
- [MCP Configuration](mcp/configuration.md) - MCP integration setup
- [.env.example](../.env.example) - Complete environment variable reference
