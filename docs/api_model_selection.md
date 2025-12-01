# Per-Query Model Selection

The REST API supports specifying the LLM model and generation parameters on a per-query basis **within the Ollama backend**. This allows you to use different Ollama models for different queries without changing configuration or switching backends.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│ UI Layer                                            │
│  ├─ Backend Selector (Ollama/OpenAI/Google)       │
│  └─ Model Selector (qwen2.5, llama3.2, etc.)      │ ← Per-query
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ HybridBackend (Provider Switcher)                  │
│  ├─ OllamaBackend ← supports model param         │
│  ├─ OpenAIAssistantsBackend (fixed GPT-4)         │
│  └─ GoogleGeminiBackend (fixed Gemini)            │
└─────────────────────────────────────────────────────┘
```

**Key Concepts:**
- **Backend switching** (Ollama ↔ OpenAI ↔ Google) is managed via `/api/config/mode`
- **Model selection within Ollama** is managed via the `model` parameter in `/api/search`
- **OpenAI** and **Google** backends use their own fixed models (configurable separately)

## Available Models API

### GET `/api/config/models`

Lists available models for the current backend.

**For the Ollama backend:**
```bash
curl http://localhost:8001/api/config/models
```

Response:
```json
{
  "models": [
    "qwen2.5:3b",
    "llama3.2:3b", 
    "mistral:7b",
    "gemma2:9b"
  ]
}
```

**For OpenAI/Google:**
Returns empty array `[]` (models are fixed per backend configuration).

## `/api/search` Endpoint

### Request Parameters

```json
{
  "query": "Your search query",
  "model": "ollama/qwen2.5:3b",         // Optional: Override LLM_MODEL_NAME
  "temperature": 0.7,                    // Optional: Override LLM_TEMPERATURE (0.0-1.0)
  "max_tokens": 2048,                    // Optional: Max tokens in response
  "top_k": 5,                            // Optional: Number of documents to retrieve
  "async": false                         // Optional: Async mode
}
```

### Examples

#### Default Model (uses config from settings.json or .env)
```bash
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?"
  }'
```

#### Custom Model Per Query
```bash
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the architecture",
    "model": "ollama/llama3.2:3b",
    "temperature": 0.3,
    "top_k": 10
  }'
```

#### Different Model for Creative vs Technical Queries
```bash
# Technical query with low temperature
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the system requirements?",
    "model": "ollama/qwen2.5:3b",
    "temperature": 0.2
  }'

# Creative query with higher temperature
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Suggest innovative use cases",
    "model": "ollama/llama3.2:3b",
    "temperature": 0.9
  }'
```

## CLI Agent Examples

The `cli_agent.py` provides full access to backend switching and model selection:

### Backend Management

```bash
# List available backends
python src/clients/cli_agent.py --list-backends

# Show current backend
python src/clients/cli_agent.py --show-backend

# Switch to Ollama (local)
python src/clients/cli_agent.py --set-backend local

# Switch to OpenAI Assistants
python src/clients/cli_agent.py --set-backend openai_assistants

# Switch to Google Gemini
python src/clients/cli_agent.py --set-backend google_gemini

# Switch to Vertex AI
python src/clients/cli_agent.py --set-backend vertex_ai_search

# Switch backend and query in one command
python src/clients/cli_agent.py "Quick query" --set-backend local
```

### Model Selection (Ollama Only)

```bash
# List available models
python src/clients/cli_agent.py --list-models

# Use default model
python src/clients/cli_agent.py "What is RAG?"

# Use specific model
python src/clients/cli_agent.py "Explain the architecture" --model llama3.2:3b

# Adjust temperature for creativity
python src/clients/cli_agent.py "Suggest ideas" --temperature 0.9

# Limit context documents
python src/clients/cli_agent.py "What features?" --top-k 10

# Combine all parameters
python src/clients/cli_agent.py "Complex query" \
  --model qwen2.5:3b \
  --temperature 0.7 \
  --max-tokens 2048 \
  --top-k 5
```

### Workflow Examples

```bash
# 1. Check which backend is active
python src/clients/cli_agent.py --show-backend

# 2. List models for current backend
python src/clients/cli_agent.py --list-models

# 3. Query with specific model
python src/clients/cli_agent.py "What is RAG?" --model llama3.2:3b

# 4. Switch to OpenAI for complex reasoning
python src/clients/cli_agent.py --set-backend openai_assistants
python src/clients/cli_agent.py "Explain quantum computing"

# 5. Switch to Google Gemini for Drive search
python src/clients/cli_agent.py --set-backend google_gemini
python src/clients/cli_agent.py "Find meeting notes from last week"

# 6. Switch back to Ollama with lightweight model
python src/clients/cli_agent.py "Simple question" --set-backend local --model qwen2.5:3b
```

### Async Mode

```bash
# Long-running query with async polling
python src/clients/cli_agent.py "Detailed explanation" \
  --async \
  --timeout 600 \
  --model llama3.2:3b
```

## Python Client Example

```python
import requests

def search_with_model(query, model=None, temperature=None):
    """Search with optional model override."""
    payload = {"query": query}
    
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    
    response = requests.post(
        "http://localhost:8001/api/search",
        json=payload,
        timeout=60
    )
    return response.json()

# Use default model
result1 = search_with_model("What is RAG?")

# Use specific model
result2 = search_with_model(
    "Explain the architecture",
    model="ollama/llama3.2:3b",
    temperature=0.3
)
```

## Async Mode with Custom Model

```bash
# Start async search with custom model
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Detailed explanation",
    "model": "ollama/qwen2.5:3b",
    "temperature": 0.5,
    "async": true
  }'

# Returns: {"job_id": "uuid-here", "status": "queued"}

# Poll for results
curl http://localhost:8001/api/search/jobs/{job_id}
```

## Available Models

Models depend on what's installed in your Ollama instance. Common options:

- `ollama/qwen2.5:3b` - Fast, balanced (default)
- `ollama/llama3.2:3b` - Meta's Llama 3.2
- `ollama/mistral:7b` - Mistral 7B
- `ollama/gemma2:9b` - Google's Gemma 2

List available models:
```bash
ollama list
```

## Temperature Guidelines

- **0.0-0.3**: Deterministic, factual responses (documentation, technical)
- **0.4-0.7**: Balanced creativity and accuracy (general questions)
- **0.8-1.0**: Creative, varied responses (brainstorming, suggestions)

## Use Cases

1. **Model A/B Testing**: Compare responses from different models for the same query
2. **Cost Optimization**: Use smaller models for simple queries, larger for complex ones
3. **Quality Control**: Use low temperature for factual queries, higher for creative tasks
4. **Performance Tuning**: Adjust parameters based on query complexity

## Notes

- **Model parameter only works with Ollama backend** (local mode)
- When using OpenAI Assistants, the model is fixed (e.g., gpt-4-turbo-preview)
- When using Google Gemini, the model is fixed (e.g., gemini-2.0-flash-exp)
- Parameters are optional - omit them to use defaults from `config/settings.json`
- Invalid model names will fallback to default model with error logging
- Use `GET /api/config/models` to see available models for current backend
