# OpenAI Assistants Backend

## Overview

The OpenAI Assistants backend provides GPT-4 quality reasoning while keeping your documents **completely local**. Unlike other cloud RAG solutions, this uses a **function calling bridge**: OpenAI orchestrates the conversation, but all document data stays in your local FAISS index.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │   OpenAI Assistant (GPT-4)    │
            │  - Orchestrates conversation  │
            │  - Decides when to search     │
            │  - Synthesizes responses      │
            └───────────┬───────────────────┘
                        │
                        │ Function Call:
                        │ search_documents(query="...", top_k=5)
                        │
                        ▼
            ┌───────────────────────────────┐
            │   Local FAISS Index           │
            │  - Your documents             │
            │  - Never uploaded to cloud    │
            │  - Instant search             │
            └───────────┬───────────────────┘
                        │
                        │ Returns results
                        │
                        ▼
            ┌───────────────────────────────┐
            │   OpenAI Assistant            │
            │  - Receives search results    │
            │  - Synthesizes answer         │
            │  - Adds citations             │
            └───────────┬───────────────────┘
                        │
                        ▼
            ┌───────────────────────────────┐
            │   User Response               │
            │  - GPT-4 quality answer       │
            │  - Inline citations [1], [2]  │
            │  - Sources listed             │
            └───────────────────────────────┘
```

## Key Features

### ✅ What You Get

- **GPT-4 Quality**: Superior reasoning, multi-step logic, conversation handling
- **Local Privacy**: Documents never uploaded to OpenAI, only queries and results
- **Automatic Citations**: OpenAI Assistant adds inline [1], [2] markers and sources
- **Function Orchestration**: Assistant decides when/how to search automatically
- **No File Upload**: Uses your existing FAISS index, no migration needed
- **Conversation Context**: Assistant manages multi-turn conversations naturally

### ❌ What You Don't Get (vs Google Vertex AI)

- No Drive/Gmail/OneDrive OAuth integration
- No automatic file sync from cloud storage
- No grounding metadata from cloud provider
- Costs per API call (but no storage fees)

### 💰 Cost Comparison

| Feature | Local (Ollama) | OpenAI Assistants | Google Vertex AI |
|---------|----------------|-------------------|------------------|
| LLM Calls | Free | ~$0.01/query | ~$0.001-0.01/query |
| Storage | Free (local) | Free (local) | $0.10/GB + $0.20/GB/day |
| File Upload | N/A | Not needed | Automatic |
| Privacy | 100% local | Queries sent | Queries + files sent |

## Setup

### 1. Get OpenAI API Key

1. Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-...`)

### 2. Configure Environment

Add to your `.env` file:

```bash
# Set mode to OpenAI Assistants
RAG_MODE=openai_assistants

# Required: Your OpenAI API key
OPENAI_API_KEY=sk-...

# Optional: Reuse existing assistant (otherwise creates new one)
# OPENAI_ASSISTANT_ID=asst_...

# Optional: Choose model (default: gpt-4-turbo-preview)
# OPENAI_ASSISTANT_MODEL=gpt-4-turbo-preview
# OPENAI_ASSISTANT_MODEL=gpt-4o  # Faster, cheaper
```

### 3. Index Documents Locally

```bash
# Index your documents into local FAISS
python -c "from src.core import rag_core; rag_core.index_path('./docs')"
```

### 4. Test the Backend

```bash
python test_openai_assistants.py
```

Expected output:
```
=== Testing Local Search ===
Documents in local store: 15
Local search returned 3 results

=== Testing OpenAI Assistants Backend ===
✅ OPENAI_API_KEY found
✅ Backend initialized
   Assistant ID: asst_abc123...
   Model: gpt-4-turbo-preview

--- Testing chat with function calling ---
Sending message to assistant...
(This will call the local search function automatically)

✅ Assistant response:

RAG (Retrieval-Augmented Generation) is an AI technique that combines retrieval 
systems with language models [1]. It works by first searching for relevant documents, 
then using those as context for generation [2]...

📚 Sources cited:
   - /Users/you/docs/rag_intro.md
   - /Users/you/docs/architecture.md
```

### 5. Start the Server

```bash
# Make sure RAG_MODE=openai_assistants in .env
./start.sh
```

## Usage

### Via UI

1. Navigate to Settings → AI Providers
2. Expand "OpenAI Assistants" section
3. Enter your API key and model
4. Save configuration
5. Go to Chat and ask questions
6. Assistant will automatically search your local documents

### Via REST API

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is RAG?"}
    ]
  }'
```

Response:
```json
{
  "content": "RAG stands for Retrieval-Augmented Generation [1]...",
  "sources": ["/path/to/doc1.md", "/path/to/doc2.md"]
}
```

### Via Python

```python
from src.core.openai_assistants_backend import OpenAIAssistantsBackend

# Initialize
backend = OpenAIAssistantsBackend()

# Chat
messages = [
    {"role": "user", "content": "What is RAG?"}
]
response = backend.chat(messages)

print(response['content'])
# RAG stands for Retrieval-Augmented Generation [1]...

print(response['sources'])
# ['/path/to/doc1.md', '/path/to/doc2.md']
```

## How It Works

### Assistant Creation

On first run, the backend creates an OpenAI Assistant with:

```python
{
  "name": "Local RAG Assistant",
  "model": "gpt-4-turbo-preview",
  "instructions": "You answer questions using local documents...",
  "tools": [{
    "type": "function",
    "function": {
      "name": "search_documents",
      "description": "Search locally indexed documents",
      "parameters": {
        "query": "string",
        "top_k": "integer"
      }
    }
  }]
}
```

The assistant is saved and reused across sessions (unless you change the instructions).

### Function Calling Flow

1. **User asks question**: "What is RAG?"

2. **Assistant receives question**: GPT-4 decides it needs information

3. **Assistant calls function**: 
   ```json
   {
     "name": "search_documents",
     "arguments": {
       "query": "RAG retrieval augmented generation",
       "top_k": 5
     }
   }
   ```

4. **Our backend executes**:
   ```python
   results = local_core.search(query, top_k=5)
   return json.dumps({"passages": [...], "total": 5})
   ```

5. **Assistant receives results**: 5 passages from local FAISS

6. **Assistant synthesizes answer**: Uses GPT-4 to create coherent response with citations

7. **User gets response**: High-quality answer with inline citations

### Citation Format

The assistant is instructed to use this format:

```
According to the documentation [1], RAG combines retrieval with generation [2].

Sources:
[1] /path/to/intro.md
[2] /path/to/architecture.md
```

Our backend parses this and returns structured `sources` array.

## Troubleshooting

### "OPENAI_API_KEY not set"

Add to `.env`:
```bash
OPENAI_API_KEY=sk-...
```

### "No documents found"

Index some documents first:
```bash
python -c "from src.core import rag_core; rag_core.index_path('./docs')"
```

### "Run failed: Invalid API key"

1. Check your API key is correct
2. Verify it's not expired
3. Check billing is active on OpenAI account

### "Run timed out"

OpenAI is slow or rate limited. Wait and retry. Default timeout is 60s.

### Assistant not calling function

Check assistant instructions. Recreate assistant by deleting `OPENAI_ASSISTANT_ID` from `.env` and restarting.

### Citations not appearing

Backend parses "Sources:" section from response. Ensure assistant instructions include citation format example.

## Advanced Configuration

### Custom Instructions

Modify `src/core/openai_assistants_backend.py`:

```python
instructions="""Your custom instructions here...
Make sure to include:
1. When to call search_documents
2. How to format citations
3. Tone/style preferences
"""
```

### Thread Persistence

Current implementation creates new thread per conversation. For persistent threads:

```python
# Store thread IDs
self.threads[user_id] = thread.id

# Reuse thread
if user_id in self.threads:
    thread_id = self.threads[user_id]
else:
    thread = self.client.beta.threads.create()
    thread_id = thread.id
```

### Model Selection

Available models:
- `gpt-4-turbo-preview`: Best quality, slowest, most expensive
- `gpt-4o`: Fast, good quality, cheaper
- `gpt-4o-mini`: Fastest, cheapest, reduced quality
- `gpt-4`: Original GPT-4 (slower than turbo)

Set in `.env`:
```bash
OPENAI_ASSISTANT_MODEL=gpt-4o  # Recommended for speed
```

### Function Parameters

Add more parameters to search function:

```python
"parameters": {
    "query": {"type": "string"},
    "top_k": {"type": "integer", "default": 5},
    "min_score": {"type": "number", "default": 0.5},  # NEW
    "filter_by": {"type": "string"}  # NEW
}
```

## Comparison with Other Backends

| Feature | Local (Ollama) | OpenAI Assistants | Google Vertex AI |
|---------|----------------|-------------------|------------------|
| **Cost** | Free | ~$0.01/query | ~$0.001-0.01/query |
| **Privacy** | 100% local | Queries sent | Files + queries sent |
| **Quality** | Good | Excellent | Excellent |
| **Speed** | Fast (local) | Medium (API) | Medium (API) |
| **Setup** | Easy | Medium | Complex |
| **Citations** | Basic | Automatic | Automatic |
| **Cloud Storage** | No | No | Yes (Drive/Gmail) |
| **Offline** | Yes | No | No |
| **Context Window** | 8K-128K | 128K | 1M-2M |

### When to Use Each

**Use Local (Ollama)** when:
- Privacy is critical
- Offline access needed
- No API budget
- Good enough quality

**Use OpenAI Assistants** when:
- Need best quality reasoning
- Want automatic function orchestration
- Privacy for documents OK (not queries)
- Have API budget

**Use Google Vertex AI** when:
- Need Drive/Gmail integration
- Want automatic file sync
- Have Google Cloud setup
- 2M token context needed

## Cost Estimation

### OpenAI Assistants Pricing (as of 2024)

**GPT-4 Turbo:**
- Input: $10 / 1M tokens
- Output: $30 / 1M tokens

**GPT-4o:**
- Input: $5 / 1M tokens
- Output: $15 / 1M tokens

**Typical Query:**
- User question: ~50 tokens
- Search results: ~1000 tokens
- Response: ~200 tokens
- **Total cost: ~$0.015 (GPT-4 Turbo) or ~$0.007 (GPT-4o)**

**100 queries/day:**
- GPT-4 Turbo: ~$1.50/day = ~$45/month
- GPT-4o: ~$0.70/day = ~$21/month

Much cheaper than uploading all files to OpenAI vector store ($0.10/GB + $0.20/GB/day).

## Migration from Other Backends

### From Local (Ollama)

1. Keep everything as-is
2. Set `RAG_MODE=openai_assistants`
3. Add `OPENAI_API_KEY`
4. Restart server

No re-indexing needed!

### From Google Vertex AI

1. Documents in local FAISS work as-is
2. Lose Drive/Gmail OAuth features
3. Keep google_gemini mode if needed
4. Set `RAG_MODE=openai_assistants`

### Hybrid Usage

Switch between modes:

```python
# Local for quick/private queries
rag_backend.set_mode("local")

# OpenAI for complex reasoning
rag_backend.set_mode("openai_assistants")

# Google for Drive access
rag_backend.set_mode("google_gemini")
```

## FAQ

**Q: Are my documents sent to OpenAI?**  
A: No! Only your query and the search results (small snippets) are sent. Full documents stay local.

**Q: How much does it cost?**  
A: ~$0.01 per query with GPT-4 Turbo, ~$0.005 with GPT-4o. No storage fees.

**Q: Can I use my own assistant?**  
A: Yes! Set `OPENAI_ASSISTANT_ID=asst_...` in `.env` to reuse existing assistant.

**Q: How do I switch models?**  
A: Set `OPENAI_ASSISTANT_MODEL=gpt-4o` in `.env` and restart.

**Q: Can I see what the assistant is doing?**  
A: Yes! Check logs for function calls: "Assistant calling search_documents: query='...'"

**Q: Does it work offline?**  
A: No, requires internet for OpenAI API. Use local mode for offline.

**Q: Can I add more functions?**  
A: Yes! Modify `_create_assistant()` to add more tools (e.g., calculator, web search).

**Q: How do I customize citation format?**  
A: Edit assistant instructions in `_create_assistant()` method.

## Next Steps

1. **Test it**: Run `python test_openai_assistants.py`
2. **Configure**: Set API key in `.env`
3. **Use it**: Chat in UI or via API
4. **Monitor**: Check logs for function calls
5. **Optimize**: Switch to GPT-4o for speed/cost

## Support

- **Issues**: GitHub Issues
- **Docs**: `/docs/openai_assistants_capabilities.md`
- **Examples**: `test_openai_assistants.py`
- **API Reference**: `/docs/api_reference.md`
