# MCP Tools Reference

The Agentic RAG MCP server exposes several tools that AI assistants can use to interact with the document retrieval system.

## Available Tools

### Document Indexing Tools

#### `index_documents_tool`

Indexes documents from a local directory path.

**Parameters:**
- `path` (string): Directory path to index (e.g., "documents", "./docs")
- `glob` (string, optional): File pattern (default: "**/*" for all supported files)

**Supported Formats:** PDF, DOCX, DOC, TXT, HTML, HTM

**Example:**
```
index_documents_tool(path="documents", glob="**/*.pdf")
```

**Returns:**
```json
{
  "indexed": 5,
  "uris": ["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
  "error": null
}
```

#### `index_url_tool`

Indexes a document from a URL.

**Parameters:**
- `url` (string): URL of the document to index
- `doc_id` (string, optional): Custom document ID
- `query` (string, optional): Alternative way to pass URL

**Supported Formats:** PDF, DOCX, HTML, TXT

**Example:**
```
index_url_tool(url="http://example.com/document.pdf")
```

**Returns:**
```json
{
  "indexed": 1,
  "uri": "http://example.com/document.pdf",
  "doc_id": "http://example.com/document.pdf"
}
```

### Search and Retrieval Tools

#### `search_tool`

Searches indexed documents and generates an LLM-powered answer.

**Parameters:**
- `query` (string): Natural language search query
- `top_k` (integer, optional): Number of passages to consider (default: 5)

**Example:**
```
search_tool(query="What are the main features?", top_k=5)
```

**Returns:**
```json
{
  "answer": "The main features include...",
  "model": "qwen2.5:0.5b",
  "usage": {"prompt_tokens": 150, "completion_tokens": 50}
}
```

#### `rerank_tool`

Re-ranks search results using a lightweight heuristic.

**Parameters:**
- `query` (string): Original search query
- `passages` (array): List of passage objects to rerank

**Example:**
```
rerank_tool(query="machine learning", passages=[{...}, {...}])
```

#### `grounded_answer_tool`

Generates an answer with citations from search results.

**Parameters:**
- `question` (string): Question to answer
- `k` (integer, optional): Number of passages to consider (default: 5)

**Returns:** Answer with inline citations

#### `verify_grounding_tool`

Verifies that an answer is properly grounded in source documents.

**Parameters:**
- `question` (string): Original question
- `answer` (string): Generated answer
- `citations` (array): List of citation references

**Returns:**
```json
{
  "confidence": 0.85,
  "citation_coverage": 0.92,
  "issues": []
}
```

### Management Tools

#### `list_indexed_documents_tool`

Lists all indexed documents without content.

**Parameters:** None

**Returns:**
```json
{
  "uris": [
    "/path/to/document1.pdf",
    "/path/to/document2.txt"
  ]
}
```

## Tool Usage Guidelines

### When to Use Each Tool

| Task | Primary Tool | Notes |
|------|-------------|--------|
| Index local files | `index_documents_tool` | Use for PDF, DOCX, TXT files |
| Index web content | `index_url_tool` | Use for URLs pointing to documents |
| Answer questions | `search_tool` | Complete end-to-end search and answer |
| List documents | `list_indexed_documents_tool` | Metadata only, no content |
| Verify answers | `verify_grounding_tool` | Quality assurance |

### Best Practices

1. **Indexing First**: Always index documents before searching
2. **Appropriate Scope**: Use `list_indexed_documents_tool` for inventory, `search_tool` for content
3. **URL vs Local**: Use `index_url_tool` for remote documents, `index_documents_tool` for local
4. **Quality Checks**: Follow answers with `verify_grounding_tool` for critical information

### Error Handling

All tools return error information in the response:

```json
{
  "error": "Description of the error",
  "indexed": 0,
  "uris": []
}
```

Common errors:
- `docs directory not found` - Path doesn't exist
- `No files found` - No matching files in directory
- `Failed to extract text` - Unsupported file format or corrupted file

## Integration Examples

### Basic Workflow

1. Index documents:
   ```
   index_documents_tool(path="documents")
   ```

2. Search and answer:
   ```
   search_tool(query="What is the main topic?")
   ```

3. Verify quality:
   ```
   verify_grounding_tool(question="...", answer="...", citations=[...])
   ```

### Advanced Usage

For complex queries requiring multiple steps:
1. Use `search_tool` for initial retrieval
2. Apply `rerank_tool` for better ranking
3. Generate final answer with `grounded_answer_tool`
4. Always verify with `verify_grounding_tool`