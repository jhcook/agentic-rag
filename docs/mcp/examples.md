# MCP Integration Examples

This document provides practical examples of using the MCP integration with AI assistants.

## Basic Setup

### Starting the MCP Server

```bash
# Terminal 1: Start the MCP server
python src/servers/mcp_server.py

# The server will be available at http://127.0.0.1:8000/mcp
```

### Configuring MCPHost

1. Copy the configuration:
   ```bash
   cp config/mcp/mcp.yaml ~/mcphost/config/
   ```

2. Update paths in `mcp.yaml` if necessary

3. Start MCPHost with your AI assistant

## Common Usage Patterns

### 1. Document Indexing

#### Index Local Documents
```
User: "Please index all the documents in the documents folder"
Assistant: I'll help you index the documents. Let me use the index_documents_tool.

index_documents_tool(path="documents")
```

#### Index a Specific File Type
```
User: "Index only PDF files from the documents directory"
Assistant: I'll index just the PDF files for you.

index_documents_tool(path="documents", glob="**/*.pdf")
```

#### Index from URL
```
User: "Index this PDF from the web: http://example.com/manual.pdf"
Assistant: I'll index that PDF document from the URL.

index_url_tool(url="http://example.com/manual.pdf")
```

### 2. Information Retrieval

#### Basic Search
```
User: "What are the main features of this project?"
Assistant: Let me search through the indexed documents to find information about the project features.

search_tool(query="What are the main features of this project?")
```

#### Specific Document Query
```
User: "According to the user manual, how do I configure the system?"
Assistant: I'll search for configuration information in the user manual.

search_tool(query="how do I configure the system")
```

#### List Available Documents
```
User: "What documents have you indexed so far?"
Assistant: Let me check what documents are currently available in the index.

list_indexed_documents_tool()
```

### 3. Quality Assurance

#### Verify Answer Grounding
```
User: "Is this answer well-supported by the documents?"
Assistant: Let me verify that the answer is properly grounded in the source documents.

verify_grounding_tool(
  question="What are the system requirements?",
  answer="The system requires Python 3.11+ and 8GB RAM",
  citations=["doc1.pdf", "requirements.txt"]
)
```

## Advanced Examples

### Multi-Step Research
```
User: "Research the company's history and current projects"
Assistant: This is a complex query that requires multiple steps. Let me break it down:

1. First, I'll search for company history
2. Then look for current projects
3. Finally, synthesize the information

search_tool(query="company history and background")
search_tool(query="current projects and initiatives")
```

### Document Analysis Workflow
```
User: "Analyze the quarterly report and summarize key metrics"
Assistant:
1. Index the quarterly report if not already indexed
2. Search for key metrics and financial data
3. Generate a comprehensive summary

index_documents_tool(path="documents", glob="**/quarterly-report.pdf")
search_tool(query="key metrics and financial performance")
```

## Integration with AI Assistants

### VS Code Copilot
When using with GitHub Copilot in VS Code:

1. Ensure MCPHost is running with the correct configuration
2. Copilot will automatically have access to the RAG tools
3. Use natural language to request document operations

### Other MCP-Compatible Assistants
For other AI assistants supporting MCP:

1. Configure the assistant to connect to the MCP server
2. Use the server URL: `http://127.0.0.1:8000/mcp`
3. The assistant will discover available tools automatically

## Error Handling Examples

### Missing Documents
```
User: "Index documents from the docs folder"
Assistant: Let me try to index the documents, but I need to check if the path exists.

index_documents_tool(path="docs")

Error: "docs directory not found"
Assistant: It seems the "docs" directory doesn't exist. Let me check what directories are available or suggest the correct path.

list_indexed_documents_tool()
Assistant: The documents directory should be "documents", not "docs". Let me try that.

index_documents_tool(path="documents")
```

### Unsupported File Types
```
User: "Index this image file"
Assistant: I need to check if image files are supported. Let me try indexing it.

index_documents_tool(path="images", glob="**/*.jpg")

Result: {"indexed": 0, "uris": [], "error": "No supported files found"}
Assistant: It looks like image files aren't supported. The system supports PDF, DOCX, TXT, and HTML files.
```

## Best Practices

### Indexing Strategy
- Index documents before asking questions about them
- Use specific file patterns when you only need certain types
- Consider document organization for better search results

### Query Formulation
- Be specific in your questions for better results
- Use natural language - the system understands context
- Follow up with verification for critical information

### Workflow Optimization
- Batch similar operations together
- Use the document listing tool to check what's available
- Leverage the reranking for improved result quality

## Troubleshooting

### Connection Issues
```
Assistant: I'm having trouble connecting to the MCP server. Let me check if it's running.

Error: Connection refused
Solution: Make sure the MCP server is started with `python src/servers/mcp_server.py`
```

### Empty Results
```
Assistant: The search returned no results. This could mean:
1. Documents haven't been indexed yet
2. The query doesn't match any content
3. The documents are in an unsupported format

Solution: First index documents, then try a broader search query
```

### Quality Issues
```
Assistant: The answer confidence is low. Let me verify the grounding.

verify_grounding_tool(question="...", answer="...")

Result: {"confidence": 0.65, "citation_coverage": 0.70}
Solution: The answer needs improvement. Let me refine the search or rephrase the question.
```