# MCP Planner Configuration

The planner is responsible for orchestrating how AI assistants interact with the MCP tools. It defines the behavior, policies, and call chains for different types of queries.

## Planner System Prompt

The planner uses the system prompt defined in `config/mcp/planner_prompt.md`. This prompt instructs the AI assistant on:

1. **Evidence Finding**: How to locate relevant information
2. **Answer Synthesis**: Creating grounded responses from evidence
3. **Grounding Verification**: Ensuring answers are supported by sources
4. **Iterative Refinement**: When and how to improve answers

## Default Call Chain

For most questions, the planner follows this sequence:

1. `retrieval-server.search_tool` - Find relevant passages
2. `retrieval-server.rerank_tool` - Improve passage ranking
3. `retrieval-server.grounded_answer_tool` - Generate cited answer
4. `retrieval-server.verify_grounding_tool` - Verify answer quality

## Tool Usage Policies

### Document Indexing

- **Local Files**: Use `retrieval-server.index_documents_tool` with `path` parameter
  - Example: "index documents" → `index_documents_tool(path="documents")`
  - Supports: PDF, DOCX, TXT, HTML files

- **URLs**: Use `retrieval-server.index_url_tool` with `url` parameter
  - Example: "index http://example.com/doc.pdf" → `index_url_tool(url="http://example.com/doc.pdf")`

### Document Queries

- **Content Questions**: Use the default call chain
  - Example: "What does the document say about X?"

- **Document Lists**: Use `retrieval-server.list_indexed_documents_tool`
  - Example: "What documents are indexed?" → `list_indexed_documents_tool()`

### Special Cases

- **Multi-hop Questions**: May use graph analysis tools (when available)
- **Date-sensitive Queries**: Prefer newer evidence with dates included
- **Unknown Information**: Refuse to speculate beyond available evidence

## Quality Assurance

### Grounding Verification

After generating answers, the planner always verifies:

- **Confidence Score**: Must be ≥ 0.70
- **Citation Coverage**: Must be ≥ 0.80

If verification fails, the planner refines the answer and re-retrieves once.

### Output Requirements

- **Citations**: Inline tags [1], [2] matching returned passages
- **Code Examples**: Concise and runnable
- **Evidence Basis**: All claims supported by sources

## Customization

### Modifying Planner Behavior

Edit `config/mcp/planner_prompt.md` to customize:

- Call chain sequences
- Quality thresholds
- Tool selection logic
- Output formatting requirements

### Creating Custom Planners

Create new prompt files for different use cases:

1. Copy `config/mcp/planner_prompt.md`
2. Modify policies and behavior
3. Update `mcp.yaml` to reference the new prompt file

## Examples

### Basic Question
```
User: "What are the main features of this project?"
Planner: search_tool → rerank_tool → grounded_answer_tool → verify_grounding_tool
```

### Document Indexing
```
User: "Index the documents in the docs folder"
Planner: index_documents_tool(path="documents")
```

### List Operation
```
User: "What documents have you indexed?"
Planner: list_indexed_documents_tool()
```

## Troubleshooting

### Low Confidence Scores

- Check document relevance to the query
- Verify documents are properly indexed
- Consider rephrasing the question

### Missing Citations

- Ensure documents contain the relevant information
- Check that indexing completed successfully
- Verify document formats are supported

### Tool Selection Issues

- Review the planner prompt for correct tool mappings
- Check that all required tools are available
- Validate MCP server connectivity