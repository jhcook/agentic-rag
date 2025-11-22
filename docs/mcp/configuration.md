# MCP Configuration

This document explains how to configure the MCP integration for connecting AI assistants to the Agentic RAG system.

## MCPHost Configuration

The `config/mcp/mcp.yaml` file contains the complete configuration for MCPHost:

```yaml
llm:
  provider: ollama
  base_url: http://127.0.0.1:11434
  model: qwen2.5:3b

mcpServers:
  retrieval-server:
    type: remote
    url: http://127.0.0.1:8000/mcp

planner:
  system_prompt_file: ./config/mcp/planner_prompt.md
```

### Configuration Sections

#### LLM Configuration
- `provider`: LLM provider (currently supports `ollama`)
- `base_url`: Base URL for the LLM API
- `model`: Model name to use for completions (e.g., `qwen2.5:3b`)

#### MCP Servers
- `retrieval-server`: Connection to the Agentic RAG MCP server
  - `type`: Connection type (`remote` for HTTP-based servers)
  - `url`: Full URL to the MCP server endpoint

#### Planner Configuration
- `system_prompt_file`: Path to the planner system prompt file

## Environment Setup

### Prerequisites

1. **Ollama**: Install and run Ollama server
   ```bash
   brew install ollama
   ollama serve
   ```

2. **MCPHost**: Install MCPHost for your AI assistant
   ```bash
   # Installation depends on your AI assistant
   # See MCPHost documentation for details
   ```

### Starting Services

1. **Start the MCP Server**:
   ```bash
   # Monolith mode (default)
   ./start.sh --role monolith

   # OR Client mode (if connecting to remote backend)
   ./start.sh --role client
   ```

2. **Configure MCPHost**:
   - Copy `config/mcp/mcp.yaml` to your MCPHost configuration directory
   - Update paths if necessary for your environment

3. **Start MCPHost** with your AI assistant

## Configuration Options

### Alternative LLM Providers

The system supports different LLM providers through LiteLLM. Modify the `llm` section:

```yaml
llm:
  provider: openai
  api_key: your-api-key-here
  model: gpt-4
```

### Multiple MCP Servers

You can connect to multiple MCP servers:

```yaml
mcpServers:
  retrieval-server:
    type: remote
    url: http://127.0.0.1:8000/mcp
  another-server:
    type: remote
    url: http://127.0.0.1:8001/mcp
```

### Custom Planner Prompts

Create custom planner behavior by modifying `config/mcp/planner_prompt.md`. The planner controls how the AI assistant uses the available tools.

## Troubleshooting

### Connection Issues

- Verify the MCP server is running on the configured port
- Check that the URL in `mcp.yaml` matches the server endpoint
- Ensure firewall allows connections to the MCP port

### LLM Issues

- Confirm Ollama is running and accessible
- Test the LLM connection independently
- Check model availability: `ollama list`

### Configuration Errors

- Validate YAML syntax in `mcp.yaml`
- Ensure file paths are correct and accessible
- Check that the planner prompt file exists