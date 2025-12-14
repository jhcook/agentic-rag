# Model Context Protocol (MCP) Integration

The Agentic RAG system integrates with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to provide AI assistants with powerful document indexing and retrieval capabilities.

## Overview

MCP allows AI assistants to securely connect to external tools and data sources. This project implements an MCP server that exposes RAG (Retrieval-Augmented Generation) capabilities, enabling AI assistants to:

- Index documents from local files or URLs
- Search through indexed documents using natural language queries
- Generate grounded answers with citations
- Verify answer grounding against source documents

## Architecture

The system supports two architectural modes:

### 1. Monolithic (Local)
Everything runs on the same machine. The MCP server accesses the document store directly.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Assistant  │────│   MCP Host      │────│   MCP Server    │
│   (Copilot)     │    │   (MCPHost)     │    │  (OllamaBackend)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Document Store  │
                       │ (pgvector +     │
                       │ cache/indexed)  │
                       └─────────────────┘
```

### 2. Distributed (Remote)
The MCP server acts as a client to a remote REST API. This allows the heavy lifting (LLM inference, Vector Search) to happen on a powerful server, while the MCP server runs locally with the user.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Assistant  │────│   MCP Host      │────│   MCP Server    │
│   (Copilot)     │    │   (MCPHost)     │    │ (RemoteBackend) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────┐
                                              │    REST API     │
                                              │    (Server)     │
                                              └─────────────────┘
                                                      │
                                                      ▼
                                               ┌─────────────────┐
                                               │ Document Store  │
                                               └─────────────────┘
```

## Quick Start

1. **Start the Services**:
   - **Monolith**: ` ./start.sh --role monolith`
   - **Distributed**:
     - Server: `./start.sh --role server`
     - Client: `./start.sh --role client` (configure `RAG_REMOTE_URL` in `.env`)

2. **Configure MCPHost**: Use the configuration in `config/mcp/mcp.yaml`
3. **Connect AI Assistant**: Point your MCP-compatible AI assistant to the server

## Configuration

- [MCP Configuration](configuration.md) - Server and client setup
- [Planner Configuration](planner.md) - AI assistant behavior and policies

## Usage

- [Available Tools](tools.md) - Complete list of MCP tools
- [Integration Examples](examples.md) - Common usage patterns

## Development

- [MCP Server Implementation](server.md) - Technical details
- [Testing MCP Integration](testing.md) - Development and testing

## Related Documentation

- [Main README](../README.md) - Project overview and setup
- [REST API](../api.md) - Alternative REST interface
- [CLI Client](../cli.md) - Command-line interface