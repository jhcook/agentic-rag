# Agentic RAG Documentation

This directory contains project documentation, guides, and specifications.

## Directory Structure

- `docs/` - Project documentation and guides
- `documents/` - Indexable source documents (PDF, DOCX, TXT, etc.)
- `config/mcp/` - MCP configuration files

## Documentation Index

### Getting Started
- [Main README](../README.md) - Project overview, installation, and setup
- [Quick Start](../README.md#quick-start) - Fast setup guide
- [Models and Configuration](models-and-configuration.md) - AI models and their integration

### MCP Integration
- [MCP Overview](mcp/index.md) - Model Context Protocol integration
- [MCP Configuration](mcp/configuration.md) - Server and client setup
- [MCP Tools](mcp/tools.md) - Available tools reference
- [MCP Examples](mcp/examples.md) - Usage examples and patterns
- [MCP Planner](mcp/planner.md) - AI assistant behavior configuration

### Development
- [MCP Server Implementation](mcp/server.md) - Technical implementation details
- [Testing MCP Integration](mcp/testing.md) - Testing strategies and procedures

### APIs and Interfaces
- [REST API](../api.md) - RESTful API documentation
- [CLI Client](../cli.md) - Command-line interface guide

## Contributing

When adding documentation:

1. **General Documentation**: Place in this `docs/` directory
2. **MCP-Specific**: Place in `docs/mcp/` subdirectory
3. **Configuration Files**: Place in `config/` directory
4. **Indexable Content**: Place in `documents/` directory

Use lowercase filenames with `.md` extension and link documents together for easy navigation.