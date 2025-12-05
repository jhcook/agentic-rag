# Agentic RAG Documentation

This directory contains project documentation, guides, and specifications.

## Directory Structure

- `docs/` - Project documentation and guides
- `config/mcp/` - MCP configuration files

## Documentation Index

### Getting Started
- [Main README](../README.md) - Project overview, installation, and setup
- [Quick Start](../README.md#quick-start) - Fast setup guide
- [Models and Configuration](models-and-configuration.md) - AI models and their integration

### AI Backend Options
- [OpenAI Assistants](openai_assistants.md) - **NEW!** GPT-4 with local document search
- [OpenAI Assistants Capabilities](openai_assistants_capabilities.md) - Technical analysis and comparison
- [Ollama Cloud Integration](ollama-cloud-architecture.md) - Cloud-hosted Ollama models with local fallback
- [Ollama Cloud GDPR Compliance](ollama-cloud-gdpr.md) - Privacy and data protection documentation
- [Cloud Provider Comparison](cloud_provider_comparison.md) - Google vs Azure vs OpenAI comparison
- [Google Integration](google_integration.md) - Setup for Google Drive and Gemini
- [Vertex AI Setup](vertex_ai_setup.md) - Enterprise Vertex AI Agent Builder configuration

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
- [REST API Reference](api_reference.md) - RESTful API documentation and OpenAPI specs
- [CLI Client](../cli.md) - Command-line interface guide

## Contributing

When adding documentation:

1. **General Documentation**: Place in this `docs/` directory
2. **MCP-Specific**: Place in `docs/mcp/` subdirectory
3. **Configuration Files**: Place in `config/` directory

Use lowercase filenames with `.md` extension and link documents together for easy navigation.