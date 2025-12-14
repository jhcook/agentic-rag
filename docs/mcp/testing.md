# Testing MCP Integration

This document covers testing strategies and procedures for the MCP integration.

## Testing Overview

The MCP integration requires testing at multiple levels:

1. **Unit Tests**: Individual tool functionality
2. **Integration Tests**: MCP server with RAG core
3. **End-to-End Tests**: Complete MCP workflows
4. **MCPHost Integration**: Testing with actual MCP clients

## Unit Testing

### Tool Testing

Each MCP tool has corresponding unit tests in `tests/test_http_server.py`:

```python
def test_index_documents_tool():
    # Test document indexing functionality
    result = index_documents_tool(path="test_docs", glob="**/*.txt")
    assert result["indexed"] > 0
    assert "uris" in result

def test_search_tool():
    # Test search functionality
    result = search_tool(query="test query")
    assert "answer" in result
    assert "model" in result
```

### Mock Testing

For testing without full dependencies:

```python
from unittest.mock import patch, MagicMock

@patch('src.core.rag_core.search')
def test_search_tool_with_mock(mock_search):
    mock_search.return_value = {"answer": "mocked answer"}
    result = search_tool(query="test")
    assert result["answer"] == "mocked answer"
```

## Integration Testing

### Server Startup Tests

```python
def test_mcp_server_startup():
    # Test that server starts without errors
    import subprocess
    process = subprocess.Popen([
        "python", "src/servers/mcp_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for startup
    time.sleep(2)

    # Check if process is still running
    assert process.poll() is None

    # Cleanup
    process.terminate()
```

### Tool Chain Testing

Test complete workflows:

```python
def test_index_and_search_workflow():
    # Index documents
    index_result = index_documents_tool(path="test_docs")
    assert index_result["indexed"] > 0

    # Search indexed content
    search_result = search_tool(query="test content")
    assert "answer" in search_result

    # Verify grounding
    verify_result = verify_grounding_tool(
        question="test question",
        answer=search_result["answer"],
        citations=[]
    )
    assert verify_result["confidence"] > 0.5
```

## End-to-End Testing

### MCP Protocol Testing

Test actual MCP protocol communication:

```python
import requests

def test_mcp_protocol():
    base_url = "http://127.0.0.1:8000"

    # Test tool discovery
    response = requests.get(f"{base_url}/mcp/tools")
    assert response.status_code == 200
    tools = response.json()
    assert "search_tool" in [t["name"] for t in tools]

    # Test tool execution
    response = requests.post(f"{base_url}/mcp/tools/search_tool", json={
        "query": "test query"
    })
    assert response.status_code == 200
    result = response.json()
    assert "answer" in result
```

### MCPHost Integration Testing

Test with actual MCPHost:

```python
def test_mcphost_integration():
    # This requires MCPHost to be running
    # Use the configuration from config/mcp/mcp.yaml

    # Test through MCPHost API
    # (Implementation depends on MCPHost testing interface)
    pass
```

## Performance Testing

### Memory Usage Testing

```python
def test_memory_usage():
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024

    # Perform memory-intensive operation
    index_documents_tool(path="large_docs")

    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory

    # Assert memory usage is within limits
    assert memory_increase < 500  # MB
```

### Response Time Testing

```python
import time

def test_search_performance():
    start_time = time.time()

    result = search_tool(query="performance test query")

    end_time = time.time()
    response_time = end_time - start_time

    # Assert reasonable response time
    assert response_time < 10.0  # seconds
```

## Load Testing

### Concurrent Requests

```python
import threading
import queue

def test_concurrent_requests():
    results = queue.Queue()

    def worker(query):
        result = search_tool(query=query)
        results.put(result)

    # Start multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=[f"query {i}"])
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Verify all results
    assert results.qsize() == 5
```

## Error Testing

### Network Failure Testing

```python
@patch('requests.get')
def test_url_indexing_network_error(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError

    result = index_url_tool(url="http://invalid.url")
    assert "error" in result
    assert result["indexed"] == 0
```

### File System Error Testing

```python
def test_missing_directory():
    result = index_documents_tool(path="/nonexistent/path")
    assert "error" in result
    assert "not found" in result["error"]
```

### LLM Failure Testing

```python
@patch('src.core.rag_core.search')
def test_llm_failure(mock_search):
    mock_search.side_effect = Exception("LLM unavailable")

    result = search_tool(query="test")
    assert "error" in result
```

## Test Data Management

### Test Document Setup

```python
import tempfile
import pathlib

@pytest.fixture
def test_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test documents
        doc1 = pathlib.Path(tmpdir) / "test1.txt"
        doc1.write_text("This is test content for document 1")

        doc2 = pathlib.Path(tmpdir) / "test2.txt"
        doc2.write_text("This is test content for document 2")

        yield tmpdir
```

### Store Cleanup

```python
@pytest.fixture(autouse=True)
def clean_store():
    # Clear indexed artifacts + pgvector rows before each test
    from src.core.rag_core import flush_cache
    flush_cache()
    yield
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test-mcp.yml
name: Test MCP Integration
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run MCP tests
      run: pytest tests/ -k "mcp" -v
    - name: Test MCP server startup
      run: |
        python src/servers/mcp_server.py &
        sleep 5
        curl -f http://localhost:8000/mcp/tools || exit 1
```

## Debugging Failed Tests

### Log Analysis

```bash
# Check MCP server logs
tail -f log/mcp_server.log

# Check test output
pytest tests/test_http_server.py -v -s
```

### Common Issues

1. **Port conflicts**: Ensure MCP port (8000) is available
2. **Memory issues**: Check system has sufficient RAM
3. **File permissions**: Ensure test directories are writable
4. **Model availability**: Verify Ollama models are downloaded

### Debugging Tools

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
from src.servers.mcp_server import search_tool
result = search_tool(query="debug test")
print(result)
```

## Test Coverage Goals

- **Unit Tests**: 90%+ coverage for individual tools
- **Integration Tests**: All tool combinations tested
- **Error Conditions**: All error paths covered
- **Performance Tests**: Response time and memory usage benchmarks
- **Compatibility Tests**: Multiple MCP client versions