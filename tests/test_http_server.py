import pytest
from unittest.mock import MagicMock, patch
from src.servers.mcp_server import index_documents_tool, index_url_tool, search_tool

# We need to mock the backend to avoid side effects and use temp store
@pytest.fixture(autouse=True)
def mock_backend(monkeypatch):
    mock = MagicMock()
    mock.upsert_document.return_value = {"upserted": True}
    mock.search.return_value = {"answer": "Test answer"}
    mock.list_documents.return_value = []
    monkeypatch.setattr("src.servers.mcp_server.backend", mock)
    return mock

def test_index_documents_tool(mock_backend):
    # We need to mock resolve_input_path and pathlib.Path.rglob/is_file
    with patch("src.servers.mcp_server.resolve_input_path") as mock_resolve:
        mock_path = MagicMock()
        mock_path.is_file.return_value = False
        # Mock rglob to return a list of mock paths
        mock_file = MagicMock()
        mock_file.__str__.return_value = "test_file.txt"
        mock_file.suffix = ".txt"
        mock_path.rglob.return_value = [mock_file]
        mock_resolve.return_value = mock_path
        
        with patch("src.servers.mcp_server.extract_text_from_file", return_value="content"):
            result = index_documents_tool(path="docs", glob="*.txt")
            assert "indexed" in result
            assert result["indexed"] == 1

def test_index_url_tool(mock_backend):
    with patch("src.servers.mcp_server.extract_text_from_file", return_value="content"):
        result = index_url_tool(url="http://example.com/test.txt", doc_id="test_url_doc")
        assert "indexed" in result
        assert result["indexed"] == 1

def test_search_tool(mock_backend):
    result = search_tool(query="test", top_k=5)
    assert "answer" in result

