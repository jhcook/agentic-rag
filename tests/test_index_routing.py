from src.servers import mcp_server


def test_extract_url_from_query():
    query = "please index https://example.com/doc.pdf asap"
    assert mcp_server._extract_url_from_query(query) == "https://example.com/doc.pdf"  # pylint: disable=protected-access


def test_index_url_tool_routes_directory(monkeypatch):
    captured = {}

    def fake_index_documents_tool(path: str, glob: str = "**/*"):
        captured["path"] = path
        captured["glob"] = glob
        return {"indexed": 3, "uris": ["dummy"]}

    monkeypatch.setattr(mcp_server, "index_documents_tool", fake_index_documents_tool)

    result = mcp_server.index_url_tool(query="index ./docs")

    assert result["indexed"] == 3
    assert captured["path"] == "./docs"
    assert captured["glob"] == "**/*"
