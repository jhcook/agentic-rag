#!/usr/bin/env python3
"""
A simple agent that interacts with the retrieval server via REST API.
Run with:
    python agent.py "your question" [optional_index_path]
"""

import json, os, sys, requests, logging
from typing import Any, Dict

from urllib3.exceptions import ReadTimeoutError
from requests.exceptions import ConnectionError, HTTPError

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

BASE = os.getenv("RAG_BASE", "http://127.0.0.1:8001")

def post(path: str, payload: Dict[str, Any]):
    """POST a JSON payload to the REST API and return the parsed response."""
    try:
        r = requests.post(f"{BASE}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except (ConnectionError, ReadTimeoutError, HTTPError) as e:
        logging.error(f"HTTP error during POST to {path}: {e}")

def index_path(path: str, glob: str="**/*.txt"):
    """Invoke the REST indexer for a local path."""
    return post("/api/index_path", {"path": path, "glob": glob})

def search(query: str):
    """Invoke the REST search endpoint."""
    return post("/api/search", {"query": query})

def control_loop(q: str, idx: str | None=None):
    """Optionally index a path, then perform a search."""
    try:
        if idx:
            index_path(path=idx)
        resp = search(q)
        return resp
    except (ConnectionError, ReadTimeoutError, HTTPError) as e:
        logging.error(f"Timeout error during search: {e}")

def parse_command(user_input: str):
    """Parse user input to detect indexing commands."""
    # Check for index commands like "index docs", "index path/to/dir"
    if user_input.lower().startswith("index "):
        # Extract the path after "index "
        path = user_input[6:].strip()
        if path:
            return "index", path
    return "search", user_input

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: agent.py 'your question' [optional_index_path]", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  agent.py 'index documents'           # Index documents directory", file=sys.stderr) 
        print("  agent.py 'what is my name?'     # Search query", file=sys.stderr)
        print("  agent.py 'search query' documents    # Search + also index documents", file=sys.stderr)
        sys.exit(2)

    user_input = sys.argv[1]
    explicit_index_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Parse the command
    command_type, content = parse_command(user_input)

    if command_type == "index":
        # Just index, don't search
        result = index_path(path=content)
        print(json.dumps(result))
    else:
        # Search (and optionally index first)
        result = control_loop(content, explicit_index_path)
        print(json.dumps(result))
