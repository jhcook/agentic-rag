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
    r = requests.post(f"{BASE}{path}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def index_path(path: str, glob: str="**/*.txt"):
    return post("/api/index_path", {"path": path, "glob": glob})

def search(query: str):
    return post("/api/search", {"query": query})

def control_loop(q: str, idx: str | None=None):
    try:
        if idx:
            index_path(path=idx)
        resp = search(q)
        return resp
    except (ConnectionError, ReadTimeoutError, HTTPError) as e:
        logging.error(f"Timeout error during search: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: agent.py 'your question' [optional_index_path]", file=sys.stderr)
        sys.exit(2)
    q = sys.argv[1]
    idx = sys.argv[2] if len(sys.argv) > 2 else None
    res = control_loop(q, idx)
    print(json.dumps(res))
