#!/usr/bin/env python3
"""
A simple agent that interacts with the retrieval server via REST API.
Run with:
    python cli_agent.py "your question" [optional_index_path] [--verbose]
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict

import requests
from requests.exceptions import HTTPError
from urllib3.exceptions import ReadTimeoutError

logger = logging.getLogger(__name__)

BASE = os.getenv("RAG_BASE", "http://127.0.0.1:8001")

def post(path: str, payload: Dict[str, Any], timeout: int = 60):
    """POST a JSON payload to the REST API and return the parsed response."""
    try:
        r = requests.post(f"{BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (requests.exceptions.ConnectionError, ReadTimeoutError,
            HTTPError) as exc:
        logger.error("HTTP error during POST to %s: %s", path, exc)
        return {"error": str(exc)}

def index_path(path: str, glob: str = "**/*.txt"):
    """Invoke the REST indexer for a local path."""
    return post("/api/index_path", {"path": path, "glob": glob})


def search(query: str, async_mode: bool = False, timeout_seconds: int = 300):
    """Invoke the REST search endpoint. In async mode, returns a job and polls."""
    if not async_mode:
        return post("/api/search", {"query": query}, timeout=timeout_seconds)

    # Async flow: request async search, then poll for completion/messages
    kick = post(
        "/api/search",
        {"query": query, "async": True, "timeout_seconds": timeout_seconds},
        timeout=30)
    job_id = kick.get("job_id")
    if not job_id:
        return kick

    elapsed = 0
    poll_interval = 3
    deadline = timeout_seconds
    last_message = None

    while elapsed < deadline:
        try:
            r = requests.get(
                f"{BASE}/api/search/jobs/{job_id}", timeout=30)
            r.raise_for_status()
            status = r.json()
        except ReadTimeoutError:
            # treat poll timeout as a heartbeat and continue
            status = None
        except Exception as exc:
            return {"error": f"failed to poll search job: {exc}"}

        if status:
            status_msg = status.get("message")
            if status_msg and status_msg != last_message:
                print(status_msg)
                last_message = status_msg

            if status.get("status") in {"completed", "timeout", "failed"}:
                search_result = status.get("result") or status
                return search_result

        time.sleep(poll_interval)
        elapsed += poll_interval

    return {
        "error": "search timeout",
        "message": last_message,
        "job_id": job_id
    }

def control_loop(q: str, idx: str | None = None, async_mode: bool = False,
                  timeout_seconds: int = 300):
    """Optionally index a path, then perform a search."""
    if idx:
        index_path(path=idx)
    return search(q, async_mode=async_mode, timeout_seconds=timeout_seconds)

def parse_command(user_input: str):
    """Parse user input to detect indexing commands."""
    if user_input.lower().startswith("index "):
        path = user_input[6:].strip()
        if path:
            return "index", path
    return "search", user_input

def extract_message_content(result: Any) -> str:
    """Extract a friendly message string from a completion-like response plus sources."""
    if result is None:
        return ""
    sources = []
    if isinstance(result, dict):
        srcs = result.get("sources") or result.get("citations") or []
        if isinstance(srcs, list):
            sources = [str(s) for s in srcs if s]
        if "error" in result:
            base = f"error: {result.get('error')}"
        else:
            base = ""
            choices = result.get("choices")
            if choices and isinstance(choices, list):
                first = choices[0] or {}
                msg_dict = first.get("message", {}) if isinstance(first, dict) else {}
                msg_content = msg_dict.get("content") if isinstance(msg_dict, dict) else None
                if msg_content:
                    base = str(msg_content).strip()
                elif "text" in first:
                    base = str(first.get("text", "")).strip()
            if not base and "answer" in result:
                base = str(result.get("answer", "")).strip()
        parts = [p for p in [base] if p]
        if sources:
            parts.append("Sources:")
            parts.extend([f"- {s}" for s in sources])
        return "\n".join(parts)
    return json.dumps(result)

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    arg_parser = argparse.ArgumentParser(
        description="CLI agent for the retrieval REST server.")
    arg_parser.add_argument(
        "user_input",
        help="Query text or 'index <path>' command")
    arg_parser.add_argument(
        "index_path",
        nargs="?",
        help="Optional path to index before search")
    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full JSON responses and verbose logs")
    arg_parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async search with polling/heartbeats")
    arg_parser.add_argument(
        "--timeout",
        dest="timeout_seconds",
        type=int,
        default=300,
        help="Search timeout seconds (async and sync)")
    return arg_parser

if __name__ == "__main__":
    main_parser = build_parser()
    args = main_parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.CRITICAL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    command_type, content = parse_command(args.user_input)

    if command_type == "index":
        search_result = index_path(path=content)
    else:
        search_result = control_loop(
            content,
            args.index_path,
            async_mode=args.async_mode,
            timeout_seconds=args.timeout_seconds)

    if args.verbose:
        print(json.dumps(search_result, ensure_ascii=False))
    else:
        output_msg = extract_message_content(search_result)
        if output_msg:
            print(output_msg)
