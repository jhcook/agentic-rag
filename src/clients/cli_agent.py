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


def search(query: str, async_mode: bool = False, timeout_seconds: int = 300,
           model: str = None, temperature: float = None, max_tokens: int = None, 
           top_k: int = None):
    """Invoke the REST search endpoint. In async mode, returns a job and polls."""
    payload = {"query": query}
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_k is not None:
        payload["top_k"] = top_k
    
    if not async_mode:
        return post("/api/search", payload, timeout=timeout_seconds)

    # Async flow: request async search, then poll for completion/messages
    payload["async"] = True
    payload["timeout_seconds"] = timeout_seconds
    kick = post("/api/search", payload, timeout=30)
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
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {"error": f"failed to poll search job: {exc}"}

        if status:
            status_msg = status.get("message")
            if status_msg and status_msg != last_message:
                print(status_msg)
                last_message = status_msg

            if status.get("status") in {"completed", "timeout", "failed"}:
                job_result = status.get("result") or status
                return job_result

        time.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout exceeded
    return {
        "error": "search timeout",
        "message": last_message,
        "job_id": job_id
    }

def control_loop(q: str, idx: str | None = None, async_mode: bool = False,
                  timeout_seconds: int = 300, model: str = None, 
                  temperature: float = None, max_tokens: int = None, top_k: int = None):
    """Optionally index a path, then perform a search."""
    if idx:
        index_path(path=idx)
    return search(q, async_mode=async_mode, timeout_seconds=timeout_seconds,
                  model=model, temperature=temperature, max_tokens=max_tokens, top_k=top_k)

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
    return str(result)

def list_models():
    """List available models from the REST API."""
    try:
        r = requests.get(f"{BASE}/api/config/models", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to list models: %s", exc)
        return []

def get_backend_mode():
    """Get current backend mode from the REST API."""
    try:
        r = requests.get(f"{BASE}/api/config/mode", timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("mode"), data.get("available_modes", [])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to get backend mode: %s", exc)
        return None, []

def set_backend_mode(mode: str) -> Dict[str, Any]:
    """Set the backend mode via REST API."""
    try:
        r = requests.post(
            f"{BASE}/api/config/mode",
            json={"mode": mode},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Failed to set backend mode: %s", exc)
        return {"error": str(exc)}

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    arg_parser = argparse.ArgumentParser(
        description="CLI agent for the retrieval REST server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Basic search
  %(prog)s "What is RAG?"
  
  # Use specific Ollama model
  %(prog)s "Explain the architecture" --model llama3.2:3b
  
  # Adjust temperature for creativity
  %(prog)s "Suggest ideas" --temperature 0.9
  
  # Limit number of context documents
  %(prog)s "What features?" --top-k 10
  
  # Async mode with custom timeout
  %(prog)s "Complex query" --async --timeout 600
  
  # Backend management
  %(prog)s --show-backend              # Show current backend
  %(prog)s --list-backends             # List available backends
  %(prog)s --set-backend local         # Switch to Ollama
  %(prog)s --set-backend openai_assistants  # Switch to OpenAI
  %(prog)s --set-backend google_gemini      # Switch to Google Gemini
  %(prog)s --set-backend vertex_ai_search   # Switch to Vertex AI
  
  # Query with different backends
  %(prog)s "Fast query" --set-backend local
  %(prog)s "Complex reasoning" --set-backend openai_assistants
  %(prog)s "Search my Drive" --set-backend google_gemini
  
  # List available models (Ollama only)
  %(prog)s --list-models
""")
    arg_parser.add_argument(
        "user_input",
        nargs="?",
        help="Your search query or 'index /path/to/docs'")
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
        help="Search timeout seconds (default: 300)")
    arg_parser.add_argument(
        "--model",
        help="Override LLM model (e.g., 'qwen2.5:3b', 'llama3.2:3b'). Ollama only.")
    arg_parser.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature 0.0-1.0 (default: 0.1). Lower=factual, higher=creative")
    arg_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens in response")
    arg_parser.add_argument(
        "--top-k",
        type=int,
        help="Number of documents to retrieve (default: 5)")
    arg_parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit")
    arg_parser.add_argument(
        "--show-backend",
        action="store_true",
        help="Show current backend mode and exit")
    arg_parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backend modes and exit")
    arg_parser.add_argument(
        "--set-backend",
        metavar="MODE",
        help="Switch to specified backend (local, openai_assistants, google_gemini, vertex_ai_search)")
    return arg_parser

if __name__ == "__main__":
    main_parser = build_parser()
    args = main_parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.CRITICAL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle --list-backends
    if args.list_backends:
        mode, available = get_backend_mode()
        if available:
            print("Available backends:")
            for backend in available:
                marker = " (current)" if backend == mode else ""
                print(f"  - {backend}{marker}")
        else:
            print("Failed to get backend information")
        exit(0)

    # Handle --show-backend
    if args.show_backend:
        mode, available = get_backend_mode()
        if mode:
            print(f"Current backend: {mode}")
            if available:
                print(f"Available backends: {', '.join(available)}")
        else:
            print("Failed to get backend information")
        exit(0)

    # Handle --set-backend
    if args.set_backend:
        result = set_backend_mode(args.set_backend)
        if "error" in result:
            print(f"Error: {result['error']}")
            exit(1)
        elif result.get("status") == "ok":
            print(f"Switched to backend: {result.get('mode', args.set_backend)}")
            # If no query provided, exit after switching
            if not args.user_input:
                exit(0)
        else:
            print(f"Backend switch result: {result}")
            if not args.user_input:
                exit(0)

    # Handle --list-models
    if args.list_models:
        models = list_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models available or backend doesn't support model listing")
        exit(0)

    # Require user_input for search/index operations
    if not args.user_input:
        main_parser.print_help()
        exit(1)

    command_type, content = parse_command(args.user_input)

    if command_type == "index":
        search_result = index_path(path=content)
    else:
        search_result = control_loop(
            content,
            args.index_path,
            async_mode=args.async_mode,
            timeout_seconds=args.timeout_seconds,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k)

    if args.verbose:
        print(json.dumps(search_result, ensure_ascii=False, indent=2))
    else:
        output_msg = extract_message_content(search_result)
        if output_msg:
            print(output_msg)
