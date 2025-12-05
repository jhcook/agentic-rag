"""
LLM Client module with concurrency control and error handling.
"""
import os
import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from ollama import AsyncClient  # type: ignore

# Import Ollama configuration
from src.core.ollama_config import (
    get_ollama_endpoint,
    get_ollama_client_headers,
)

# Load .env early
load_dotenv()

# Enable debug logging if available
# try:
#     import litellm  # type: ignore
#     # os.environ['LITELLM_LOG'] = 'DEBUG'
# except (ImportError, AttributeError):
#     pass

try:
    from litellm import completion  # type: ignore
    from litellm.exceptions import APIConnectionError, Timeout  # type: ignore
    from litellm.llms.ollama.common_utils import OllamaError  # type: ignore
except ImportError:
    completion = None
    APIConnectionError = Exception
    Timeout = Exception
    OllamaError = Exception

logger = logging.getLogger(__name__)

# -------- Configuration --------
# Use dynamic endpoint resolution from ollama_config
def _get_ollama_api_base() -> str:
    """Get Ollama API base URL based on current mode."""
    return get_ollama_endpoint()

OLLAMA_API_BASE = _get_ollama_api_base()  # For backward compatibility
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")
ASYNC_LLM_MODEL_NAME = os.getenv("ASYNC_LLM_MODEL_NAME", "llama3.2:1b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "1"))

# Global semaphore for concurrency control
_LLM_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
_SYNC_LLM_SEMAPHORE = threading.Semaphore(LLM_MAX_CONCURRENCY)

def get_llm_config() -> Dict[str, Any]:
    """Return current LLM configuration."""
    return {
        "api_base": _get_ollama_api_base(),
        "model": LLM_MODEL_NAME,
        "async_model": ASYNC_LLM_MODEL_NAME,
        "temperature": LLM_TEMPERATURE,
        "timeout": LLM_TIMEOUT,
        "max_concurrency": LLM_MAX_CONCURRENCY,
    }

async def safe_completion(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Execute a completion with concurrency limits and error handling.
    This is an async wrapper around the synchronous litellm.completion (or async if supported).
    """
    if completion is None:
        raise RuntimeError("LiteLLM not available")

    if system_prompt:
        # Prepend system prompt if not already in messages
        if messages and messages[0].get("role") == "system":
            pass # Already has system prompt
        else:
            messages = [{"role": "system", "content": system_prompt}] + messages

    # Handle model override
    model = kwargs.pop("model", LLM_MODEL_NAME)

    # Handle other overrides or defaults
    api_base = kwargs.pop("api_base", _get_ollama_api_base())
    temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
    timeout = kwargs.pop("timeout", LLM_TIMEOUT)
    stream = kwargs.pop("stream", False)
    
    # Get headers for cloud authentication
    headers = get_ollama_client_headers()
    extra_headers = kwargs.pop("extra_headers", {})
    if headers:
        extra_headers.update(headers)

    async with _LLM_SEMAPHORE:
        try:
            # Run blocking completion in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()

            def _call():
                return completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    temperature=temperature,
                    timeout=timeout,
                    stream=stream,
                    extra_headers=extra_headers if extra_headers else None,
                    **kwargs
                )

            return await loop.run_in_executor(None, _call)

        except (ValueError, OllamaError, APIConnectionError, Timeout) as exc:
            logger.error("LLM Completion Error: %s", exc)
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Unexpected error in LLM completion: %s", exc)
            raise

async def safe_chat(
    messages: Optional[List[Dict[str, str]]] = None,
    query: Optional[List[str]] = None,
    model: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Send a chat query using the AsyncClient with concurrency limits.
    Supports either raw 'query' list (converted to user messages) or structured 'messages'.
    """
    async with _LLM_SEMAPHORE:
        endpoint = _get_ollama_api_base()
        headers = get_ollama_client_headers()
        client = AsyncClient(host=endpoint, headers=headers if headers else None)

        if messages is None:
            if query is None:
                raise ValueError("Either 'messages' or 'query' must be provided")
            messages = [{"content": str(text), "role": "user"} for text in query]

        target_model = model or ASYNC_LLM_MODEL_NAME

        try:
            resp = await client.chat(
                model=target_model,
                messages=messages,
                **kwargs
            )
            return resp
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Async LLM Chat Error: %s", exc)
            raise

def sync_completion(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Synchronous wrapper for safe_completion.
    Uses a threading semaphore for concurrency control to avoid asyncio loop issues.
    """
    if completion is None:
        raise RuntimeError("LiteLLM not available")

    if system_prompt:
        if not (messages and messages[0].get("role") == "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages

    # Handle model override
    model = kwargs.pop("model", LLM_MODEL_NAME)

    # Handle other overrides or defaults
    api_base = kwargs.pop("api_base", _get_ollama_api_base())
    temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
    timeout = kwargs.pop("timeout", LLM_TIMEOUT)
    stream = kwargs.pop("stream", False)
    
    # Get headers for cloud authentication
    headers = get_ollama_client_headers()
    extra_headers = kwargs.pop("extra_headers", {})
    if headers:
        extra_headers.update(headers)

    # Use threading semaphore for sync calls
    with _SYNC_LLM_SEMAPHORE:
        try:
            return completion(
                model=model,
                messages=messages,
                api_base=api_base,
                temperature=temperature,
                timeout=timeout,
                stream=stream,
                extra_headers=extra_headers if extra_headers else None,
                **kwargs
            )
        except (ValueError, OllamaError, APIConnectionError, Timeout) as exc:
            logger.error("LLM Completion Error: %s", exc)
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Unexpected error in LLM completion: %s", exc)
            raise
