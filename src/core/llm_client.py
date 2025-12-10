"""
LLM Client module with concurrency control and error handling.
Refactored to use LangChain/LangGraph standards.
"""
import os
import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import httpx

from src.core.exceptions import ConfigurationError, AuthenticationError, ProviderError

# Import Ollama configuration
from src.core.ollama_config import (
    get_ollama_endpoint,
    get_ollama_client_headers,
)

# Load .env early
load_dotenv()

logger = logging.getLogger(__name__)

# -------- Configuration --------
# Use dynamic endpoint resolution from ollama_config
def _get_ollama_api_base() -> str:
    """Get Ollama API base URL based on current mode."""
    return get_ollama_endpoint()

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

def _convert_messages(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """Convert dict messages to LangChain messages."""
    lc_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            # Fallback for other roles?
            lc_messages.append(HumanMessage(content=content))
    return lc_messages

def _get_chat_model(
    model: str,
    api_base: str,
    temperature: float,
    timeout: int,
    **kwargs
) -> ChatOllama:
    """Create a configured ChatOllama instance."""
    # Handle model name: remove 'ollama/' prefix if present as ChatOllama likely expects just the model tag
    # or keep it if that's what user configured. Typically ChatOllama takes "llama2" etc.
    if model.startswith("ollama/"):
        model = model.replace("ollama/", "", 1)
        
    headers = get_ollama_client_headers()
    extra_headers = kwargs.pop("extra_headers", {})
    if headers:
        extra_headers.update(headers)

    return ChatOllama(
        model=model,
        base_url=api_base,
        temperature=temperature,
        timeout=timeout,
        headers=extra_headers if extra_headers else None,
        **kwargs
    )

async def safe_completion(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Execute a completion with concurrency limits and error handling.
    Returns LangChain AIMessage object (or similar).
    """
    if system_prompt:
        # Prepend system prompt if not already in messages
        if messages and messages[0].get("role") == "system":
            pass # Already has system prompt
        else:
            messages = [{"role": "system", "content": system_prompt}] + messages

    # Handle model override
    model = kwargs.pop("model", LLM_MODEL_NAME)
    api_base = kwargs.pop("api_base", _get_ollama_api_base())
    temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
    timeout = kwargs.pop("timeout", LLM_TIMEOUT)
    # stream = kwargs.pop("stream", False) # Not fully supported in this wrapper yet

    lc_messages = _convert_messages(messages)
    
    chat = _get_chat_model(model, api_base, temperature, timeout, **kwargs)

    async with _LLM_SEMAPHORE:
        try:
            # Run in executor if invoke is sync, but ChatOllama has ainvoke
            return await chat.ainvoke(lc_messages)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise AuthenticationError(
                    f"Authentication failed for LLM provider: {exc.response.text}"
                ) from exc
            else:
                raise ProviderError(
                    f"LLM provider error ({exc.response.status_code}): {exc.response.text}"
                ) from exc
        except httpx.RequestError as exc:
            raise ConfigurationError(
                f"Failed to connect to LLM provider. Check your network or 'ollamaMode' settings. Error: {str(exc)}"
            ) from exc
        except Exception as exc: 
            logger.error("LLM Completion Error: %s", exc)
            raise

async def safe_chat(
    messages: Optional[List[Dict[str, str]]] = None,
    query: Optional[List[str]] = None,
    model: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Send a chat query using LangChain with concurrency limits.
    """
    async with _LLM_SEMAPHORE:
        if messages is None:
            if query is None:
                raise ValueError("Either 'messages' or 'query' must be provided")
            messages = [{"content": str(text), "role": "user"} for text in query]

        model_name = model or ASYNC_LLM_MODEL_NAME
        api_base = kwargs.pop("api_base", _get_ollama_api_base())
        temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
        timeout = kwargs.pop("timeout", LLM_TIMEOUT)

        lc_messages = _convert_messages(messages)
        chat = _get_chat_model(model_name, api_base, temperature, timeout, **kwargs)

        try:
            return await chat.ainvoke(lc_messages)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise AuthenticationError(
                    f"Authentication failed for LLM provider: {exc.response.text}"
                ) from exc
            else:
                raise ProviderError(
                    f"LLM provider error ({exc.response.status_code}): {exc.response.text}"
                ) from exc
        except httpx.RequestError as exc:
            raise ConfigurationError(
                f"Failed to connect to LLM provider. Check your network or 'ollamaMode' settings. Error: {str(exc)}"
            ) from exc
        except Exception as exc:
            logger.error("Async LLM Chat Error: %s", exc)
            raise

def sync_completion(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Synchronous wrapper for safe_completion.
    """
    if system_prompt:
        if not (messages and messages[0].get("role") == "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages

    model = kwargs.pop("model", LLM_MODEL_NAME)
    api_base = kwargs.pop("api_base", _get_ollama_api_base())
    temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
    timeout = kwargs.pop("timeout", LLM_TIMEOUT)

    lc_messages = _convert_messages(messages)
    chat = _get_chat_model(model, api_base, temperature, timeout, **kwargs)

    with _SYNC_LLM_SEMAPHORE:
        try:
            return chat.invoke(lc_messages)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise AuthenticationError(
                    f"Authentication failed for LLM provider: {exc.response.text}"
                ) from exc
            else:
                raise ProviderError(
                    f"LLM provider error ({exc.response.status_code}): {exc.response.text}"
                ) from exc
        except httpx.RequestError as exc:
            raise ConfigurationError(
                f"Failed to connect to LLM provider. Check your network or 'ollamaMode' settings. Error: {str(exc)}"
            ) from exc
        except Exception as exc:
            logger.error("LLM Completion Error: %s", exc)
            raise
