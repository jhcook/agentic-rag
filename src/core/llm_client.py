"""
LLM Client module with concurrency control and error handling.
Refactored to use LangChain/LangGraph standards.
"""
import os
import logging
import asyncio
import threading
import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
# from langchain_ollama import ChatOllama  <-- Removed
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import httpx

from src.core.exceptions import ConfigurationError, AuthenticationError, ProviderError

# Import Ollama configuration
from src.core.ollama_config import (
    get_ollama_endpoint,
    get_ollama_client_headers,
    get_ollama_api_key,
    get_ollama_mode,
    get_ollama_cloud_proxy,
    get_requests_ca_bundle,
)
from src.core.config_paths import get_ca_bundle_path

# Load .env early
load_dotenv()

# Apply CA Bundle to environment for libraries that respect SSL_CERT_FILE (like httpx)
def reload_llm_config():
    """Reload LLM configuration, including CA bundle and API key."""
    _ca_bundle = get_ca_bundle_path()
    if _ca_bundle:
        os.environ["SSL_CERT_FILE"] = _ca_bundle
        os.environ["REQUESTS_CA_BUNDLE"] = _ca_bundle
    
    # Set Ollama API key as environment variable for ollama library
    _api_key = get_ollama_api_key()
    if _api_key:
        os.environ["OLLAMA_API_KEY"] = _api_key

reload_llm_config()

logger = logging.getLogger(__name__)


def _get_settings_model_name() -> Optional[str]:
    """Return the configured model from config/settings.json if present."""
    try:
        from src.core.config_paths import SETTINGS_PATH

        if not SETTINGS_PATH.exists():
            return None

        # Local import to avoid adding a global dependency on json.
        import json  # pylint: disable=import-outside-toplevel

        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)

        model = settings.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    return None


def _get_default_model_name() -> str:
    """Resolve default model name with correct precedence (settings > env)."""
    return _get_settings_model_name() or os.getenv("LLM_MODEL_NAME", "ollama/llama3.2:1b")


def _get_default_async_model_name() -> str:
    """Resolve default async model name with correct precedence (settings > env)."""
    # If settings defines a model, use it for both sync and async.
    configured = _get_settings_model_name()
    if configured:
        return configured
    return os.getenv("ASYNC_LLM_MODEL_NAME", "llama3.2:1b")

def _redact_text(text: str) -> str:
    """Redact API key from text."""
    api_key = get_ollama_api_key()
    if not api_key or not text:
        return text
    return text.replace(api_key, "***REDACTED***")

# -------- Configuration --------
# Use dynamic endpoint resolution from ollama_config
def _get_ollama_api_base() -> str:
    """Get Ollama API base URL based on current mode."""
    return get_ollama_endpoint()

# Defaults are resolved dynamically (settings > env) to support runtime config changes.
LLM_MODEL_NAME = _get_default_model_name()
ASYNC_LLM_MODEL_NAME = _get_default_async_model_name()
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
        "model": _get_default_model_name(),
        "async_model": _get_default_async_model_name(),
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

    # Use LiteLLM Model Factory
    from langchain_community.chat_models import ChatLiteLLM

    # LiteLLM Configuration logic
    # Ensure api_base is passed as 'api_base' (or base_url depending on version, but api_base is standard for OpenAI-compat)
    # AND allow passing specific LiteLLM params like 'proxy' via kwargs or environment.

    # Fix model name for Ollama: LiteLLM expects "ollama/modelname" to know it's Ollama.
    # If the user configured just "llama3", we might need to prepend "ollama/" if we know the mode is local.
    # However, to be "provider agnostic", we should ideally respect the configured model string directly.
    # BUT, existing config (settings.json) might use plain "llama3.2:1b" while implying Ollama.
    # Let's check the provider/mode.
    
    mode = None
    try:
        mode = get_ollama_mode()
    except Exception:
        mode = "local"

    final_model = model
    # Heuristic: If we are in "local" mode and the model doesn't have a provider prefix (like 'gpt-', 'claude-', 'ollama/'), 
    # assume it's an Ollama model and prepend 'ollama/' so LiteLLM handles it correctly.
    if mode == "local" and "/" not in final_model and not final_model.startswith(("gpt-", "claude-", "gemini-")):
        final_model = f"ollama/{final_model}"
        logger.debug(f"Auto-prefixed model '{model}' with 'ollama/' for LiteLLM local mode")

    client_kwargs = {}
    if proxy_url and is_https:
        client_kwargs["proxy"] = proxy_url

    # LiteLLM needs 'api_base' for Ollama/OpenAI-compatible endpoints
    # It prioritizes 'api_base' over environment variables if passed explicitly.
    # Note: For real OpenAI, we shouldn't pass our local Ollama API base.
    # Only pass api_base if it is explicitly an Ollama endpoint or a custom proxy.
    active_api_base = None
    if final_model.startswith("ollama/"):
        active_api_base = api_base
    elif "openai/" in final_model and "localhost" in api_base:
         # If using a local OpenAI-compatible proxy
         active_api_base = api_base
    
    return ChatLiteLLM(
        model=final_model,
        api_base=active_api_base, # Pass None if standard OpenAI/Anthropic to let env vars handle it
        temperature=temperature,
        timeout=timeout,
        model_kwargs=extra_headers if extra_headers else {},
        **kwargs # Pass other kwargs like 'proxy' if supported or via client_kwargs
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
    model = kwargs.pop("model", _get_default_model_name())
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
                error_detail = exc.response.text or "No response body"
                # Redact API key
                error_detail = _redact_text(error_detail)
                logger.error(f"Authentication failed for LLM provider ({exc.response.status_code}): {error_detail}")
                raise AuthenticationError(
                    f"Authentication failed for LLM provider ({exc.response.status_code}): {error_detail}"
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

        model_name = model or _get_default_async_model_name()
        api_base = kwargs.pop("api_base", _get_ollama_api_base())
        temperature = kwargs.pop("temperature", LLM_TEMPERATURE)
        timeout = kwargs.pop("timeout", LLM_TIMEOUT)

        lc_messages = _convert_messages(messages)
        chat = _get_chat_model(model_name, api_base, temperature, timeout, **kwargs)

        try:
            return await chat.ainvoke(lc_messages)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                error_detail = exc.response.text or "No response body"
                # Redact API key
                error_detail = _redact_text(error_detail)
                logger.error(f"Authentication failed for LLM provider ({exc.response.status_code}): {error_detail}")
                raise AuthenticationError(
                    f"Authentication failed for LLM provider ({exc.response.status_code}): {error_detail}"
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
                error_detail = exc.response.text or "No response body"
                # Redact API key
                error_detail = _redact_text(error_detail)
                logger.error(f"Authentication failed for LLM provider ({exc.response.status_code}): {error_detail}")
                raise AuthenticationError(
                    f"Authentication failed for LLM provider ({exc.response.status_code}): {error_detail}"
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
