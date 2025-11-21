"""
Embedding utilities and device-safe loader.
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Iterable
from pathlib import Path
import shutil

from sentence_transformers import SentenceTransformer  # type: ignore

try:
    from litellm import embedding as litellm_embedding  # type: ignore
except ImportError:
    litellm_embedding = None

# Torch/env safety knobs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_EMBEDDER: Optional[SentenceTransformer] = None
_EMBEDDER_NAME: Optional[str] = None

OPENAI_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _clear_sentence_transformer_cache(model_name: str, logger: logging.Logger) -> bool:
    """Delete cached model artifacts for a given SentenceTransformer model."""
    cleared = False
    sanitized = model_name.replace("/", "_")
    hf_sanitized = model_name.replace("/", "--")

    def _torch_cache_paths() -> Iterable[Path]:
        root_env = os.getenv("SENTENCE_TRANSFORMERS_HOME")
        if root_env:
            root = Path(root_env).expanduser()
        else:
            torch_home = Path(os.getenv("TORCH_HOME") or Path.home() / ".cache" / "torch").expanduser()
            root = torch_home / "sentence_transformers"
        if root.exists():
            yield root / sanitized
            yield from root.glob(f"*{sanitized}*")

    def _hf_cache_paths() -> Iterable[Path]:
        hf_root = Path(os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub").expanduser()
        if hf_root.exists():
            yield hf_root / f"models--{hf_sanitized}"
            yield from hf_root.glob(f"models--*{hf_sanitized}*")

    seen = set()
    for path in list(_torch_cache_paths()) + list(_hf_cache_paths()):
        if not path or not path.exists():
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            shutil.rmtree(path)
            logger.info("Cleared cached model files at %s", path)
            cleared = True
        except Exception as exc:
            logger.warning("Failed to clear cached model files at %s: %s", path, exc)
    return cleared


class LiteLLMEmbedder:
    """Minimal adapter to expose a SentenceTransformer-like interface for LiteLLM embeddings."""

    def __init__(self, model_name: str, api_base: Optional[str] = None):
        if litellm_embedding is None:
            raise ImportError("litellm is required for OpenAI embedding models")
        self.model_name = model_name
        self.api_base = api_base
        self._dim = OPENAI_EMBED_DIMS.get(model_name, 1536)

    def encode(self, inputs, **_: Any):
        if litellm_embedding is None:
            raise ImportError("litellm is required for OpenAI embedding models")
        if isinstance(inputs, str):
            payload = [inputs]
        else:
            payload = list(inputs)
        resp = litellm_embedding(model=self.model_name, input=payload, api_base=self.api_base)
        # litellm returns {"data": [{"embedding": ..., "index": 0}, ...]}
        return [item["embedding"] for item in getattr(resp, "data", resp.get("data", []))]  # type: ignore[union-attr]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


def get_embedder(model_name: str, debug_mode: bool, logger: logging.Logger) -> Optional[object]:
    """
    Lazily load and cache the embedding model with CPU-first defaults.
    Respects debug_mode by skipping model loading.
    """
    global _EMBEDDER, _EMBEDDER_NAME

    if debug_mode:
        logger.info("Debug mode enabled - skipping embedding model loading")
        return None

    reload_needed = _EMBEDDER is None or _EMBEDDER_NAME != model_name
    if not reload_needed:
        return _EMBEDDER

    try:
        logger.info("Loading embedding model: %s", model_name)
        if model_name in OPENAI_EMBED_DIMS:
            api_base = os.getenv("OPENAI_API_BASE")
            _EMBEDDER = LiteLLMEmbedder(model_name, api_base=api_base)
            _EMBEDDER_NAME = model_name
            return _EMBEDDER

        try:
            import torch  # Local import so unit tests can swap it out
            torch.set_num_threads(1)
        except Exception:
            pass
        
        # Try loading with default settings first
        try:
            _EMBEDDER = SentenceTransformer(
                model_name,
                device='cpu',
                backend='torch',
                model_kwargs={"trust_remote_code": True}
            )
            _EMBEDDER_NAME = model_name
        except Exception as exc:
            logger.warning("Initial load of '%s' failed: %s. Retrying with cache clear and CPU force.", model_name, exc)
            _clear_sentence_transformer_cache(model_name, logger)
            _EMBEDDER = SentenceTransformer(
                model_name,
                device='cpu',
                backend='torch',
                model_kwargs={"trust_remote_code": True}
            )
            _EMBEDDER_NAME = model_name

        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
    except Exception as exc:
        # Catch ALL errors during load (NotImplementedError, OSError, etc)
        logger.warning("Retrying embedder load on CPU due to error: %s", exc)
        try:
            _clear_sentence_transformer_cache(model_name, logger)
            _EMBEDDER = SentenceTransformer(
                model_name,
                device='cpu',
                backend='torch',
                model_kwargs={"trust_remote_code": True}
            )
            _EMBEDDER_NAME = model_name
        except Exception as inner_exc:
            logger.warning("User embedding model '%s' failed after CPU retry: %s; falling back", model_name, inner_exc)
            fallback = "all-MiniLM-L6-v2"
            _clear_sentence_transformer_cache(fallback, logger)
            _EMBEDDER = SentenceTransformer(
                fallback,
                device='cpu',
                backend='torch'
            )
            _EMBEDDER_NAME = fallback
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
    except Exception as exc:
        # Prefer to keep running with a smaller, local model rather than crash on meta-tensor/device errors
        fallback = "all-MiniLM-L6-v2"
        logger.warning(
            "Embedding model '%s' failed to load (%s). Falling back to '%s'.",
            model_name, exc, fallback
        )
        try:
            _clear_sentence_transformer_cache(model_name, logger)
            _EMBEDDER = SentenceTransformer(fallback, device='cpu', backend='torch')
            _EMBEDDER_NAME = fallback
            import torch
            torch.set_num_threads(1)
            logger.info("Fallback embedding model '%s' loaded successfully", fallback)
        except Exception as inner_exc:
            logger.critical(
                "Fallback embedding model '%s' also failed to load: %s", fallback, inner_exc
            )
            raise

    return _EMBEDDER
