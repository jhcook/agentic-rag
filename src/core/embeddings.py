"""
Embedding utilities and device-safe loader.
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Iterable, Any, List
from pathlib import Path
import shutil
import platform
import hashlib
import threading
import numpy as np

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
_EMBEDDER_LOCK = threading.Lock()

OPENAI_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

DEFAULT_SENTENCE_TRANSFORMER_FALLBACKS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
]

# macOS-specific builds of PyTorch 2.5+ have been the most likely to throw the meta tensor error.
PLATFORM_FALLBACKS = {
    "Darwin": [
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
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


def _clean_incomplete_model_artifacts(model_name: str, logger: logging.Logger) -> bool:
    """Remove partial/incomplete download artifacts that can poison later loads."""
    hf_root = Path(os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub").expanduser()
    target_root = hf_root / f"models--{model_name.replace('/', '--')}"
    if not target_root.exists():
        return False

    removed = False
    for pattern in ("**/*.incomplete", "**/tmp_*"):
        for artifact in target_root.glob(pattern):
            try:
                artifact.unlink(missing_ok=True)  # type: ignore[arg-type]
                removed = True
            except Exception as exc:
                logger.debug("Failed to remove artifact %s: %s", artifact, exc)
    if removed:
        logger.info("Removed incomplete download artifacts for %s", model_name)
    return removed


def _pin_torch_threads():
    """Limit PyTorch thread usage to avoid oversubscription on small hosts."""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


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


class SimpleHashEmbedder:
    """
    Extremely lightweight fallback embedder that hashes tokens into a fixed-size
    bag-of-words vector. This keeps the pipeline operational when torch models
    cannot be loaded (e.g., CPU-only hosts hitting meta tensor errors).
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, inputs, **_: Any):
        if isinstance(inputs, str):
            payload = [inputs]
        else:
            payload = list(inputs)

        vectors = []
        for text in payload:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in str(text).split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                vec[bucket] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.vstack(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


def _parse_fallback_models(primary_model: str) -> List[str]:
    """Build an ordered list of fallback models based on env vars and platform defaults."""
    env_fallbacks = [
        candidate.strip()
        for candidate in os.getenv("EMBED_MODEL_FALLBACKS", "").split(",")
        if candidate.strip()
    ]
    platform_fallbacks = PLATFORM_FALLBACKS.get(platform.system(), [])

    ordered: List[str] = [primary_model]
    for candidate in env_fallbacks + platform_fallbacks + DEFAULT_SENTENCE_TRANSFORMER_FALLBACKS:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _load_sentence_transformer(model_name: str, logger: logging.Logger) -> SentenceTransformer:
    """Load a SentenceTransformer with retries that scrub stale cache artifacts."""
    last_exc: Optional[BaseException] = None
    for attempt in range(2):
        try:
            embedder = SentenceTransformer(
                model_name,
                device='cpu',
                backend='torch'
            )
            _pin_torch_threads()
            return embedder
        except NotImplementedError as exc:
            last_exc = exc
            logger.warning("Embedding load attempt %s for '%s' hit meta tensor error: %s", attempt + 1, model_name, exc)
            _clear_sentence_transformer_cache(model_name, logger)
            _clean_incomplete_model_artifacts(model_name, logger)
        except Exception as exc:
            last_exc = exc
            logger.warning("Embedding model '%s' failed to load on attempt %s: %s", model_name, attempt + 1, exc)
            _clear_sentence_transformer_cache(model_name, logger)
            _clean_incomplete_model_artifacts(model_name, logger)
            # A general failure is unlikely to succeed on an immediate retry; break early.
            break
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Unknown embedding load failure for {model_name}")


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

    with _EMBEDDER_LOCK:
        reload_needed = _EMBEDDER is None or _EMBEDDER_NAME != model_name
        if not reload_needed:
            return _EMBEDDER

        for candidate in _parse_fallback_models(model_name):
            try:
                logger.info("Loading embedding model: %s", candidate)
                if candidate in OPENAI_EMBED_DIMS:
                    api_base = os.getenv("OPENAI_API_BASE")
                    _EMBEDDER = LiteLLMEmbedder(candidate, api_base=api_base)
                    _EMBEDDER_NAME = candidate
                    return _EMBEDDER

                _EMBEDDER = _load_sentence_transformer(candidate, logger)
                _EMBEDDER_NAME = candidate
                return _EMBEDDER
            except Exception as exc:
                logger.warning("Embedding model '%s' unavailable (%s)", candidate, exc)
                continue

        if os.getenv("DISABLE_SIMPLE_EMBEDDER") not in ("1", "true", "True"):
            logger.error(
                "All embedding models failed to load; falling back to SimpleHashEmbedder (dim=%s). "
                "Set DISABLE_SIMPLE_EMBEDDER=1 to disable this behavior.", 256
            )
            _EMBEDDER = SimpleHashEmbedder()
            _EMBEDDER_NAME = "simple-hash"
            return _EMBEDDER

        raise RuntimeError(
            f"No embedding models could be loaded from candidates: {_parse_fallback_models(model_name)}"
        )
