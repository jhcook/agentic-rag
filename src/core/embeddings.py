"""
Embedding utilities and device-safe loader.
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Iterable, Any
from pathlib import Path
import shutil

from sentence_transformers import SentenceTransformer  # type: ignore

# Torch/env safety knobs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Prefer a local cache bundled with the app so offline installs work
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = Path(os.getenv("RAG_CACHE_DIR", _REPO_ROOT / "cache")).expanduser()
_HF_HOME = Path(os.getenv("HF_HOME", _CACHE_ROOT / "huggingface")).expanduser()
_ST_HOME = Path(os.getenv("SENTENCE_TRANSFORMERS_HOME", _CACHE_ROOT / "sentence_transformers")).expanduser()

for _p in (_CACHE_ROOT, _HF_HOME, _ST_HOME):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_HOME / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_ST_HOME))

_EMBEDDER: Optional[SentenceTransformer] = None  # pylint: disable=invalid-name
_EMBEDDER_NAME: Optional[str] = None  # pylint: disable=invalid-name

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
            torch_home = Path(
                os.getenv("TORCH_HOME") or Path.home() / ".cache" / "torch"
            ).expanduser()
            root = torch_home / "sentence_transformers"
        if root.exists():
            yield root / sanitized
            yield from root.glob(f"*{sanitized}*")

    def _hf_cache_paths() -> Iterable[Path]:
        hf_root = Path(
            os.getenv("HF_HOME") or
            os.getenv("HUGGINGFACE_HUB_CACHE") or
            Path.home() / ".cache" / "huggingface" / "hub"
        ).expanduser()
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
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to clear cached model files at %s: %s", path, exc)
    return cleared

def _is_meta_tensor_error(exc: Exception) -> bool:
    """Check if exception is related to meta tensor device issues."""
    error_msg = str(exc).lower()
    return any(keyword in error_msg for keyword in [
        "meta tensor",
        "cannot copy out of meta",
        "to_empty()",
        "meta device",
        "no data",
    ])


def _is_device_error(exc: Exception) -> bool:
    """Check if exception is related to device placement issues."""
    error_msg = str(exc).lower()
    return any(keyword in error_msg for keyword in [
        "device",
        "cuda",
        "mps",
        "cpu",
        "expected all tensors",
    ])


def _load_with_strategy(
    model_name: str,
    strategy: dict,
    logger: logging.Logger,
) -> SentenceTransformer:
    """Load model with specific strategy parameters."""
    logger.info("Attempting load with strategy: %s", strategy.get("name", "unknown"))
    return SentenceTransformer(
        model_name,
        device=strategy.get("device", "cpu"),
        backend=strategy.get("backend", "torch"),
        model_kwargs=strategy.get("model_kwargs", {})
    )


def get_embedder(
    model_name: str,
    debug_mode: bool,
    logger: logging.Logger,
) -> Optional[object]:  # pylint: disable=too-many-statements
    """
    Lazily load and cache the embedding model with automatic mitigation strategies.

    Implements progressive fallback strategies to handle:
    - Meta tensor device errors
    - CUDA/MPS device issues
    - Cache corruption
    - Model incompatibilities

    Respects debug_mode by skipping model loading.
    """
    # pylint: disable=global-statement
    global _EMBEDDER, _EMBEDDER_NAME

    if debug_mode:
        logger.info("Debug mode enabled - skipping embedding model loading")
        return None

    reload_needed = _EMBEDDER is None or _EMBEDDER_NAME != model_name
    if not reload_needed:
        return _EMBEDDER

    # Handle OpenAI/API-based models
    if model_name in OPENAI_EMBED_DIMS:
        logger.warning("OpenAI embeddings temporarily unavailable after LangGraph migration. Falling back to default.")
        # Fallthrough to default strategies


    # Configure torch before loading
    try:
        # pylint: disable=import-outside-toplevel
        import torch
        torch.set_num_threads(1)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    # Progressive loading strategies - try each until one succeeds
    strategies = [
        {
            "name": "default",
            "device": "cpu",
            "backend": "torch",
            "model_kwargs": {"trust_remote_code": True, "low_cpu_mem_usage": False}
        },
        {
            "name": "low_memory",
            "device": "cpu",
            "backend": "torch",
            "model_kwargs": {"trust_remote_code": True, "low_cpu_mem_usage": True}
        },
        {
            "name": "minimal",
            "device": "cpu",
            "backend": "torch",
            "model_kwargs": {"trust_remote_code": True}
        },
        {
            "name": "basic",
            "device": "cpu",
            "backend": "torch",
            "model_kwargs": {}
        }
    ]

    last_error = None

    for strategy in strategies:
        try:
            logger.info(
                "Loading embedding model '%s' with strategy: %s",
                model_name,
                strategy["name"],
            )
            _EMBEDDER = _load_with_strategy(model_name, strategy, logger)
            _EMBEDDER_NAME = model_name
            logger.info("Successfully loaded '%s' with strategy: %s", model_name, strategy["name"])
            return _EMBEDDER

        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_error = exc
            error_type = "meta_tensor" if _is_meta_tensor_error(exc) else \
                        "device" if _is_device_error(exc) else "general"

            logger.warning(
                "Strategy '%s' failed for '%s' (%s error): %s",
                strategy["name"], model_name, error_type, exc
            )

            # For meta tensor or device errors, clear cache and retry once
            if _is_meta_tensor_error(exc) or _is_device_error(exc):
                logger.info("Detected %s error - clearing cache and retrying", error_type)
                if _clear_sentence_transformer_cache(model_name, logger):
                    try:
                        _EMBEDDER = _load_with_strategy(model_name, strategy, logger)
                        _EMBEDDER_NAME = model_name
                        logger.info("Successfully loaded '%s' after cache clear", model_name)
                        return _EMBEDDER
                    except Exception as retry_exc:  # pylint: disable=broad-exception-caught
                        logger.warning("Cache clear retry failed: %s", retry_exc)
                        last_error = retry_exc

    # All strategies failed - fall back to reliable model
    fallback = "all-MiniLM-L6-v2"
    logger.warning(
        "All loading strategies failed for '%s' (last error: %s). Falling back to '%s'",
        model_name, last_error, fallback
    )

    try:
        _clear_sentence_transformer_cache(fallback, logger)
        _EMBEDDER = SentenceTransformer(
            fallback,
            device='cpu',
            backend='torch'
        )
        _EMBEDDER_NAME = fallback
        logger.info("Fallback embedding model '%s' loaded successfully", fallback)
        return _EMBEDDER

    except Exception as fallback_exc:  # pylint: disable=broad-exception-caught
        logger.critical(
            "Fallback embedding model '%s' also failed to load: %s",
            fallback, fallback_exc
        )
        raise RuntimeError(
            f"Failed to load both requested model '{model_name}' and fallback '{fallback}'"
        ) from fallback_exc
