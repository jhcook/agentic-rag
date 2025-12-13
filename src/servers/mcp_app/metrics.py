import logging
from typing import Dict, Optional

import requests
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from starlette.responses import Response

from src.servers.mcp_app.memory import get_memory_usage, MAX_MEMORY_MB
from src.core.rag_core import get_store, EMBED_MODEL_NAME
from src.core import pgvector_store
from src.servers.mcp_app.memory import get_system_memory_mb

logger = logging.getLogger(__name__)

METRICS_REGISTRY = CollectorRegistry()
MCP_DOCUMENTS_INDEXED = Gauge(
    "mcp_documents_indexed_total",
    "Number of documents indexed in the MCP store.",
    registry=METRICS_REGISTRY,
)
MCP_MEMORY_USAGE_MB = Gauge(
    "mcp_memory_usage_megabytes",
    "Current process RSS memory usage in megabytes.",
    registry=METRICS_REGISTRY,
)
MCP_MEMORY_LIMIT_MB = Gauge(
    "mcp_memory_limit_megabytes",
    "Configured memory limit in megabytes.",
    registry=METRICS_REGISTRY,
)
MCP_EMBEDDING_VECTORS = Gauge(
    "mcp_embedding_vectors_total",
    "Total vectors stored in the embedding index.",
    registry=METRICS_REGISTRY,
)
MCP_EMBEDDING_CHUNKS = Gauge(
    "mcp_embedding_chunks_total",
    "Total text chunks tracked for embeddings.",
    registry=METRICS_REGISTRY,
)
MCP_EMBEDDING_DIM = Gauge(
    "mcp_embedding_dimension",
    "Embedding dimension used by the current model.",
    registry=METRICS_REGISTRY,
)

OLLAMA_UP = Gauge(
    "ollama_up",
    "Whether the Ollama API responded successfully.",
    registry=METRICS_REGISTRY,
)
OLLAMA_RUNNING_MODELS = Gauge(
    "ollama_running_models",
    "Count of running models reported by Ollama /api/ps.",
    registry=METRICS_REGISTRY,
)
OLLAMA_AVAILABLE_MODELS = Gauge(
    "ollama_available_models",
    "Count of available models reported by Ollama /api/tags.",
    registry=METRICS_REGISTRY,
)
OLLAMA_RUNNING_MODEL_SIZE_BYTES = Gauge(
    "ollama_running_model_size_bytes",
    "Resident size of running models reported by Ollama.",
    ["model", "digest"],
    registry=METRICS_REGISTRY,
)
OLLAMA_RUNNING_MODEL_VRAM_BYTES = Gauge(
    "ollama_running_model_vram_bytes",
    "VRAM usage of running models reported by Ollama.",
    ["model", "digest"],
    registry=METRICS_REGISTRY,
)
OLLAMA_MODEL_SIZE_BYTES = Gauge(
    "ollama_model_size_bytes",
    "Size of available Ollama models.",
    ["model", "digest"],
    registry=METRICS_REGISTRY,
)


def _normalize_ollama_base(base: str) -> str:
    return base.rstrip("/")


def _fetch_ollama_metrics(base_url: str) -> Dict[str, Optional[int]]:
    """Fetch metrics from Ollama endpoints (/api/ps and /api/tags)."""
    base_url = _normalize_ollama_base(base_url)
    metrics: Dict[str, Optional[int]] = {"up": False}
    try:
        ps_resp = requests.get(f"{base_url}/api/ps", timeout=2)
        ps_resp.raise_for_status()
        data = ps_resp.json() or {}
        metrics["running_models"] = data.get("models", []) or []
        metrics["running_model_count"] = len(metrics["running_models"])
        metrics["up"] = True
    except Exception as exc:
        logger.debug("Ollama /api/ps metrics unavailable: %s", exc)

    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=2)
        tags_resp.raise_for_status()
        body = tags_resp.json() or {}
        metrics["available_models"] = body.get("models", []) or []
        metrics["up"] = metrics["up"] or True
    except Exception as exc:
        logger.debug("Ollama /api/tags metrics unavailable: %s", exc)

    return metrics


def _set_labeled_gauge(gauge: Gauge, labels: Dict[str, str], value: Optional[int]):
    """Safely set a labeled gauge when a numeric value is available."""
    if value is None:
        return
    gauge.labels(**labels).set(value)


def refresh_prometheus_metrics(ollama_base: str):
    """Update Prometheus gauges with current server state."""
    try:
        store = get_store()
        MCP_DOCUMENTS_INDEXED.set(len(getattr(store, "docs", {})))
    except Exception as exc:
        logger.debug("Failed to update document metrics: %s", exc)

    try:
        MCP_MEMORY_USAGE_MB.set(get_memory_usage())
        MCP_MEMORY_LIMIT_MB.set(MAX_MEMORY_MB)
    except Exception as exc:
        logger.debug("Failed to update memory metrics: %s", exc)

    try:
        stats = pgvector_store.stats(embedding_model=EMBED_MODEL_NAME)
        MCP_EMBEDDING_VECTORS.set(int(stats.get("chunks", 0)))
        MCP_EMBEDDING_CHUNKS.set(int(stats.get("chunks", 0)))
        MCP_EMBEDDING_DIM.set(int(stats.get("embedding_dim", 0)))
    except Exception as exc:
        logger.debug("Failed to update embedding metrics: %s", exc)

    try:
        ollama = _fetch_ollama_metrics(ollama_base)
        OLLAMA_UP.set(1 if ollama.get("up") else 0)

        running = ollama.get("running_models", []) or []
        OLLAMA_RUNNING_MODELS.set(len(running))
        for model in running:
            try:
                name = model.get("name") or "unknown"
                digest = model.get("digest") or "unknown"
                size = model.get("size")
                vram = model.get("vram")
                _set_labeled_gauge(OLLAMA_RUNNING_MODEL_SIZE_BYTES, {"model": name, "digest": digest}, size)
                _set_labeled_gauge(OLLAMA_RUNNING_MODEL_VRAM_BYTES, {"model": name, "digest": digest}, vram)
            except Exception as exc:
                logger.debug("Failed to set running model gauge: %s", exc)

        available = ollama.get("available_models", []) or []
        OLLAMA_AVAILABLE_MODELS.set(len(available))
        for model in available:
            try:
                name = model.get("name") or "unknown"
                digest = model.get("digest") or "unknown"
                size = model.get("size")
                _set_labeled_gauge(OLLAMA_MODEL_SIZE_BYTES, {"model": name, "digest": digest}, size)
            except Exception as exc:
                logger.debug("Failed to set model size gauge: %s", exc)
    except Exception as exc:
        logger.debug("Failed to update Ollama metrics: %s", exc)


def metrics_response(refresh_error=None) -> Response:
    """Render Prometheus metrics payload."""
    try:
        payload = generate_latest(METRICS_REGISTRY)
        if refresh_error:
            payload += f"# metrics refresh error: {refresh_error}\n".encode()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)
    except Exception as exc:
        logger.error("Failed to render metrics: %s", exc, exc_info=True)
        return Response(f"# metrics unavailable: {exc}\n", status_code=200, media_type="text/plain")
