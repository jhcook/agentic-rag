from prometheus_client import (
    CollectorRegistry,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

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

def metrics_endpoint(_request):
    payload = generate_latest(METRICS_REGISTRY)
    return Response(payload, media_type=CONTENT_TYPE_LATEST)
