
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any

# --- Common Models ---

class UpsertReq(BaseModel):
    """Request model for upserting a document."""
    uri: str
    text: Optional[str] = None
    binary_base64: Optional[str] = None

class IndexPathReq(BaseModel):
    """Request model for indexing a filesystem path."""
    path: str
    glob: Optional[str] = "**/*.txt"

class IndexUrlReq(BaseModel):
    """Request model for indexing remote URLs."""
    url: str
    doc_id: Optional[str] = Field(default=None, alias="docId")
    query: Optional[str] = None  # Used by mcp tool

class SearchReq(BaseModel):
    """Request model for performing a search."""
    query: str = Field(description="The search query")
    async_mode: bool = Field(default=False, alias="async", description="Run search asynchronously with polling")
    timeout_seconds: Optional[int] = Field(default=300, description="Timeout for async searches in seconds")
    model: Optional[str] = Field(default=None, description="Override LLM model (Ollama only, e.g. 'ollama/qwen2.5:3b')")
    temperature: Optional[float] = Field(default=None, description="Generation temperature 0.0-1.0 (lower=factual, higher=creative)")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response")
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve (default 5)")

    model_config = ConfigDict(populate_by_name=True)

class VectorSearchReq(BaseModel):
    """Request model for vector search (without LLM)."""
    query: str
    k: Optional[int] = 5

class GroundedAnswerReq(BaseModel):
    """Request model for grounded answer generation."""
    question: str
    k: Optional[int] = 3
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    config: Optional[Dict[str, Any]] = None

class RerankReq(BaseModel):
    """Request model for reranking passages."""
    query: str
    passages: List[Dict[str, Any]]
    top_k: Optional[int] = None

class VerifyReq(BaseModel):
    """Request model for grounding verification (citations)."""
    question: str
    draft_answer: str
    citations: List[str]

class VerifySimpleReq(BaseModel):
    """Request model for grounding verification (passages provided)."""
    question: str
    draft_answer: str
    passages: List[Dict[str, Any]]

class DeleteDocsReq(BaseModel):
    """Request model for deleting documents by URI."""
    uris: list[str]

class LoggingConfigReq(BaseModel):
    """Request model for logging configuration."""
    debug_mode: bool

# --- Response Models ---

class HealthResp(BaseModel):
    """Response model for health checks."""
    status: str
    base_path: str
    documents: int
    vectors: int
    memory_mb: float
    memory_limit_mb: int
    total_size_bytes: int = 0
    store_file_bytes: int = 0

class DocumentInfo(BaseModel):
    """Document metadata for listings."""
    uri: str
    size_bytes: int

class DocumentsResp(BaseModel):
    """Response model for document listing."""
    documents: list[DocumentInfo]

class Job(BaseModel):
    """Async job model."""
    id: str
    type: str
    status: str
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    """Model for a chat message."""
    role: str
    content: str

class ChatReq(BaseModel):
    """Request model for chat completion."""
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    config: Optional[Dict[str, Any]] = None

# --- Config Models ---

class ConfigModeReq(BaseModel):
    """Request model for setting the backend mode."""
    mode: str = Field(description="Backend mode: 'ollama', 'openai_assistants', 'google_gemini', or 'vertex_ai_search'")

class OpenAIConfigModel(BaseModel):
    """OpenAI configuration model."""
    api_key: str = Field(description="OpenAI API key (sk-...)", alias="apiKey")
    model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model to use")
    assistant_id: str = Field(default="", description="Optional OpenAI Assistant ID", alias="assistantId")
    
    class Config:
        populate_by_name = True

class VertexConfigReq(BaseModel):
    """Request model for Vertex AI configuration."""
    project_id: str
    location: str
    data_store_id: str

class OllamaModeReq(BaseModel):
    """Request model for setting Ollama mode."""
    mode: str = Field(description="Ollama mode: 'local', 'cloud', or 'auto'")

class OllamaTestConnectionReq(BaseModel):
    """Request model for testing Ollama Cloud connection."""
    api_key: Optional[str] = Field(default=None, description="API key to test (optional, uses configured key if not provided)")
    endpoint: Optional[str] = Field(default=None, description="Endpoint to test (optional, uses configured endpoint if not provided)")
    proxy: Optional[str] = Field(default=None, description="HTTPS proxy to use when testing (optional)")
    ca_bundle: Optional[str] = Field(default=None, description="Path to CA bundle PEM file (optional)")

class OllamaCloudConfigReq(BaseModel):
    """Request model for persisting Ollama Cloud secrets/config."""
    api_key: Optional[str] = Field(default=None, description="API key to store (masked value retains existing)", alias="apiKey")
    endpoint: Optional[str] = Field(default=None, description="Cloud endpoint URL")
    proxy: Optional[str] = Field(default=None, description="HTTPS proxy URL")
    ca_bundle: Optional[str] = Field(default=None, description="Path to CA bundle PEM file", alias="caBundle")

    model_config = ConfigDict(populate_by_name=True)

class OllamaStatusResp(BaseModel):
    """Response model for Ollama connection status."""
    mode: str
    endpoint: str
    cloud_available: bool
    local_available: bool
    cloud_status: Optional[str] = None  # "connected", "disconnected", "error", None
    local_status: Optional[str] = None  # "connected", "disconnected", "error", None

class OllamaTestConnectionResp(BaseModel):
    """Response model for Ollama Cloud connection test."""
    success: bool
    message: str

class AppConfigReq(BaseModel):
    """Request model for application configuration."""
    api_endpoint: str = Field(alias="apiEndpoint")
    model: str
    embedding_model: str = Field(alias="embeddingModel")
    temperature: str
    top_p: str = Field(alias="topP")
    top_k: str = Field(alias="topK")
    repeat_penalty: str = Field(alias="repeatPenalty")
    seed: str
    num_ctx: str = Field(alias="numCtx")
    mcp_host: str = Field(alias="mcpHost")
    mcp_port: str = Field(alias="mcpPort")
    mcp_path: str = Field(alias="mcpPath")
    rag_host: str = Field(alias="ragHost")
    rag_port: str = Field(alias="ragPort")
    rag_path: str = Field(alias="ragPath")
    debug_mode: Optional[bool] = Field(default=False, alias="debugMode")
    ollama_cloud_proxy: Optional[str] = Field(default=None, alias="ollamaCloudProxy")
    ollama_cloud_endpoint: Optional[str] = Field(default=None, alias="ollamaCloudEndpoint")
    ollama_mode: Optional[str] = Field(default=None, alias="ollamaMode")

    model_config = ConfigDict(populate_by_name=True)

class QualityMetricsResp(BaseModel):
    """Aggregated quality metrics for searches."""
    total_searches: int
    failed_searches: int
    responses_with_sources: int
    total_sources: int
    fallback_responses: int
    success_rate: float
    avg_sources: float

class LoadStoreReq(BaseModel):
    """Request model for sending the store to an LLM."""
    _ : Optional[bool] = True
