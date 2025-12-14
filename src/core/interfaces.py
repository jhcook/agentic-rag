"""Protocol interfaces for RAG backends."""
from typing import Protocol, List, Dict, Any

# pylint: disable=unnecessary-ellipsis
class RAGBackend(Protocol):
    """Interface for RAG operations (Local or Remote)."""

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Search for documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Optional parameters like model, temperature, max_tokens
        """
        ...

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        """Add or update a document."""
        ...

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        """Index a directory."""
        ...

    def grounded_answer(self, question: str, k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Generate an answer based on search results."""
        ...

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents with metadata (uri, size, etc)."""
        ...

    def rebuild_index(self) -> None:
        """Rebuild the vector index from the canonical indexed artifacts."""
        ...

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank passages based on relevance to the query."""
        ...

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        """Verify that an answer is grounded in the cited documents."""
        ...

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        """Delete documents by URI."""
        ...

    def flush_cache(self) -> Dict[str, Any]:
        """Clear all indexed content (vectors + indexed artifacts)."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics (health check)."""
        ...

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Conversational chat with the backend."""
        ...
