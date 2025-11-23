from typing import Protocol, List, Dict, Any, Optional

class RAGBackend(Protocol):
    """Interface for RAG operations (Local or Remote)."""

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for documents."""
        ...

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        """Add or update a document."""
        ...

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        """Index a directory."""
        ...

    def grounded_answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Generate an answer based on search results."""
        ...

    def load_store(self) -> bool:
        """Load or reload the document store."""
        ...

    def save_store(self) -> bool:
        """Save the document store to disk."""
        ...

    def list_documents(self) -> List[str]:
        """List all indexed document URIs."""
        ...

    def rebuild_index(self) -> None:
        """Rebuild the vector index from the document store."""
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
        """Clear the document store."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics (health check)."""
        ...

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Conversational chat with the backend."""
        ...
