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
