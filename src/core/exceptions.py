"""Custom exceptions for the Agentic RAG system."""

class RAGError(Exception):
    """Base exception for RAG system."""
    pass

class ConfigurationError(RAGError):
    """Raised when the system is misconfigured (e.g. missing keys, invalid URLs)."""
    pass

class AuthenticationError(RAGError):
    """Raised when authentication fails (e.g. 401 Unauthorized, invalid tokens)."""
    pass

class ProviderError(RAGError):
    """Raised when the backend provider errors (e.g. 500, overload)."""
    pass
