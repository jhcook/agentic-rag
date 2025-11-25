"""
OpenAI Assistants backend with function calling bridge to local FAISS.

This backend uses OpenAI Assistants API to orchestrate conversations,
but keeps all document data local. The assistant calls our local search
function instead of uploading files to OpenAI.
"""

import logging
import json
import os
import time
from typing import List, Dict, Any

from openai import OpenAI, OpenAIError
from openai.types.beta.threads import Run

# Import local RAG core for search
# pylint: disable=cyclic-import,protected-access
import src.core.rag_core as local_core

logger = logging.getLogger(__name__)
# pylint: disable=protected-access


class OpenAIAssistantsBackend:
    """
    OpenAI Assistants backend with local search function calling.

    Architecture:
    - User documents stay in local FAISS (privacy, no upload)
    - OpenAI Assistant orchestrates conversation flow
    - When assistant needs info, it calls our search_documents function
    - We execute local search and return results
    - Assistant synthesizes response with GPT-4 quality
    """

    def __init__(self):
        """Initialize OpenAI Assistants backend."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=self.api_key)
        self.assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
        self.model = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4-turbo-preview")

        # Create or get assistant
        if not self.assistant_id:
            self.assistant = self._create_assistant()
            self.assistant_id = self.assistant.id
            logger.info("Created OpenAI Assistant: %s", self.assistant_id)
        else:
            try:
                self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
                logger.info("Using existing OpenAI Assistant: %s", self.assistant_id)
            except OpenAIError as exc:
                logger.warning("Failed to retrieve assistant %s: %s", self.assistant_id, exc)
                self.assistant = self._create_assistant()
                self.assistant_id = self.assistant.id

        # Active threads (could persist these)
        self.threads: Dict[str, str] = {}

        logger.info("OpenAI Assistants backend initialized")

    def _create_assistant(self):
        """Create a new OpenAI Assistant with function calling."""
        return self.client.beta.assistants.create(
            name="Local RAG Assistant",
            instructions=(
                "You are a helpful AI assistant that answers questions based on the user's "
                "local documents.\n\n"
                "When a user asks a question, use the search_documents function to find relevant "
                "information from their indexed documents. Then synthesize a clear, accurate "
                "answer based on the retrieved context.\n\n"
                "IMPORTANT: Always cite your sources using [1], [2], etc. format inline where "
                "you use information.\n\n"
                'At the end of your response, include a "Sources:" section listing the document '
                "URIs you referenced.\n\n"
                "Example format:\n"
                "According to the documentation [1], the main feature is X. This is further "
                "explained in [2] where it mentions Y.\n\n"
                "Sources:\n"
                "[1] /path/to/document1.txt\n"
                "[2] /path/to/document2.md\n\n"
                "If no relevant documents are found, politely say you don't have information "
                "about that topic in the indexed documents."
            ),
            model=self.model,
            tools=[{
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": (
                        "Search the user's locally indexed documents for relevant information. "
                        "Returns the most relevant document passages based on semantic similarity."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant documents"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of documents to return (1-20)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            }]
        )

    def search_documents_function(self, query: str, top_k: int = 5) -> str:
        """
        Function called by OpenAI Assistant to search local documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            JSON string with search results
        """
        try:
            # Call local FAISS search
            results = local_core.search(query, top_k=min(top_k, 20))

            # Format for assistant consumption
            if "error" in results:
                return json.dumps({"error": results["error"]})

            passages = []
            for i, result in enumerate(results.get("results", []), 1):
                passages.append({
                    "index": i,
                    "uri": result.get("uri", "unknown"),
                    "text": result.get("text", ""),
                    "score": result.get("score", 0.0)
                })

            return json.dumps({
                "passages": passages,
                "total": len(passages)
            })

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error in search_documents_function: %s", exc)
            return json.dumps({"error": str(exc)})

    def _handle_requires_action(self, run: Run, thread_id: str) -> Run:
        """Handle function calls requested by the assistant."""
        if not run.required_action:
            return run

        tool_outputs = []

        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            if tool_call.function.name == "search_documents":
                try:
                    # Parse arguments
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")
                    top_k = args.get("top_k", 5)

                    logger.info(
                        "Assistant calling search_documents: query='%s', top_k=%s",
                        query,
                        top_k,
                    )

                    # Execute local search
                    output = self.search_documents_function(query, top_k)

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output
                    })

                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("Error handling function call: %s", exc)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps({"error": str(exc)})
                    })

        # Submit tool outputs back to assistant
        if tool_outputs:
            run = self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        return run

    def _wait_for_run_completion(self, run: Run, thread_id: str, timeout: int = 60) -> Run:
        """Wait for run to complete, handling function calls."""
        start_time = time.time()

        while run.status in ["queued", "in_progress", "requires_action"]:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Run timed out after {timeout}s")

            if run.status == "requires_action":
                run = self._handle_requires_action(run, thread_id)

            time.sleep(0.5)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )

        if run.status == "failed":
            error = run.last_error.message if run.last_error else "Unknown error"
            raise RuntimeError(f"Run failed: {error}")

        return run

    @staticmethod
    def _parse_sources_section(content: str) -> List[str]:
        """Extract sources from assistant response content."""
        if "Sources:" not in content:
            return []

        sources: List[str] = []
        sources_section = content.split("Sources:", maxsplit=1)[1].strip()
        for line in sources_section.splitlines():
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            parts = line.split("]", 1)
            if len(parts) > 1:
                uri = parts[1].strip()
                if uri:
                    sources.append(uri)
        return sources

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """
        Chat with the OpenAI Assistant.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional options (ignored for now)

        Returns:
            Dict with 'content' and optional 'sources'
        """
        try:
            if kwargs:
                logger.debug("Extra chat kwargs ignored: %s", list(kwargs.keys()))
            user_message = next(
                (msg.get("content") for msg in reversed(messages) if msg.get("role") == "user"),
                None,
            )
            if not user_message:
                return {"error": "No user message found"}

            thread = self.client.beta.threads.create()
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            run = self._wait_for_run_completion(run, thread.id)

            thread_messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                order="desc",
                limit=1
            )
            if not thread_messages.data:
                return {"error": "No response from assistant"}

            assistant_message = thread_messages.data[0]
            content_blocks = [
                block.text.value
                for block in assistant_message.content
                if getattr(block, "type", None) == "text"
            ]
            content = "".join(content_blocks)
            sources = self._parse_sources_section(content)

            return {"content": content, "sources": sources or None}

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error in OpenAI Assistants chat: %s", exc)
            return {"error": str(exc)}

    # Delegate other methods to local backend
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for documents."""
        return local_core.search(query, top_k=top_k)

    def upsert_document(self, uri: str, text: str) -> Dict[str, Any]:
        """Add or update a document."""
        return local_core.upsert_document(uri, text)

    def index_path(self, path: str, glob: str = "**/*") -> Dict[str, Any]:
        """Index a directory path."""
        # Use LocalBackend implementation
        from src.core.factory import LocalBackend  # pylint: disable=import-outside-toplevel
        backend = LocalBackend()
        return backend.index_path(path, glob)

    def grounded_answer(self, question: str, k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        """Generate a grounded answer using OpenAI Assistant."""
        _ = k  # Argument kept for interface compatibility; OpenAI Assistant handles retrieval size
        # Convert to chat format
        messages = [{"role": "user", "content": question}]
        return self.chat(messages, **kwargs)

    def load_store(self) -> bool:
        """Load the document store."""
        local_core.load_store()
        return True

    def save_store(self) -> bool:
        """Save the document store."""
        local_core.save_store()
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        local_core._ensure_store_synced()
        store = local_core.get_store()
        return [{"uri": uri, "size": len(text)} for uri, text in store.docs.items()]

    def rebuild_index(self) -> None:
        """Rebuild the vector index."""
        local_core._rebuild_faiss_index()

    def rerank(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank passages."""
        return local_core.rerank(query, passages)

    def verify_grounding(self, question: str, answer: str, citations: List[str]) -> Dict[str, Any]:
        """Verify answer grounding."""
        return local_core.verify_grounding(question, answer, citations)

    def delete_documents(self, uris: List[str]) -> Dict[str, Any]:
        """Delete documents."""
        from src.core.factory import LocalBackend  # pylint: disable=import-outside-toplevel
        backend = LocalBackend()
        return backend.delete_documents(uris)

    def flush_cache(self) -> Dict[str, Any]:
        """Flush cache."""
        from src.core.factory import LocalBackend  # pylint: disable=import-outside-toplevel
        backend = LocalBackend()
        return backend.flush_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        from src.core.factory import LocalBackend  # pylint: disable=import-outside-toplevel
        backend = LocalBackend()
        stats = backend.get_stats()
        stats["backend"] = "openai_assistants"
        stats["assistant_id"] = self.assistant_id
        stats["model"] = self.model
        return stats
