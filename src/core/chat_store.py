"""PostgreSQL-backed chat persistence.

Stores chat sessions and messages locally for transcript rehydration.
"""

import json
import logging
import uuid
from typing import List, Dict, Any, Optional

from src.core import pgvector_store

logger = logging.getLogger(__name__)


class ChatStore:
    """Persistent storage for chat sessions and messages using PostgreSQL."""

    def __init__(self, db_path: str = None):
        # db_path is no longer used, but kept for backward compatibility during initialization.
        pass

    def _get_connection(self):
        """Get a database connection from the pool."""
        return pgvector_store.get_pool().connection()

    def _recompute_session_updated_at(self, conn, session_id: str) -> None:
        """Recompute a session's updated_at based on its remaining messages."""
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversations
                SET updated_at = COALESCE(
                    (SELECT MAX(created_at) FROM conversation_messages WHERE session_id = %s),
                    created_at
                )
                WHERE id = %s
                """,
                (session_id, session_id),
            )

    def create_session(
        self,
        title: str = "New Chat",
        metadata: Dict[str, Any] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Create a new chat session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        meta_json = json.dumps(metadata) if metadata else None

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (id, title, metadata) VALUES (%s, %s, %s)",
                    (session_id, title, meta_json),
                )
            conn.commit()
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details (excluding soft-deleted)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, title, created_at, updated_at, metadata FROM conversations WHERE id = %s AND deleted_at IS NULL", (session_id,))
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "title": row[1],
                        "created_at": row[2].isoformat(),
                        "updated_at": row[3].isoformat(),
                        "metadata": json.loads(row[4]) if row[4] else None,
                    }
        return None

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List recent chat sessions (excluding soft-deleted)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title, created_at, updated_at, metadata FROM conversations WHERE deleted_at IS NULL ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cur.fetchall()
                sessions = []
                for row in rows:
                    sessions.append(
                        {
                            "id": row[0],
                            "title": row[1],
                            "created_at": row[2].isoformat(),
                            "updated_at": row[3].isoformat(),
                            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else None,
                        }
                    )
                return sessions

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        display_content: Optional[str] = None,
        sources: Optional[List[str]] = None,
        kind: Optional[str] = None,
    ) -> str:
        """Add a message to a session."""
        msg_id = str(uuid.uuid4())

        if display_content is None:
            display_content = content

        sources_json = json.dumps(sources) if sources is not None else None

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    (
                        "INSERT INTO conversation_messages (id, session_id, role, content, display_content, sources, kind) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    ),
                    (msg_id, session_id, role, content, display_content, sources_json, kind),
                )
                self._recompute_session_updated_at(conn, session_id)
            conn.commit()
        return msg_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by time."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, role, content, display_content, sources, kind, created_at FROM conversation_messages WHERE session_id = %s AND deleted_at IS NULL ORDER BY created_at ASC",
                    (session_id,),
                )
                rows = cur.fetchall()
                messages: List[Dict[str, Any]] = []
                for row in rows:
                    messages.append(
                        {
                            "id": row[0],
                            "role": row[1],
                            "content": row[2],
                            "display_content": row[3],
                            "sources": row[4] if isinstance(row[4], list) else json.loads(row[4]) if row[4] else [],
                            "kind": row[5],
                            "created_at": row[6].isoformat(),
                        }
                    )
                return messages

    def delete_session(self, session_id: str) -> bool:
        """Soft delete a session (marks as deleted without removing data)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET deleted_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                    (session_id,),
                )
                deleted_count = cur.rowcount
            conn.commit()
            return deleted_count > 0

    def delete_message(self, session_id: str, message_id: str) -> bool:
        """Mark a single message as deleted."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversation_messages SET deleted_at = NOW() WHERE id = %s AND session_id = %s",
                    (message_id, session_id),
                )
                deleted = cur.rowcount > 0
                if deleted:
                    self._recompute_session_updated_at(conn, session_id)
            conn.commit()
            return deleted

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET title = %s WHERE id = %s",
                    (title, session_id),
                )
                updated_count = cur.rowcount
            conn.commit()
            return updated_count > 0