"""SQLite-backed chat persistence.

Stores chat sessions and messages locally for transcript rehydration.
"""

import sqlite3
import json
import logging
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChatStore:
    """Persistent storage for chat sessions and messages using SQLite."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # Ensure FK constraints like ON DELETE CASCADE are enforced.
        try:
            conn.execute("PRAGMA foreign_keys=ON;")
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return conn

    def _recompute_session_updated_at(self, conn: sqlite3.Connection, session_id: str) -> None:
        """Recompute a session's updated_at based on its remaining messages."""
        conn.execute(
            """
            UPDATE sessions
            SET updated_at = COALESCE(
                (SELECT MAX(created_at) FROM messages WHERE session_id = ?),
                created_at
            )
            WHERE id = ?
            """,
            (session_id, session_id),
        )

    def _init_db(self):
        """Initialize the database schema."""
        try:
            with self._get_connection() as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL;")
                
                # Sessions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        created_at REAL,
                        updated_at REAL,
                        metadata TEXT
                    );
                """)
                
                # Messages table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        display_content TEXT,
                        sources TEXT,
                        kind TEXT,
                        created_at REAL,
                        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);")

                # Backward-compatible schema migration for existing databases.
                # SQLite doesn't support IF NOT EXISTS for ADD COLUMN in all versions.
                for column_name, column_type in (
                    ("display_content", "TEXT"),
                    ("sources", "TEXT"),
                    ("kind", "TEXT"),
                ):
                    try:
                        conn.execute(
                            f"ALTER TABLE messages ADD COLUMN {column_name} {column_type};"
                        )
                    except sqlite3.OperationalError as exc:
                        # Ignore "duplicate column name" (already migrated)
                        if "duplicate column name" not in str(exc).lower():
                            raise
                
        except Exception as e:
            logger.error("Failed to initialize chat database: %s", e)
            raise

    def create_session(
        self,
        title: str = "New Chat",
        metadata: Dict[str, Any] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Create a new chat session.

        If session_id is provided, it will be used (idempotency is handled by callers).
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        now = time.time()
        meta_json = json.dumps(metadata or {})
        
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, title, now, now, meta_json)
            )
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List recent chat sessions."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            return [dict(row) for row in rows]

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
        now = time.time()

        if display_content is None:
            display_content = content

        sources_json = None
        if sources is not None:
            try:
                sources_json = json.dumps(sources)
            except Exception:  # pylint: disable=broad-exception-caught
                sources_json = None
        
        with self._get_connection() as conn:
            # Insert message
            conn.execute(
                (
                    "INSERT INTO messages (id, session_id, role, content, "
                    "display_content, sources, kind, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (msg_id, session_id, role, content, display_content, sources_json, kind, now)
            )
            self._recompute_session_updated_at(conn, session_id)
        return msg_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by time."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,)
            ).fetchall()
            messages: List[Dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                # Normalize sources back to list for API consumers.
                raw_sources = d.get("sources")
                if isinstance(raw_sources, str) and raw_sources:
                    try:
                        parsed = json.loads(raw_sources)
                        d["sources"] = parsed if isinstance(parsed, list) else []
                    except Exception:  # pylint: disable=broad-exception-caught
                        d["sources"] = []
                elif raw_sources is None:
                    d["sources"] = []
                messages.append(d)
            return messages

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def delete_message(self, session_id: str, message_id: str) -> bool:
        """Delete a single message from a session."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE id = ? AND session_id = ?",
                (message_id, session_id),
            )
            deleted = cursor.rowcount > 0
            if deleted:
                self._recompute_session_updated_at(conn, session_id)
            return deleted

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id)
            )
            return cursor.rowcount > 0
