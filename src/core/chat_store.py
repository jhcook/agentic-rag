
import sqlite3
import json
import logging
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChatStore:
    """
    Persistent storage for chat sessions and messages using SQLite.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(
            str(self.db_path), 
            check_same_thread=False
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
                        created_at REAL,
                        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);")
                
        except Exception as e:
            logger.error("Failed to initialize chat database: %s", e)
            raise

    def create_session(self, title: str = "New Chat", metadata: Dict[str, Any] = None) -> str:
        """Create a new chat session."""
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

    def add_message(self, session_id: str, role: str, content: str) -> str:
        """Add a message to a session."""
        msg_id = str(uuid.uuid4())
        now = time.time()
        
        with self._get_connection() as conn:
            # Insert message
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (msg_id, session_id, role, content, now)
            )
            # Update session timestamp
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id)
            )
        return msg_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by time."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id)
            )
            return cursor.rowcount > 0
