"""PostgreSQL + pgvector vector store.

This module provides the project's vector search/index storage via pgvector.
It is intentionally synchronous to match the existing rag_core call sites.

Connection parameters are sourced from (highest precedence first):
- environment variables (PGVECTOR_*)
- config/settings.json (pgvector* keys)

Secrets (password) should be provided via environment variables. The UI can set
these at runtime via REST endpoints, but Docker Compose initialization still
requires PGVECTOR_PASSWORD in .env.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pgvector.psycopg import Vector, register_vector
from psycopg_pool import ConnectionPool


def redact_error_message(message: str) -> str:
    """Redact connection secrets from error messages.

    psycopg/pgvector errors can include DSNs/conninfo strings which may contain
    passwords. This helper is used anywhere we log or return raw exception text.
    """

    redacted = message
    redacted = re.sub(r"(password=)\S+", r"\1***MASKED***", redacted)
    redacted = re.sub(r"(postgres(?:ql)?://[^:\s]+:)[^@\s]+@", r"\1***MASKED***@", redacted)
    redacted = re.sub(r"(PGVECTOR_PASSWORD\s*[:=]\s*)\S+", r"\1***MASKED***", redacted, flags=re.IGNORECASE)
    return redacted


@dataclass(frozen=True)
class PgvectorConfig:
    """Connection configuration for the local pgvector Postgres instance."""

    host: str
    port: int
    dbname: str
    user: str
    password: str


def _read_settings() -> Dict[str, Any]:
    settings_path = Path("config/settings.json")
    if not settings_path.exists():
        return {}
    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_secrets() -> Dict[str, Any]:
    secrets_path = Path("secrets/pgvector_config.json")
    if not secrets_path.exists():
        return {}
    try:
        return json.loads(secrets_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def load_pgvector_config() -> PgvectorConfig:
    """Load pgvector connection config from env and settings.json."""

    settings = _read_settings()
    secrets = _read_secrets()

    host = os.getenv("PGVECTOR_HOST") or settings.get("pgvectorHost") or "127.0.0.1"
    port_str = os.getenv("PGVECTOR_PORT") or settings.get("pgvectorPort") or "5432"
    dbname = os.getenv("PGVECTOR_DB") or settings.get("pgvectorDb") or "agentic_rag"
    user = os.getenv("PGVECTOR_USER") or settings.get("pgvectorUser") or "agenticrag"
    password = os.getenv("PGVECTOR_PASSWORD") or secrets.get("password") or ""

    try:
        port = int(port_str)
    except (TypeError, ValueError):
        port = 5432

    if not password:
        raise RuntimeError("PGVECTOR_PASSWORD is not set")

    return PgvectorConfig(host=host, port=port, dbname=dbname, user=user, password=password)


_POOL: Optional[ConnectionPool] = None


def _configure_connection(conn: Any) -> None:
    """Configure a new psycopg connection for pgvector.

    pgvector requires registering type adapters so Python `Vector` values can be
    sent as query parameters.
    """

    try:
        register_vector(conn)
    except Exception:
        # Best-effort: if registration fails, queries using Vector will fail
        # with a clear adaptation error.
        return


def _dsn(cfg: PgvectorConfig) -> str:
    # Keep explicit for clarity; psycopg will escape values.
    return (
        f"host={cfg.host} port={cfg.port} dbname={cfg.dbname} user={cfg.user} password={cfg.password} "
        "connect_timeout=5"
    )


def get_pool() -> ConnectionPool:
    """Return a global connection pool."""

    global _POOL  # pylint: disable=global-statement
    if _POOL is not None:
        return _POOL

    cfg = load_pgvector_config()
    _POOL = ConnectionPool(
        conninfo=_dsn(cfg),
        min_size=1,
        max_size=8,
        timeout=10,
        max_idle=60,
        reconnect_timeout=5,
        configure=_configure_connection,
        name="agentic-rag-pgvector",
    )
    return _POOL


def close_pool() -> None:
    """Close the global pool if open."""

    global _POOL  # pylint: disable=global-statement
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def test_connection() -> Tuple[bool, str]:
    """Test DB connectivity."""

    try:
        pool = get_pool()
        with pool.connection() as conn:
            _configure_connection(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                _ = cur.fetchone()
        return True, "ok"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, redact_error_message(str(exc))


def migrate_schema(embed_dim: int) -> None:
    """Create pgvector extension and tables/indexes (idempotent)."""

    ddl = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
        "CREATE TABLE IF NOT EXISTS rag_documents ("
        "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
        "  uri TEXT UNIQUE NOT NULL,"
        "  text_sha256 TEXT,"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
        "  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        ");",
        "CREATE TABLE IF NOT EXISTS rag_chunks ("
        "  chunk_id BIGSERIAL PRIMARY KEY,"
        "  document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,"
        "  uri TEXT NOT NULL,"
        "  chunk_index INT NOT NULL,"
        "  chunk_text TEXT NOT NULL,"
        "  start_offset INT NULL,"
        "  end_offset INT NULL,"
        "  embedding_model TEXT NOT NULL,"
        "  embedding_dim INT NOT NULL,"
        f"  embedding VECTOR({embed_dim}) NOT NULL,"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
        "  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
        "  UNIQUE(uri, chunk_index, embedding_model)"
        ");",
        "CREATE INDEX IF NOT EXISTS idx_rag_documents_uri ON rag_documents(uri);",
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_uri ON rag_chunks(uri);",
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_model ON rag_chunks(embedding_model);",
        # Composite index for efficient DELETE during upsert (WHERE uri=%s AND embedding_model=%s)
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_uri_model ON rag_chunks(uri, embedding_model);",
        # HNSW index for normalized inner-product search
        "CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw "
        "  ON rag_chunks USING hnsw (embedding vector_ip_ops);",
        "CREATE TABLE IF NOT EXISTS performance_metrics ("
        "  id BIGSERIAL PRIMARY KEY,"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
        "  operation TEXT NOT NULL,"
        "  duration_ms DOUBLE PRECISION NOT NULL,"
        "  error TEXT NULL,"
        "  model TEXT NULL,"
        "  tokens INT NULL"
        ");",
        # Backfill columns in case the table exists from an older schema
        "ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();",
        "ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS operation TEXT NOT NULL DEFAULT 'unknown';",
        "ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0;",
        "ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS error TEXT NULL;",
        "ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS model TEXT NULL;",
        "ALTER TABLE performance_metrics ADD COLUMN IF NOT EXISTS tokens INT NULL;",
        "CREATE INDEX IF NOT EXISTS idx_performance_metrics_operation ON performance_metrics(operation);",
        "CREATE INDEX IF NOT EXISTS idx_performance_metrics_created_at ON performance_metrics(created_at DESC);",
    ]

    pool = get_pool()
    with pool.connection() as conn:
        _configure_connection(conn)
        with conn.cursor() as cur:
            for stmt in ddl:
                cur.execute(stmt)
        conn.commit()


def _now_ts() -> float:
    return time.time()


def upsert_document_chunks(
    *,
    uri: str,
    text_sha256: Optional[str] = None,
    chunks: Sequence[str],
    offsets: Sequence[Tuple[int, int]],
    embeddings: np.ndarray,
    embedding_model: str,
) -> int:
    """Upsert a document row and its chunk vectors. Returns number of chunks written."""

    if len(chunks) != len(offsets) or len(chunks) != len(embeddings):
        raise ValueError("chunks/offsets/embeddings length mismatch")

    pool = get_pool()
    with pool.connection() as conn:
        _configure_connection(conn)
        with conn.cursor() as cur:
            # Upsert document
            cur.execute(
                "INSERT INTO rag_documents(uri, text_sha256, updated_at) VALUES (%s, %s, NOW()) "
                "ON CONFLICT (uri) DO UPDATE SET text_sha256 = EXCLUDED.text_sha256, updated_at = EXCLUDED.updated_at "
                "RETURNING id",
                (uri, text_sha256),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("Failed to upsert rag_documents")
            document_id = row[0]

            # Remove previous chunks for this uri+model (simplest correctness)
            cur.execute(
                "DELETE FROM rag_chunks WHERE uri=%s AND embedding_model=%s",
                (uri, embedding_model),
            )

            # Insert chunks
            dim = int(embeddings.shape[1])
            for idx, (chunk_text, (start, end)) in enumerate(zip(chunks, offsets)):
                cur.execute(
                    "INSERT INTO rag_chunks(document_id, uri, chunk_index, chunk_text, start_offset, end_offset, "
                    "embedding_model, embedding_dim, embedding, updated_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())",
                    (
                        document_id,
                        uri,
                        idx,
                        chunk_text,
                        int(start),
                        int(end),
                        embedding_model,
                        dim,
                        Vector(embeddings[idx].tolist()),
                    ),
                )
        conn.commit()

    return len(chunks)


def list_documents() -> List[Dict[str, Any]]:
    """List documents tracked in rag_documents."""

    pool = get_pool()
    with pool.connection() as conn:
        _configure_connection(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uri, text_sha256, updated_at FROM rag_documents ORDER BY updated_at DESC"
            )
            rows = cur.fetchall() or []

    out: List[Dict[str, Any]] = []
    for uri, text_sha256, updated_at in rows:
        out.append(
            {
                "uri": str(uri),
                "text_sha256": str(text_sha256) if text_sha256 is not None else None,
                "updated_at": str(updated_at) if updated_at is not None else None,
            }
        )
    return out


def delete_documents(uris: Sequence[str], embedding_model: Optional[str] = None) -> int:
    """Delete documents (and cascade chunks). Returns deleted count."""

    if not uris:
        return 0

    pool = get_pool()
    with pool.connection() as conn:
        _configure_connection(conn)
        with conn.cursor() as cur:
            if embedding_model:
                # Delete only chunks for a model, leaving doc rows
                cur.execute(
                    "DELETE FROM rag_chunks WHERE uri = ANY(%s) AND embedding_model=%s",
                    (list(uris), embedding_model),
                )
                deleted = cur.rowcount
            else:
                cur.execute("DELETE FROM rag_documents WHERE uri = ANY(%s)", (list(uris),))
                deleted = cur.rowcount
        conn.commit()

    return int(deleted or 0)


def wipe_all() -> None:
    """Delete all stored docs/chunks."""

    pool = get_pool()
    with pool.connection() as conn:
        _configure_connection(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE rag_chunks RESTART IDENTITY CASCADE")
            cur.execute("TRUNCATE rag_documents RESTART IDENTITY CASCADE")
        conn.commit()


def search_chunks(
    *,
    query_embedding: np.ndarray,
    k: int,
    embedding_model: str,
) -> List[Dict[str, Any]]:
    """Search top-k chunks using normalized inner product (via vector_ip_ops)."""

    if query_embedding.ndim == 2:
        if query_embedding.shape[0] != 1:
            raise ValueError("query_embedding must be 1D or shape (1, dim)")
        query_embedding = query_embedding[0]
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be 1D")

    pool = get_pool()
    with pool.connection() as conn:
        _configure_connection(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uri, chunk_text, start_offset, end_offset, - (embedding <#> %s) AS score "
                "FROM rag_chunks "
                "WHERE embedding_model=%s "
                "ORDER BY embedding <#> %s "
                "LIMIT %s",
                (
                    Vector(query_embedding.tolist()),
                    embedding_model,
                    Vector(query_embedding.tolist()),
                    int(k),
                ),
            )
            rows = cur.fetchall() or []

    results: List[Dict[str, Any]] = []
    for uri, chunk_text, start, end, score in rows:
        results.append(
            {
                "uri": str(uri),
                "text": str(chunk_text),
                "start": int(start) if start is not None else None,
                "end": int(end) if end is not None else None,
                "score": float(score) if score is not None else 0.0,
            }
        )
    return results


def stats(*, embedding_model: Optional[str] = None) -> Dict[str, Any]:
    """Return basic stats for health and UI."""

    pool = get_pool()
    with pool.connection() as conn:
        _configure_connection(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rag_documents")
            docs = int(cur.fetchone()[0])
            if embedding_model:
                cur.execute(
                    "SELECT COUNT(*) FROM rag_chunks WHERE embedding_model=%s",
                    (embedding_model,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM rag_chunks")
            chunks = int(cur.fetchone()[0])

            if embedding_model:
                cur.execute(
                    "SELECT MAX(embedding_dim) FROM rag_chunks WHERE embedding_model=%s",
                    (embedding_model,),
                )
            else:
                cur.execute("SELECT MAX(embedding_dim) FROM rag_chunks")
            dim_row = cur.fetchone()
            embed_dim = int(dim_row[0]) if dim_row and dim_row[0] is not None else 0

    return {"documents": docs, "chunks": chunks, "embedding_dim": embed_dim}


def insert_performance_metric(
    operation: str,
    duration_ms: float,
    error: Optional[str] = None,
    model: Optional[str] = None,
    tokens: Optional[int] = None,
    token_count: Optional[int] = None,
) -> None:
    """
    Persist a performance metric row.

    token_count is accepted for backward compatibility with call sites that
    passed that name; it wins only when `tokens` is not provided.
    """
    tokens_value = tokens if tokens is not None else token_count

    pool = get_pool()
    try:
        with pool.connection() as conn:
            _configure_connection(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO performance_metrics(operation, duration_ms, error, model, tokens) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        str(operation),
                        float(duration_ms),
                        str(error) if error is not None else None,
                        str(model) if model is not None else None,
                        int(tokens_value) if tokens_value is not None else None,
                    ),
                )
            conn.commit()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Let caller decide how to handle; redact potentially sensitive details.
        raise RuntimeError(redact_error_message(str(exc))) from exc


def get_performance_metrics(
    hours: int = 24,
) -> List[Dict[str, Any]]:
    """
    Retrieve performance metrics from the last N hours, newest first.
    """
    if hours <= 0:
        hours = 24

    pool = get_pool()
    try:
        with pool.connection() as conn:
            _configure_connection(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT created_at, operation, duration_ms, error, model, tokens "
                    "FROM performance_metrics "
                    "WHERE created_at >= NOW() - INTERVAL %s "
                    "ORDER BY created_at DESC "
                    "LIMIT 1000",
                    (f"{int(hours)} hours",),
                )
                rows = cur.fetchall() or []
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(redact_error_message(str(exc))) from exc

    results: List[Dict[str, Any]] = []
    for created_at, operation, duration_ms, error, model, tokens in rows:
        results.append(
            {
                "created_at": str(created_at),
                "operation": str(operation),
                "duration_ms": float(duration_ms),
                "error": str(error) if error is not None else None,
                "model": str(model) if model is not None else None,
                "tokens": int(tokens) if tokens is not None else None,
            }
        )
    return results
