from __future__ import annotations

import base64
import logging
import multiprocessing
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import queue as pyqueue

try:  # pragma: no cover - optional dependency guard
    from requests.exceptions import SSLError as RequestsSSLError  # type: ignore
except Exception:  # pragma: no cover
    class RequestsSSLError(Exception):
        """Fallback SSL error type when requests is unavailable."""

        pass

try:  # pragma: no cover - optional dependency guard
    from urllib3.exceptions import SSLError as Urllib3SSLError  # type: ignore
except Exception:  # pragma: no cover
    class Urllib3SSLError(Exception):
        """Fallback SSL error type when urllib3 is unavailable."""

        pass

from src.core.extractors import extract_text_from_file
from src.core.indexer import index_path, upsert_document

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from multiprocessing import Queue as MPQueue
else:  # pragma: no cover - runtime alias suffices
    MPQueue = multiprocessing.Queue  # type: ignore[attr-defined]

QueueMessage = Dict[str, Any]

JOB_QUEUE: Optional["MPQueue[QueueMessage]"] = None
RESULT_QUEUE: Optional["MPQueue[QueueMessage]"] = None
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
RESULT_THREAD: Optional[threading.Thread] = None
STARTED = False
# Search jobs map and lock (for async search heartbeats)
SEARCH_JOBS: Dict[str, Dict[str, Any]] = {}
SEARCH_JOBS_LOCK = threading.Lock()
CANCELED: set[str] = set()


def _iso_now() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def start_worker(env: Optional[Dict[str, str]] = None):
    """Initialize queues, spawn worker process, and start result listener."""
    global JOB_QUEUE, RESULT_QUEUE, RESULT_THREAD, STARTED
    if STARTED:
        return
    JOB_QUEUE = MPQueue()
    RESULT_QUEUE = MPQueue()
    env = env or {}

    proc = multiprocessing.Process(
        target=_index_worker_loop,
        args=(JOB_QUEUE, RESULT_QUEUE, env),
        daemon=True,
    )
    proc.start()

    def _result_listener():
        while True:
            try:
                msg = RESULT_QUEUE.get()
            except (EOFError, OSError):
                break
            if not msg:
                continue
            job_id = msg.get("id")
            with JOBS_LOCK:
                job = JOBS.get(job_id, {"id": job_id})
                job.update(msg)
                job["last_update"] = _iso_now()
                if msg.get("status") == "failed":
                    error_msg = msg.get("error") or "Indexing job failed"
                    if msg.get("error_type") == "ssl_error":
                        job["notification"] = {
                            "type": "ssl_error",
                            "message": (
                                "Indexing stalled because the embedding model could not be "
                                "downloaded due to an SSL certificate validation failure. "
                                "Update your certificate trust store and retry the upload."
                            ),
                        }
                    else:
                        job["notification"] = {
                            "type": "error",
                            "message": error_msg,
                        }
                JOBS[job_id] = job
            try:
                # No legacy store sync: canonical content is indexed artifacts + pgvector.
                pass
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to sync store after job %s: %s", job_id, exc, exc_info=True)

    RESULT_THREAD = threading.Thread(target=_result_listener, daemon=True)
    RESULT_THREAD.start()
    STARTED = True


def enqueue_job(payload: Dict[str, Any], job_type: str = "upsert_document") -> str:
    """Enqueue a job; starts worker if needed."""
    if not STARTED or JOB_QUEUE is None:
        env_copy = {k: v for k, v in os.environ.items() if k.startswith(("RAG_", "EMBED_", "LLM_", "MCP_", "OLLAMA_"))}
        start_worker(env_copy)
    job_id = str(uuid.uuid4())
    now_iso = _iso_now()
    job = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "queued_at": now_iso,
        "last_update": now_iso,
        **payload,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    if JOB_QUEUE is None:
        raise RuntimeError("Worker queue not initialized")
    JOB_QUEUE.put(job)
    return job_id


def get_jobs() -> Dict[str, Dict[str, Any]]:
    with JOBS_LOCK:
        return dict(JOBS)


def cancel_job(job_id: str) -> bool:
    """Mark a job as canceled if it hasn't started."""
    with JOBS_LOCK:
        if job_id not in JOBS:
            return False
        JOBS[job_id]["status"] = "canceled"
        JOBS[job_id]["last_update"] = _iso_now()
        CANCELED.add(job_id)
    return True


def cancel_all_jobs() -> int:
    """Mark all non-terminal jobs as canceled and drain queued work."""
    canceled = 0
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if job.get("status") in ("completed", "failed", "canceled"):
                continue
            job["status"] = "canceled"
            job["last_update"] = _iso_now()
            CANCELED.add(job_id)
            canceled += 1
    # Drain pending items from the queue to prevent processing
    if JOB_QUEUE is not None:
        while True:
            try:
                queued_job = JOB_QUEUE.get_nowait()
            except (pyqueue.Empty, OSError):
                break
            if not queued_job:
                continue
            jid = queued_job.get("id")
            CANCELED.add(jid)
            with JOBS_LOCK:
                JOBS[jid] = {**queued_job, "status": "canceled", "last_update": _iso_now()}
            if RESULT_QUEUE is not None:
                try:
                    RESULT_QUEUE.put({
                        "id": jid,
                        "status": "canceled",
                        "last_update": _iso_now(),
                        "finished_at": _iso_now(),
                    })
                except Exception:
                    pass
    return canceled


def get_search_jobs():
    return SEARCH_JOBS, SEARCH_JOBS_LOCK


def _index_worker_loop(
    job_queue: "MPQueue[QueueMessage]",
    result_queue: "MPQueue[QueueMessage]",
    env: Dict[str, str],
) -> None:
    """Run indexing jobs in an isolated process to keep the main server responsive."""
    for key, value in env.items():
        os.environ[key] = value
    from dotenv import load_dotenv as _ld
    _ld()
    from src.core.rag_core import (
        index_path as _worker_index_path,
        upsert_document as _worker_upsert,
        rebuild_index as _worker_rebuild,
    )
    while True:
        job = job_queue.get()
        if not job:
            continue
        if job.get("type") == "stop":
            break
        job_id = job.get("id")
        try:
            if job_id in CANCELED:
                finished_at = _iso_now()
                result_queue.put({
                    "id": job_id,
                    "status": "canceled",
                    "last_update": finished_at,
                    "finished_at": finished_at,
                })
                continue
            started_at = _iso_now()
            result_queue.put({
                "id": job_id,
                "status": "running",
                "started_at": started_at,
                "last_update": started_at,
            })
            if job["type"] == "index_path":
                res = _worker_index_path(job["path"], job.get("glob", "**/*.txt"))
            elif job["type"] == "upsert_document":
                uri = job["uri"]
                text = job.get("text") or ""
                binary_b64 = job.get("binary_base64")
                uri_suffix = Path(uri).suffix.lower()
                binary_exts = {".pdf", ".doc", ".docx", ".pages"}
                if binary_b64:
                    try:
                        data = base64.b64decode(binary_b64)
                        suffix = uri_suffix or ".tmp"
                        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
                            tmp.write(data)
                            tmp.flush()
                            extracted = extract_text_from_file(Path(tmp.name))
                            text = extracted or text
                    except Exception as exc:
                        logger.error("Failed to extract text from binary upload %s: %s", uri, exc)
                if not text:
                    if binary_b64 and uri_suffix in binary_exts:
                        finished_at = _iso_now()
                        result_queue.put({
                            "id": job_id,
                            "status": "failed",
                            "error": "no text extracted",
                            "uri": uri,
                            "last_update": finished_at,
                            "finished_at": finished_at,
                        })
                        continue
                    try:
                        if binary_b64:
                            raw = base64.b64decode(binary_b64)
                            text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                if not text:
                    finished_at = _iso_now()
                    result_queue.put({
                        "id": job_id,
                        "status": "failed",
                        "error": "empty or non-text content",
                        "uri": uri,
                        "last_update": finished_at,
                        "finished_at": finished_at,
                    })
                    continue
                res = _worker_upsert(uri, text)
            else:
                res = {"error": f"unknown job type {job['type']}"}
            # _worker_rebuild()  <-- REMOVED: Redundant O(N^2) rebuild. Upsert handles its own persistence.
            finished_at = _iso_now()
            result_queue.put({
                "id": job_id,
                "status": "completed",
                "result": res,
                "last_update": finished_at,
                "finished_at": finished_at,
            })
        except Exception as exc:
            error_msg = str(exc)
            error_type = None
            if isinstance(exc, (RequestsSSLError, Urllib3SSLError)) or "SSL" in error_msg or "CERTIFICATE_VERIFY_FAILED" in error_msg:
                error_type = "ssl_error"
            finished_at = _iso_now()
            result_queue.put({
                "id": job_id,
                "status": "failed",
                "error": error_msg,
                "error_type": error_type,
                "last_update": finished_at,
                "finished_at": finished_at,
            })
