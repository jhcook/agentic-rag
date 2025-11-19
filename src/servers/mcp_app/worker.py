import base64
import multiprocessing
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import os

from src.core.extractors import _extract_text_from_file
from src.core.indexer import index_path, upsert_document
from src.core.store import load_store, save_store
from src.core.rag_core import _ensure_store_synced

logger = logging.getLogger(__name__)

JOB_QUEUE: Optional[multiprocessing.Queue] = None
RESULT_QUEUE: Optional[multiprocessing.Queue] = None
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
RESULT_THREAD: Optional[threading.Thread] = None
STARTED = False
# Search jobs map and lock (for async search heartbeats)
SEARCH_JOBS: Dict[str, Dict[str, Any]] = {}
SEARCH_JOBS_LOCK = threading.Lock()


def start_worker(env: Optional[Dict[str, str]] = None):
    """Initialize queues, spawn worker process, and start result listener."""
    global JOB_QUEUE, RESULT_QUEUE, RESULT_THREAD, STARTED
    if STARTED:
        return
    JOB_QUEUE = multiprocessing.Queue()
    RESULT_QUEUE = multiprocessing.Queue()
    env = env or {}

    proc = multiprocessing.Process(
        target=_index_worker_loop, args=(JOB_QUEUE, RESULT_QUEUE, env), daemon=True
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
                JOBS[job_id] = job
            try:
                load_store()
                _rebuild_faiss_index()
                # ensure any external changes are reflected in the current store
                _ensure_store_synced()
                save_store()
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to refresh store after job %s: %s", job_id, exc, exc_info=True)

    RESULT_THREAD = threading.Thread(target=_result_listener, daemon=True)
    RESULT_THREAD.start()
    STARTED = True


def enqueue_job(payload: Dict[str, Any], job_type: str = "upsert_document") -> str:
    """Enqueue a job; starts worker if needed."""
    if not STARTED or JOB_QUEUE is None:
        env_copy = {k: v for k, v in os.environ.items() if k.startswith(("RAG_", "EMBED_", "LLM_", "MCP_", "OLLAMA_"))}
        start_worker(env_copy)
    job_id = str(uuid.uuid4())
    job = {"id": job_id, "type": job_type, **payload, "status": "queued"}
    with JOBS_LOCK:
        JOBS[job_id] = job
    JOB_QUEUE.put(job)  # type: ignore
    return job_id


def get_jobs() -> Dict[str, Dict[str, Any]]:
    with JOBS_LOCK:
        return dict(JOBS)


def get_search_jobs():
    return SEARCH_JOBS, SEARCH_JOBS_LOCK


def _index_worker_loop(job_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, env: Dict[str, str]):
    """Run indexing jobs in an isolated process to keep the main server responsive."""
    for k, v in env.items():
        import os
        os.environ[k] = v
    from dotenv import load_dotenv as _ld
    _ld()
    from src.core.rag_core import (
        index_path as _worker_index_path,
        upsert_document as _worker_upsert,
        load_store as _worker_load_store,
        save_store as _worker_save_store,
        _rebuild_faiss_index as _worker_rebuild,
    )
    while True:
        job = job_queue.get()
        if not job:
            continue
        if job.get("type") == "stop":
            break
        job_id = job.get("id")
        try:
            _worker_load_store()
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
                            extracted = _extract_text_from_file(Path(tmp.name))
                            text = extracted or text
                    except Exception as exc:
                        logger.error("Failed to extract text from binary upload %s: %s", uri, exc)
                if not text:
                    if binary_b64 and uri_suffix in binary_exts:
                        result_queue.put({"id": job_id, "status": "failed", "error": "no text extracted", "uri": uri})
                        continue
                    try:
                        if binary_b64:
                            raw = base64.b64decode(binary_b64)
                            text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                if not text:
                    result_queue.put({"id": job_id, "status": "failed", "error": "empty or non-text content", "uri": uri})
                    continue
                res = _worker_upsert(uri, text)
            else:
                res = {"error": f"unknown job type {job['type']}"}
            _worker_rebuild()
            _worker_save_store()
            result_queue.put({"id": job_id, "status": "completed", "result": res})
        except Exception as exc:
            result_queue.put({"id": job_id, "status": "failed", "error": str(exc)})
