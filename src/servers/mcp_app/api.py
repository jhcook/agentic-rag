from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import psutil
import anyio
import json
import time
import uuid

from src.core.rag_core import (
    search,
    get_store,
    _rebuild_faiss_index,
    get_faiss_globals,
    MAX_MEMORY_MB,
    save_store,
    grounded_answer,
    rerank,
    verify_grounding,
)
from src.core.indexer import index_path, upsert_document
from src.core.extractors import _extract_text_from_file
from src.core.store import DB_PATH
from src.servers.mcp_app import worker as worker_mod

rest_api = FastAPI(title="mcp-rest-shim")

rest_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@rest_api.post("/upsert_document")
async def rest_upsert_document(request: Request):
    body = await request.json()
    uri = body.get("uri", "")
    text = body.get("text", "")
    binary_b64 = body.get("binary_base64")
    if not uri:
        return JSONResponse({"error": "uri is required"}, status_code=422)
    if not text and not binary_b64:
        return JSONResponse({"error": "either text or binary_base64 is required"}, status_code=422)
    try:
        job_payload = {"uri": uri, "text": text, "binary_base64": binary_b64}
        job_id = worker_mod.enqueue_job(job_payload, job_type="upsert_document")
        return JSONResponse({"job_id": job_id, "status": "queued"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/index_path")
async def rest_index_path(request: Request):
    body = await request.json()
    path = body.get("path", "")
    glob = body.get("glob", "**/*.txt")
    try:
        job_id = worker_mod.enqueue_job({"path": path, "glob": glob}, job_type="index_path")
        return JSONResponse({"job_id": job_id, "status": "queued"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/search")
async def rest_search(request: Request):
    body = await request.json()
    query = body.get("query", "")
    async_mode = body.get("async", False)
    timeout_seconds = int(body.get("timeout_seconds", 300))
    start = time.time()
    try:
        # Async mode: queue and return job id immediately with heartbeat messages
        if async_mode:
            job_id = str(uuid.uuid4())
            state = {"id": job_id, "status": "running", "message": "hold tight", "result": None, "error": None}
            JOBS, JOBS_LOCK = worker_mod.get_search_jobs()
            with JOBS_LOCK:
                JOBS[job_id] = state

            def _run_search():
                messages = [
                    (0, "hold tight"),
                    (30, "this is taking longer than we thought"),
                    (60, "what a mission"),
                ]
                try:
                    result_local = search(query, top_k=5)
                    # Normalize result
                    if hasattr(result_local, "model_dump"):
                        result_local = result_local.model_dump()
                    elif isinstance(result_local, dict):
                        result_local = json.loads(json.dumps(result_local, default=str))
                    else:
                        result_local = {"answer": str(result_local)}
                    with JOBS_LOCK:
                        JOBS[job_id].update({"status": "completed", "message": "done", "result": result_local})
                except Exception as exc:
                    with JOBS_LOCK:
                        JOBS[job_id].update({"status": "failed", "message": str(exc), "error": str(exc)})

            # Background worker thread with heartbeat/timer
            def _worker():
                from threading import Event, Thread
                finished = Event()

                def _search_wrapper():
                    _run_search()
                    finished.set()

                t = Thread(target=_search_wrapper, daemon=True)
                t.start()

                messages = [
                    (0, "hold tight"),
                    (30, "this is taking longer than we thought"),
                    (60, "what a mission"),
                ]
                while not finished.wait(timeout=1):
                    elapsed = int(time.time() - start)
                    msg = None
                    for threshold, text in messages:
                        if elapsed >= threshold:
                            msg = text
                    if msg:
                        with JOBS_LOCK:
                            if JOBS.get(job_id, {}).get("status") == "running":
                                JOBS[job_id]["message"] = msg
                    if elapsed >= timeout_seconds:
                        # Timed out: return passages-only
                        try:
                            passages = search(query, top_k=5, max_context_chars=4000)
                        except Exception as exc:
                            passages = {"error": str(exc)}
                        with JOBS_LOCK:
                            JOBS[job_id].update({
                                "status": "timeout",
                                "message": "ok here's the best we can do for now, try later",
                                "result": passages,
                            })
                        finished.set()
                t.join(timeout=1)

            anyio.to_thread.start_blocking_portal().start_task_soon(_worker)
            return JSONResponse({"job_id": job_id, "status": "queued"})

        # Sync path as before
        result = await anyio.to_thread.run_sync(search, query)
        # Normalize possible ModelResponse/dicts to JSON-serializable payload
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif isinstance(result, dict):
            result = json.loads(json.dumps(result, default=str))
        else:
            result = {"answer": str(result)}
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.post("/grounded_answer")
async def rest_grounded_answer(request: Request):
    body = await request.json()
    question = body.get("question", "")
    k = int(body.get("k", 3))
    try:
        result = await anyio.to_thread.run_sync(grounded_answer, question, k)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.post("/rerank")
async def rest_rerank(request: Request):
    body = await request.json()
    query = body.get("query", "")
    passages = body.get("passages", [])
    try:
        result = await anyio.to_thread.run_sync(rerank, query, passages)
        return JSONResponse({"results": result})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.post("/verify_grounding")
async def rest_verify_grounding(request: Request):
    body = await request.json()
    question = body.get("question", "")
    draft_answer = body.get("draft_answer", "")
    citations = body.get("citations", [])
    try:
        result = await anyio.to_thread.run_sync(verify_grounding, question, draft_answer, citations)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.get("/documents")
async def rest_documents(_request: Request):
    try:
        store = get_store()
        docs = []
        for uri, text in getattr(store, "docs", {}).items():
            try:
                size = len(str(text).encode("utf-8", errors="ignore"))
            except Exception:
                size = 0
            docs.append({"uri": uri, "size_bytes": size})
        return JSONResponse({"documents": docs})
    except Exception:
        return JSONResponse({"documents": []})


@rest_api.post("/documents/delete")
async def rest_documents_delete(request: Request):
    try:
        body = await request.json()
        uris = body.get("uris", [])
        store = get_store()
        deleted = 0
        for uri in uris:
            if uri in store.docs:
                del store.docs[uri]
                deleted += 1
        save_store()
        _rebuild_faiss_index()
        return JSONResponse({"deleted": deleted})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/flush_cache")
async def rest_flush_cache():
    try:
        store = get_store()
        store.docs.clear()
        removed = False
        if DB_PATH and Path(DB_PATH).exists():
            try:
                Path(DB_PATH).unlink()
                removed = True
            except OSError:
                removed = False
        save_store()
        _rebuild_faiss_index()
        return JSONResponse({"status": "flushed", "db_removed": removed, "documents": len(store.docs)})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.get("/health")
async def rest_health():
    try:
        store = get_store()
        index, _, _ = get_faiss_globals()
        docs = len(getattr(store, "docs", {}))
        vectors = index.ntotal if index is not None else 0  # type: ignore[attr-defined]
        if docs > 0 and vectors == 0:
            # Try a single rebuild to repopulate vectors if index is empty
            _rebuild_faiss_index()
            index, _, _ = get_faiss_globals()
            vectors = index.ntotal if index is not None else 0  # type: ignore[attr-defined]
        return JSONResponse({
            "status": "ok",
            "documents": docs,
            "vectors": vectors,
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "memory_limit_mb": MAX_MEMORY_MB,
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.get("/jobs")
async def rest_jobs():
    jobs = worker_mod.get_jobs()
    return JSONResponse({"jobs": list(jobs.values())})
