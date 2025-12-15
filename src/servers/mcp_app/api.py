from fastapi import FastAPI, Request, Depends
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import psutil
import anyio
import json
import time
import uuid
import logging

from src.core.rag_core import MAX_MEMORY_MB
from src.core.indexer import index_path, upsert_document
from src.core.extractors import extract_text_from_file, extract_text_from_bytes
from src.servers.mcp_app import worker as worker_mod
from src.servers.mcp_app.admin_auth import require_admin_access
from src.core.factory import get_rag_backend
from src.core.models import (
    UpsertReq, IndexPathReq, SearchReq, VectorSearchReq,
    GroundedAnswerReq, RerankReq, VerifyReq, DeleteDocsReq,
    IndexUrlReq, LoggingConfigReq
)

logger = logging.getLogger(__name__)

rest_api = FastAPI(title="mcp-rest-shim")

# Initialize backend
backend = get_rag_backend()

# Note: Middleware is handled by http_app.py since this app is mounted
# Do not add middleware here as it conflicts with Starlette mounting

@rest_api.post("/upsert_document")
async def rest_upsert_document(req: UpsertReq):
    if not req.uri:
        return JSONResponse({"error": "uri is required"}, status_code=422)
    if not req.text and not req.binary_base64:
        return JSONResponse({"error": "either text or binary_base64 is required"}, status_code=422)
    try:
        job_payload = {"uri": req.uri, "text": req.text, "binary_base64": req.binary_base64}
        job_id = worker_mod.enqueue_job(job_payload, job_type="upsert_document")
        return JSONResponse({"job_id": job_id, "status": "queued"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.post("/extract")
async def rest_extract(request: Request):
    """Extract text from an uploaded file (in-memory, no storage)."""
    try:
        form = await request.form()
        file = form.get("file")
        if not file:
            return JSONResponse({"error": "file is required"}, status_code=422)
        
        content = await file.read()
        filename = getattr(file, "filename", "unknown")
        text = extract_text_from_bytes(content, filename)
        return JSONResponse({"text": text, "filename": filename})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)



@rest_api.post("/index_path")
async def rest_index_path(req: IndexPathReq):
    try:
        job_id = worker_mod.enqueue_job({"path": req.path, "glob": req.glob}, job_type="index_path")
        return JSONResponse({"job_id": job_id, "status": "queued"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/search")
async def rest_search(req: SearchReq):
    start = time.time()
    try:
        # Async mode: queue and return job id immediately with heartbeat messages
        if req.async_mode:
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
                    # Build kwargs from request
                    kwargs = {}
                    if req.top_k is not None:
                        kwargs['top_k'] = req.top_k
                    if req.model is not None:
                        kwargs['model'] = req.model
                    if req.temperature is not None:
                        kwargs['temperature'] = req.temperature
                    if req.max_tokens is not None:
                        kwargs['max_tokens'] = req.max_tokens
                    
                    result_local = backend.search(req.query, **kwargs)
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
                    if elapsed >= (req.timeout_seconds or 300):
                        # Timed out: return passages-only
                        try:
                            # Fallback with same parameters but likely will fail or be partial
                            passages = backend.search(req.query, top_k=req.top_k or 5)
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
        # Build kwargs
        kwargs = {}
        if req.top_k is not None:
            kwargs['top_k'] = req.top_k
        if req.model is not None:
            kwargs['model'] = req.model
        if req.temperature is not None:
            kwargs['temperature'] = req.temperature
        if req.max_tokens is not None:
            kwargs['max_tokens'] = req.max_tokens

        result = await anyio.to_thread.run_sync(lambda: backend.search(req.query, **kwargs))
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

@rest_api.post("/vector_search")
async def rest_vector_search(req: VectorSearchReq):
    """Vector search endpoint that returns raw search results without LLM synthesis."""
    try:
        # Import rag_core to access vector_search helper
        from src.core import rag_core
        results = await anyio.to_thread.run_sync(rag_core.vector_search, req.query, req.k)
        return JSONResponse({"results": results})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.post("/grounded_answer")
async def rest_grounded_answer(req: GroundedAnswerReq):
    try:
        # Pass all optional args
        kwargs = {"k": req.k or 3}
        if req.model:
             kwargs["model"] = req.model
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.config:
            kwargs.update(req.config)

        result = await anyio.to_thread.run_sync(lambda: backend.grounded_answer(req.question, **kwargs))
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.post("/rerank")
async def rest_rerank(req: RerankReq):
    try:
        result = await anyio.to_thread.run_sync(backend.rerank, req.query, req.passages)
        return JSONResponse({"results": result})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.post("/verify_grounding")
async def rest_verify_grounding(req: VerifyReq):
    try:
        result = await anyio.to_thread.run_sync(backend.verify_grounding, req.question, req.draft_answer, req.citations)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@rest_api.get("/documents")
async def rest_documents(_request: Request):
    try:
        docs = await anyio.to_thread.run_sync(backend.list_documents)
        return JSONResponse(
            {
                "documents": [
                    {"uri": d.get("uri"), "size_bytes": d.get("size", 0)}
                    for d in (docs or [])
                ]
            }
        )
    except Exception:
        return JSONResponse({"documents": []})


@rest_api.post("/documents/delete")
async def rest_documents_delete(req: DeleteDocsReq, _admin: None = Depends(require_admin_access)):
    try:
        result = await anyio.to_thread.run_sync(backend.delete_documents, req.uris)
        return JSONResponse(result)
    except Exception as exc:
        logger.error("documents/delete failed: %s", exc)
        return JSONResponse({"error": "delete failed"}, status_code=500)


@rest_api.post("/flush_cache")
async def rest_flush_cache(_admin: None = Depends(require_admin_access)):
    try:
        result = await anyio.to_thread.run_sync(backend.flush_cache)
        return JSONResponse(result)
    except Exception as exc:
        logger.error("flush_cache failed: %s", exc)
        return JSONResponse({"error": "flush_cache failed"}, status_code=500)


@rest_api.get("/health")
async def rest_health():
    try:
        stats = await anyio.to_thread.run_sync(backend.get_stats)
        return JSONResponse(stats)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.get("/jobs")
async def rest_jobs():
    jobs = worker_mod.get_jobs()
    return JSONResponse({"jobs": list(jobs.values())})

@rest_api.post("/jobs/{job_id}/cancel")
async def rest_cancel_job(job_id: str):
    """Cancel a queued indexing job."""
    success = worker_mod.cancel_job(job_id)
    if not success:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse({"status": "canceled", "id": job_id})


@rest_api.post("/index_url")
async def rest_index_url(req: IndexUrlReq):
    """Index a remote URL by delegating to MCP server tool."""
    if not req.url and not req.query:
        return JSONResponse({"error": "url is required"}, status_code=422)
    
    try:
        from src.servers.mcp_server import index_url_tool  # pylint: disable=import-outside-toplevel
        result = await anyio.to_thread.run_sync(index_url_tool, req.url, req.doc_id, req.query)
        return JSONResponse(result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/config/logging")
async def rest_update_logging(req: LoggingConfigReq):
    """Update logging level dynamically based on debug mode setting."""
    try:
        from src.servers.mcp_app.logging_config import update_logging_level
        update_logging_level(req.debug_mode)
        
        import logging
        return JSONResponse({
            "status": "updated", 
            "debug_mode": req.debug_mode, 
            "log_level": logging.getLevelName(logging.DEBUG if req.debug_mode else logging.INFO)
        })
    except Exception as exc:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Failed to update logging level: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
