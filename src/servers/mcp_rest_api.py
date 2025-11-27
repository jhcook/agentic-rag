"""
REST API shim for MCP server.
"""
from pathlib import Path
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from src.core.rag_core import (
    get_store,
    upsert_document,
    index_path,
    search,
    save_store,
    rebuild_faiss_index,
    get_faiss_globals,
    DB_PATH,
    OLLAMA_API_BASE,
)
from src.servers.mcp_server import get_memory_usage, MAX_MEMORY_MB, refresh_prometheus_metrics

rest_api = FastAPI(title="mcp-rest-shim")

rest_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@rest_api.post("/upsert_document")
async def rest_upsert_document(req: Request):
    """Upsert a document."""
    body = await req.json()
    uri = body.get("uri", "")
    text = body.get("text", "")
    try:
        result = upsert_document(uri, text)
        return JSONResponse(result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/index_path")
async def rest_index_path(req: Request):
    """Index a directory path."""
    body = await req.json()
    path = body.get("path", "")
    glob = body.get("glob", "**/*.txt")
    try:
        result = index_path(path, glob)
        return JSONResponse(result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/search")
async def rest_search(req: Request):
    """Search documents."""
    body = await req.json()
    query = body.get("query", "")
    try:
        result = search(query)
        return JSONResponse(result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.get("/documents")
async def rest_documents():
    """List documents."""
    try:
        store = get_store()
        docs = []
        for uri, text in getattr(store, "docs", {}).items():
            try:
                size = len(str(text).encode("utf-8", errors="ignore"))
            except Exception:  # pylint: disable=broad-exception-caught
                size = 0
            docs.append({"uri": uri, "size_bytes": size})
        return JSONResponse({"documents": docs})
    except Exception:  # pylint: disable=broad-exception-caught
        return JSONResponse({"documents": []})


@rest_api.post("/documents/delete")
async def rest_documents_delete(req: Request):
    """Delete documents."""
    try:
        body = await req.json()
        uris = body.get("uris", [])
        store = get_store()
        deleted = 0
        for uri in uris:
            if uri in store.docs:
                del store.docs[uri]
                deleted += 1
        save_store()
        rebuild_faiss_index()
        refresh_prometheus_metrics(OLLAMA_API_BASE)
        return JSONResponse({"deleted": deleted})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.post("/flush_cache")
async def rest_flush_cache():
    """Flush the document cache."""
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
        rebuild_faiss_index()
        refresh_prometheus_metrics(OLLAMA_API_BASE)
        return JSONResponse({
            "status": "flushed",
            "db_removed": removed,
            "documents": len(store.docs)
        })
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({"error": str(exc)}, status_code=500)


@rest_api.get("/health")
async def rest_health():
    """Health check endpoint."""
    try:
        store = get_store()
        index, _, _ = get_faiss_globals()
        docs = len(getattr(store, "docs", {}))
        vectors = index.ntotal if index is not None else 0  # type: ignore[attr-defined]
        total_size = sum(
            len(str(t).encode("utf-8", errors="ignore"))
            for t in getattr(store, "docs", {}).values()
        )
        return JSONResponse({
            "status": "ok",
            "documents": docs,
            "vectors": vectors,
            "memory_mb": get_memory_usage(),
            "memory_limit_mb": MAX_MEMORY_MB,
            "total_size_bytes": total_size
        })
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return JSONResponse({"error": str(exc)}, status_code=500)
