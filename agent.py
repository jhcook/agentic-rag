#!/usr/bin/env python3
"""

"""

import json, os, sys, requests
from typing import Any, Dict, List

BASE = os.getenv("RAG_BASE", "http://127.0.0.1:8001")

def post(path: str, payload: Dict[str, Any]):
    r = requests.post(f"{BASE}{path}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def index_path(path: str, glob: str="**/*.txt"):
    return post("/api/index_path", {"path": path, "glob": glob})

def search(query: str, k=12, hybrid=True):
    return post("/api/search", {"query": query, "k": k, "hybrid": hybrid})

def rerank(query: str, passages: List[Dict[str, Any]]):
    return post("/api/rerank", {"query": query, "passages": passages})

def grounded_answer(query: str, passages: List[Dict[str, Any]], k=8):
    return post("/api/grounded_answer", {"query": query, "passages": passages, "k": k})

def verify_grounding(query: str, answer: str, citations: List[str]):
    return post("/api/verify_grounding", {"query": query, "answer": answer, "citations": citations})

def score_retriever_conf(passages: List[Dict[str, Any]]) -> float:
    if not passages:
        return 0.0
    s0 = float(passages[0].get("score", 0.0))
    s1 = float(passages[1].get("score", 0.0)) if len(passages) > 1 else 0.0
    return min(1.0, 0.5 + 0.5 * (s0 / max(1e-6, s0 + s1 + 1)))

def control_loop(q: str, idx: str | None=None):
    if idx:
        index_path(idx)

    cands = search(q, k=12, hybrid=True)
    ranked = rerank(q, cands)
    rconf = score_retriever_conf(ranked)

    if rconf < 0.65 and " " in q:
        # naive subqueries
        parts = [p.strip() for p in q.replace(" vs ", " and ").split(" and ")]
        for subq in parts[:3]:
            ranked += search(subq, k=6, hybrid=True)
        ranked = rerank(q, ranked)

    topk = ranked[:8]
    draft = grounded_answer(q, topk, k=8)
    verdict = verify_grounding(q, draft.get("answer", ""), draft.get("citations", []))

    if verdict.get("answer_conf", 0.0) >= 0.70 and verdict.get("citation_coverage", 0.0) >= 0.80:
        return {"final": draft, "verdict": verdict, "iterations": 0}

    # iterate once
    refined = q + ". Clarify: " + "; ".join(verdict.get("missing_facts", []))
    c2 = search(refined, k=12, hybrid=True)
    r2 = rerank(refined, c2)
    t2 = r2[:8]
    d2 = grounded_answer(refined, t2, k=8)
    v2 = verify_grounding(refined, d2.get("answer", ""), d2.get("citations", []))
    return {"final": d2, "verdict": v2, "iterations": 1}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: agent.py 'your question' [optional_index_path]", file=sys.stderr)
        sys.exit(2)
    q = sys.argv[1]
    idx = sys.argv[2] if len(sys.argv) > 2 else None
    res = control_loop(q, idx)
    print(json.dumps(res, indent=2))
