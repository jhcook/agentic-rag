"""
Lightweight retrieval heuristics: reranking, low-signal filtering, synthesis.
"""

from __future__ import annotations
from typing import Dict, List, Any

SIGNATURE_PHRASES = (
    "regards",
    "warm regards",
    "best,",
    "best regards",
    "thanks",
    "thank you",
    "cheers",
    "sincerely",
)


def rerank(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Boost passages containing query terms and penalize short/signature-like text.
    """
    lowered_terms = [t for t in query.lower().split() if t]

    def _score(p: Dict[str, Any]) -> float:
        base = float(p.get("score", 0.0) or 0.0)
        text = str(p.get("text", "")).lower()
        term_hits = sum(text.count(term) for term in lowered_terms) if lowered_terms else 0

        penalty = 0.0
        word_count = len(text.split())
        if word_count < 20:
            penalty += 0.7
        elif word_count < 40:
            penalty += 0.3
        if any(phrase in text for phrase in SIGNATURE_PHRASES):
            penalty += 0.5

        return base + term_hits * 0.1 - penalty

    ranked = sorted(passages, key=_score, reverse=True)
    for p in ranked:
        p["score"] = _score(p)
    return ranked


def synthesize_answer(question: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a deterministic answer by concatenating top passages.
    """
    if not passages:
        return {"answer": "I don't know.", "citations": []}

    top_passages = passages[:3]
    answer_parts = [p.get("text", "") for p in top_passages if p.get("text")]
    answer = " ".join(answer_parts).strip()
    citations = [p.get("uri", "") for p in top_passages if p.get("uri")]
    return {"answer": answer or "I don't know.", "citations": citations}


def is_low_signal(passage: Dict[str, Any]) -> bool:
    """Detect boilerplate/signature passages."""
    text = str(passage.get("text", "")).strip()
    if not text:
        return True
    lowered = text.lower()
    words = lowered.split()
    if len(words) < 12:
        return True
    if any(phrase in lowered for phrase in SIGNATURE_PHRASES):
        return True
    return False
