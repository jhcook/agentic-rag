"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""

from __future__ import annotations
from collections import Counter
from typing import List, Dict, Any, Optional
import logging, pathlib, json, os, hashlib, time, re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------- In-memory store (toy) --------
class Store:
    """A simple in-memory document store."""
    def __init__(self):
        self.docs: Dict[str, str] = {}
        self.index: List[Dict[str, Any]] = []
        logger.info("Initialized new Store instance")

    def add(self, uri: str, text: str):
        """Add a document to the store and index."""
        logger.debug(f"Adding document: {uri}")
        self.docs[uri] = text
        
        # Clear existing index entries for this URI
        self.index = [idx for idx in self.index if idx["uri"] != uri]
        
        # Create index entry with preprocessed terms
        terms = {word.lower() for word in text.split() 
                if word.isalpha() and len(word) > 1}
        
        # Add new index entry
        index_entry = {
            "uri": uri,
            "text": text,
            "terms": terms
        }
        self.index.append(index_entry)
        
        logger.debug(f"Added document {uri} with {len(terms)} terms: {terms}")
        logger.debug(f"Store now contains {len(self.docs)} documents")

    def hybrid_search(self, query: str, k: int = 12) -> List[Dict[str, Any]]:
        """A naive hybrid search: counts term overlaps."""
        query_terms = {t.lower() for t in query.split() 
                      if t.isalpha() and len(t) > 1}
        logger.debug(f"Searching for terms: {query_terms}")
        
        if not query_terms:
            logger.warning("No valid search terms found in query")
            return []
        
        scored = []
        for doc in self.index:
            doc_terms = doc.get("terms", set())  # Get terms safely
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                scored.append({
                    "uri": doc["uri"],
                    "text": doc["text"],
                    "score": float(overlap) / len(query_terms)
                })
                logger.debug(f"Match found in {doc['uri']} with score {float(overlap) / len(query_terms)}")
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        logger.debug(f"Found {len(scored)} matching documents")
        return scored[:k]

STORE = Store()

# -------- Persist store --------
DB_PATH = os.getenv("RAG_DB", "./rag_store.jsonl")

def _hash_uri(uri: str) -> str:
    return hashlib.sha1(uri.encode()).hexdigest()

def save_store():
    """Save the store to disk."""
    try:
        logger.info(f"Saving store to {DB_PATH}")
        with open(DB_PATH, "w", encoding="utf-8") as f:
            for uri, text in STORE.docs.items():
                rec = {
                    "uri": uri,
                    "id": _hash_uri(uri),
                    "text": text,
                    "ts": int(time.time())
                }
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Successfully saved {len(STORE.docs)} documents")
    except Exception as e:
        logger.error(f"Error saving store: {str(e)}")
        raise

def load_store():
    """Load the store from disk."""
    global STORE
    if not os.path.exists(DB_PATH):
        logger.warning(f"Store file not found at {DB_PATH}")
        return
    
    try:
        logger.info(f"Loading store from {DB_PATH}")
        new_store = Store()  # Create fresh store
        with open(DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if "uri" in rec and "text" in rec:
                        new_store.add(rec["uri"], rec["text"])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
                    continue
        STORE = new_store  # Only update global after successful load
        logger.info(f"Successfully loaded {len(STORE.docs)} documents")
    except Exception as e:
        logger.error(f"Error loading store: {str(e)}")
        raise

def upsert_document(uri: str, text: str) -> dict:
    """Upsert a single document into the store."""
    existed = uri in STORE.docs
    STORE.add(uri, text)
    save_store()
    return {"upserted": True, "existed": existed}

# -------- Retrieval / Rerank / Synthesis / Verify --------
def index_documents(uris: List[str]) -> Dict[str, Any]:
    """Index a list of document URIs."""
    count = 0
    for uri in uris:
        try:
            text = pathlib.Path(uri).read_text(encoding="utf-8", errors="ignore")
            STORE.add(str(pathlib.Path(uri)), text)
            save_store()
            count += 1
        except Exception:
            pass
    return {"indexed": count}

def index_path(path: str, glob: str = "**/*.txt") -> Dict[str, Any]:
    """Index all text files in a given path matching the glob pattern."""
    p = pathlib.Path(path)
    files = list(p.rglob(glob))
    count = 0
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            STORE.add(str(fp), text)
            save_store()
            count += 1
        except Exception:
            pass
    return {"indexed": count, "path": str(p), "glob": glob}

def search(query: str, k: int = 12, hybrid: bool = True) -> List[Dict[str, Any]]:
    """Search for passages relevant to the query."""
    # Only return hits that contain the query string
    # hits = [h for h in STORE.hybrid_search(query, k) if query.lower() in h["text"].lower()]
    if not hybrid:
        return []
    hits = STORE.hybrid_search(query, k)
    return [{"text": h["text"], "score": h["score"], "uri": h["uri"], "meta": {}} for h in hits]

def preprocess(text: str) -> List[str]:
    """Lowercase, remove punctuation, and tokenize."""
    return re.findall(r'\b\w+\b', text.lower())

def relevance_score(query_tokens: List[str], passage_tokens: List[str]) -> float:
    """Score relevance based on token frequency overlap."""
    passage_counter = Counter(passage_tokens)
    score = sum(passage_counter[token] for token in query_tokens)
    return score / len(query_tokens) if query_tokens else 0.0

# def rerank_crossencoder(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     q_terms = {t for t in query.lower().split() if t.isalpha()}
#     ranked: List[Dict[str, Any]] = []
#     for p in passages:
#         p_terms = {t for t in p["text"].lower().split() if t.isalpha()}
#         overlap = len(q_terms & p_terms) / max(1, len(q_terms))
#         short_bonus = 0.05 if len(p["text"]) < 400 else 0.0
#         ranked.append({**p, "score": float(p.get("score", 0.0)) + overlap + short_bonus})
#     ranked.sort(key=lambda x: x["score"], reverse=True)
#     return ranked

def rerank_crossencoder(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank passages based on relevance to the query using token frequency heuristic."""
    if isinstance(passages, str):
        passages = [{"text": passages, "score": 0.0}]

    query_tokens = preprocess(query)
    ranked: List[Dict[str, Any]] = []

    for p in passages:
        passage_tokens = preprocess(p.get("text", ""))
        score = relevance_score(query_tokens, passage_tokens)
        short_bonus = 0.05 if len(p.get("text", "")) < 400 else 0.0
        total_score = float(p.get("score", 0.0)) + score + short_bonus
        ranked.append({**p, "score": total_score})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    print(ranked)
    return ranked

def rerank(query: str, passages: List[Dict[str, Any]], model: str = "cross-encoder-mini") -> List[Dict[str, Any]]:
    """Rerank passages based on their relevance to the query."""
    return rerank_crossencoder(query, passages)

def synthesize_answer(query: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synthesize an answer based on the query and passages."""
    ctx = "\n\n".join(f"[{i+1}] " + p["text"][:300] for i, p in enumerate(passages))
    answer = (
        f"Draft based on {len(passages)} passages.\n\n"
        f"Question: {query}\n\nEvidence:\n{ctx}\n\n"
        f"Answer (summarized; cite like [1], [2]): ..."
    )
    cites = [p["uri"] for p in passages]
    return {"answer": answer, "citations": cites}

def grounded_answer(query: str, passages: Optional[List[Dict[str, Any]]] = None, k: int = 8) -> Dict[str, Any]:
    """Generate a grounded answer based on the query and passages."""
    if not passages:
        passages = search(query, k=k)
        passages = rerank(query, passages)
        passages = passages[:k]
    return synthesize_answer(query, passages)

def verify_grounding_simple(query: str, answer: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify the grounding of the answer based on the passages."""
    cited_idxs = {i for i in range(len(passages)) if f"[{i+1}]" in answer}
    citation_coverage = len(cited_idxs) / max(1, len(passages))
    q_terms = {t for t in query.lower().split() if t.isalpha()}
    cited_text = "\n".join(passages[i]["text"] for i in cited_idxs) if cited_idxs else ""
    p_terms = {t for t in cited_text.lower().split() if t.isalpha()}
    relevance = len(q_terms & p_terms) / max(1, len(q_terms))
    raw_conf = 0.6 * citation_coverage + 0.4 * relevance
    answer_conf = round(0.4 + 0.6 * raw_conf, 2)
    missing = []
    if citation_coverage < 0.6:
        missing.append("Increase specific citations")
    if relevance < 0.5:
        missing.append("Retrieved evidence may not match the query well; refine sub-queries")
    return {
        "answer_conf": answer_conf,
        "citation_coverage": round(citation_coverage, 2),
        "missing_facts": missing,
    }

def verify_grounding(query: str, answer: str, citations: Optional[List[str]] = None) -> Dict[str, Any]:
    """Verify the grounding of the answer based on the citations."""
    passages = [{"uri": uri, "text": STORE.docs.get(uri, ""), "score": 1.0, "meta": {}} for uri in (citations or [])]
    return verify_grounding_simple(query, answer, passages)
