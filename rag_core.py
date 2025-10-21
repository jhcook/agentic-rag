"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""

from __future__ import annotations
from collections import Counter
from typing import List, Dict, Any, Optional
import logging, pathlib, json, os, hashlib, time, re

from litellm import completion, _turn_on_debug
_turn_on_debug()

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
        send_store_to_llm()
        logger.info(f"Successfully loaded {len(STORE.docs)} documents")
    except Exception as e:
        logger.error(f"Error loading store: {str(e)}")
        raise

def upsert_document(uri: str, text: str) -> dict:
    """Upsert a single document into the store."""
    existed = uri in STORE.docs
    STORE.add(uri, text)
    send_to_llm(text)
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
            send_to_llm(text)
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
            send_to_llm(text)
            save_store()
            count += 1
        except Exception:
            pass
    return {"indexed": count, "path": str(p), "glob": glob}

def send_to_llm(query: str) -> str:
    """Send the query to the LLM and return the response."""
    logger.debug(f"Document Text (first 100 chars): {query[:100]}")
    try:
        resp = completion(
                    model="ollama/llama3.2:3b",
                    messages = [{ "content": f"{query}","role": "assistant"}],
                    api_base="http://localhost:11434",
                    stream=False,
                    timeout=120)
        try:
            return resp.choices[0].message["content"]
        except (AttributeError, TypeError, KeyError):
            return resp["choices"][0]["message"]["content"]
    except ValueError as e:
        return f"Value Error: {e}"

def send_store_to_llm():
    """Send STORE to LLM for processing."""
    for uri, text in STORE.docs.items():
        logger.debug(f"Loading STORE URI: {uri}")
        send_to_llm(text)

def search(query: str) -> List[Dict[str, Any]]:
    """Query the LLM."""
    logger.info(f"Searching for query: {query}")
    
    try:
    # Query the LLM model
        resp = completion(
                    model="ollama/llama3.2:3b",
                    messages = [{ "content": f"{query}","role": "user"}],
                    api_base="http://localhost:11434",
                    stream=False,
                    timeout=60)
        try:
            return resp.choices[0].message["content"]
        except (AttributeError, TypeError, KeyError):
            return resp["choices"][0]["message"]["content"]
    except ValueError as e:
        return f"Value Error: {e}"
