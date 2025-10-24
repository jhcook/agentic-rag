"""
Core RAG functions: indexing, searching, reranking, synthesizing, verifying.
"""

from __future__ import annotations
from typing import List, Dict, Any
from httpcore import RemoteProtocolError
import logging, pathlib, json, os, hashlib, time, asyncio

from ollama import AsyncClient
from litellm import completion, _turn_on_debug
from litellm.exceptions import APIConnectionError
from litellm.llms.ollama.common_utils import OllamaError
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
                except json.JSONDecodeError as e:
                    logger.warning(f"load_store: {e}")
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
    for _ in range(3):
        try:
            asyncio.run(send_to_llm([text]))
        except APIConnectionError:
            time.sleep(1)
            continue
        break
    save_store()
    return {"upserted": True, "existed": existed}

# -------- Retrieval / Index --------
def index_documents(uris: List[str]) -> Dict[str, Any]:
    """Index a list of document URIs."""
    count = 0
    for uri in uris:
        try:
            text = pathlib.Path(uri).read_text(encoding="utf-8", errors="ignore")
            STORE.add(str(pathlib.Path(uri)), text)
            for _ in range(3):
                try:
                    asyncio.run(send_to_llm([text]))
                    asyncio.get_event_loop().set_debug(True)
                except APIConnectionError:
                    time.sleep(1)
                    continue
                break
            save_store()
            count += 1
        except Exception:
            pass
    return {"indexed": count}

def _collect_files(path: str, glob: str) -> List[pathlib.Path]:
    """Collect files from the given path matching the glob pattern."""
    logger.debug(f"Collecting files from {path} with glob {glob}")
    p = pathlib.Path(path)
    return list(p.rglob(glob))

def _read_and_store_files(files: List[pathlib.Path]) -> List[str]:
    """Read and store files into the STORE."""
    logger.debug(f"Reading and storing {len(files)} files")
    texts = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            STORE.add(str(fp), text)
            texts.append(text)
        except Exception:
            pass
    save_store()
    return texts

def index_path(path: str, glob: str = "**/*.txt"):
    """Index all text files in a given path matching the glob pattern."""
    files = _collect_files(path, glob)
    texts = _read_and_store_files(files)
    resp = None

    # Safely call send_to_llm: if an event loop is already running, schedule the coroutine
    # instead of calling asyncio.run (which raises if a loop is active).
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        logger.info("Detected running asyncio event loop — scheduling send_to_llm without blocking")
        try:
            # Schedule the coroutine on the running loop. The returned object is a Task;
            # callers running in async context can await it. We avoid blocking here.
            resp = running_loop.create_task(send_to_llm(texts))
        except Exception as e:
            logger.error(f"Failed to schedule send_to_llm on running loop: {e}")
    else:
        # No running loop: safe to use asyncio.run
        resp = asyncio.run(send_to_llm(texts))

    return resp

async def send_to_llm(query: List[str]) -> str:
    """Send the query to the LLM and return the response."""
    client = AsyncClient(host="http://127.0.0.1:11434")
    messages = [{ "content": f"{text}", "role": "user"} for text in query]
    try:
        resp = await client.chat(
            model="llama3.2:3b",
            messages = messages
        )
        return resp
    except (ValueError, APIConnectionError, RemoteProtocolError) as e:
        logger.debug(f"send_to_llm: {e}")
        raise

def send_store_to_llm():
    """Send STORE to LLM for processing, waiting for completion whether an
    event loop is running or not.
    """
    logger.debug("Loading STORE")
    resp = None
    texts = [text for _, text in STORE.docs.items()]

    for _ in range(3):
        try:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop and running_loop.is_running():
                # Schedule the coroutine on the running loop and block until it finishes.
                # This is thread-safe and returns a concurrent.futures.Future.
                future = asyncio.run_coroutine_threadsafe(send_to_llm(texts), running_loop)
                resp = future.result(timeout=120)  # wait for completion (or raise)
            else:
                # No running loop: asyncio.run blocks until the coroutine completes.
                resp = asyncio.run(send_to_llm(texts))

        except APIConnectionError:
            time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"send_store_to_llm failed: {e}")
            raise
        break

    return resp

def search(query: str) -> List[Dict[str, Any]]:
    """Query the LLM."""
    logger.info(f"Searching for query: {query}")
    for _ in range(3):
        try:
            resp = completion(
                        model="ollama/llama3.2:3b",
                        messages = [{"content": f"{query}","role": "user"}],
                        api_base="http://localhost:11434",
                        stream=False, timeout=120)
            return resp
        except (ValueError, OllamaError) as e:
            logger.error(f"Ollama API Error: {e}")
