#!/usr/bin/env python3
"""
A simple standalone indexer to debug and fix the indexing issue.
This bypasses the complex lazy loading and tests the core functionality.
"""

import json
import pathlib
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment
load_dotenv()

def simple_index_docs() -> Dict[str, Any]:
    """Simple indexer that just reads and stores docs without complex embedding."""
    docs_dir = pathlib.Path("documents")
    if not docs_dir.exists():
        return {"error": "documents directory not found", "indexed": 0}

    txt_files = list(docs_dir.glob("**/*.txt"))
    if not txt_files:
        return {"error": "no txt files found in documents", "indexed": 0}

    # Simple storage
    cache_dir = pathlib.Path("cache")
    cache_dir.mkdir(exist_ok=True)
    store_file = cache_dir / "simple_store.json"

    docs = {}
    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs[str(file_path)] = {
                "uri": str(file_path),
                "text": text,
                "length": len(text)
            }
            print(f"Indexed: {file_path}")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Error reading {file_path}: {exc}")

    # Save to simple store
    with open(store_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(docs)} documents to {store_file}")
    return {"indexed": len(docs), "files": list(docs.keys()), "store_file": str(store_file)}

if __name__ == "__main__":
    result = simple_index_docs()
    print("Result:", json.dumps(result, indent=2))
