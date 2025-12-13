#!/usr/bin/env python3
"""
Test script for OpenAI Assistants backend with local search function calling.

Prerequisites:
1. Set OPENAI_API_KEY in .env
2. Have some documents indexed in local pgvector
3. Set RAG_MODE=openai_assistants in .env

Usage:
    python test_openai_assistants.py
"""

import os
import sys
import traceback

from dotenv import load_dotenv
from openai import OpenAIError

from src.core.openai_assistants_backend import OpenAIAssistantsBackend
import src.core.rag_core as local_core

load_dotenv()


def test_local_search():
    """Test that local search works."""
    print("\n=== Testing Local Search ===")

    # Ensure we have some documents
    store = local_core.get_store()
    doc_count = len(store.docs)
    print(f"Documents in local store: {doc_count}")

    if doc_count == 0:
        print("⚠️  No documents found. Please index some documents first.")
        print("Example: python -c 'from src.core import rag_core; rag_core.index_path(\"./docs\")'")
        return False

    # Test local search
    results = local_core.search("What is RAG?", top_k=3)
    print(f"Local search returned {len(results.get('results', []))} results")

    if results.get('results'):
        print("\nSample result:")
        print(f"  URI: {results['results'][0]['uri']}")
        print(f"  Score: {results['results'][0]['score']:.3f}")
        print(f"  Text preview: {results['results'][0]['text'][:100]}...")

    return True


def test_openai_assistants():
    """Test OpenAI Assistants backend."""
    print("\n=== Testing OpenAI Assistants Backend ===")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env")
        print("Get your key from: https://platform.openai.com/api-keys")
        return False

    print("✅ OPENAI_API_KEY found")

    # Initialize backend
    try:
        print("\nInitializing OpenAI Assistants backend...")
        backend = OpenAIAssistantsBackend()
        print("✅ Backend initialized")
        print(f"   Assistant ID: {backend.assistant_id}")
        print(f"   Model: {backend.model}")
    except (OpenAIError, ValueError, RuntimeError) as exc:
        print(f"❌ Failed to initialize backend: {exc}")
        traceback.print_exc()
        return False

    # Test the search function directly
    print("\n--- Testing search_documents function ---")
    try:
        result = backend.search_documents_function("What is RAG?", top_k=3)
        print(f"✅ Search function returned: {str(result)[:200]}...")
    except OpenAIError as exc:
        print(f"❌ Search function failed: {exc}")
        traceback.print_exc()
        return False

    # Test chat with function calling
    print("\n--- Testing chat with function calling ---")
    try:
        messages = [
            {"role": "user", "content": "What is RAG? Please search my documents for information."}
        ]
        print("Sending message to assistant...")
        print("(This will call the local search function automatically)")

        response = backend.chat(messages)

        if "error" in response:
            print(f"❌ Chat failed: {response['error']}")
            return False

        print("\n✅ Assistant response:")
        print(f"\n{response['content']}\n")

        if response.get('sources'):
            print("📚 Sources cited:")
            for source in response['sources']:
                print(f"   - {source}")

        return True

    except (OpenAIError, RuntimeError, ValueError) as exc:
        print(f"❌ Chat failed: {exc}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("OpenAI Assistants Backend Test Suite")
    print("=" * 70)

    # Test local search first
    if not test_local_search():
        print("\n❌ Local search test failed. Fix this before testing OpenAI Assistants.")
        sys.exit(1)

    # Test OpenAI Assistants
    if not test_openai_assistants():
        print("\n❌ OpenAI Assistants test failed.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\nYou can now use OpenAI Assistants backend by:")
    print("1. Set RAG_MODE=openai_assistants in .env")
    print("2. Restart the server")
    print("3. Chat in the UI - assistant will call local search automatically")


if __name__ == "__main__":
    main()
