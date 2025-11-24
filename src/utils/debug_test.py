#!/usr/bin/env python3
"""
Minimal test to identify what's causing the indexing hang
"""
import sys
import os
import time

print(f"Starting debug test at {time.time()}")

# Test 1: Basic imports
try:
    print("Testing basic imports...")
    import numpy as np
    print("✓ numpy imported")
    
    import sentence_transformers
    print("✓ sentence_transformers imported")
    
    from sentence_transformers import SentenceTransformer
    print("✓ SentenceTransformer imported")
    
    print("All imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Test 2: Model loading
try:
    print("Testing model loading...")
    model_name = "Snowflake/snowflake-arctic-embed-xs"
    print(f"Loading model: {model_name}")
    
    # Set offline mode
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    embedder = SentenceTransformer(model_name, device='cpu', model_kwargs={"low_cpu_mem_usage": False})
    print(f"✓ Model loaded successfully. Dimension: {embedder.get_sentence_embedding_dimension()}")
    
except Exception as e:
    print(f"Model loading failed: {e}")
    sys.exit(1)

# Test 3: Simple embedding
try:
    print("Testing simple embedding...")
    test_text = "This is a test sentence."
    
    start_time = time.time()
    embedding = embedder.encode([test_text])
    end_time = time.time()
    
    print(f"✓ Embedding generated in {end_time - start_time:.2f} seconds")
    print(f"Embedding shape: {embedding.shape}")
    
except Exception as e:
    print(f"Embedding failed: {e}")
    sys.exit(1)

# Test 4: File reading
try:
    print("Testing file operations...")
    from pathlib import Path
    
    docs_dir = Path("documents")
    if not docs_dir.exists():
        print(f"documents directory not found")
        sys.exit(1)
    
    txt_files = list(docs_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} txt files")
    
    for txt_file in txt_files:
        content = txt_file.read_text(encoding='utf-8')
        print(f"✓ Read {txt_file.name}: {len(content)} characters")
        
        # Test embedding one file
        if content.strip():
            start_time = time.time()
            file_embedding = embedder.encode([content[:1000]])  # First 1000 chars only
            end_time = time.time()
            print(f"✓ Embedded {txt_file.name} in {end_time - start_time:.2f} seconds")
            break
    
except Exception as e:
    print(f"File operations failed: {e}")
    sys.exit(1)

print(f"All tests passed! Completed at {time.time()}")