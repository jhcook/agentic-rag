
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanity_check")

# Add src to path
sys.path.append(os.getcwd())

def test_imports():
    logger.info("TEST: Imports")
    try:
        import numpy
        logger.info(f"✓ numpy: {numpy.__version__}")
    except ImportError as e:
        logger.error(f"✗ numpy failed: {e}")
        return False

    try:
        import torch
        logger.info(f"✓ torch: {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ torch failed: {e}")
        return False

    try:
        import httpx
        logger.info(f"✓ httpx: {httpx.__version__}")
    except ImportError as e:
        logger.error(f"✗ httpx failed: {e}")
        return False
        
    try:
        from src.core import rag_core
        logger.info("✓ src.core.rag_core imported")
        
        # Check specific exceptions that were problematic
        try:
            _ = rag_core.OllamaError
            logger.info("✓ rag_core.OllamaError found")
        except AttributeError:
             logger.error("✗ rag_core.OllamaError NOT found")
             
        try:
            _ = rag_core.APIConnectionError
            logger.info("✓ rag_core.APIConnectionError found")
        except AttributeError:
             logger.error("✗ rag_core.APIConnectionError NOT found")

        try:
            _ = rag_core.Timeout
            logger.info("✓ rag_core.Timeout found")
        except AttributeError:
             logger.error("✗ rag_core.Timeout NOT found")

    except Exception as e:
        logger.error(f"✗ src.core.rag_core import failed: {e}")
        return False

    return True

def test_backend_factory():
    logger.info("\nTEST: Backend Factory")
    try:
        from src.core.factory import get_rag_backend
        backend = get_rag_backend()
        logger.info(f"✓ Backend initialized: {type(backend).__name__}")
        
        modes = backend.get_available_modes()
        logger.info(f"✓ Available modes: {modes}")
        
        # Test imports of modified modules to ensure no broken references
        import src.core.google_backend
        logger.info("✓ src.core.google_backend imported")
        
        import src.core.config_paths
        logger.info("✓ src.core.config_paths imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Backend initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=== STARTING SANITY CHECK ===")
    
    if not test_imports():
        logger.error("Stopping due to import failures")
        sys.exit(1)
        
    if not test_backend_factory():
        logger.error("Stopping due to backend factory failures")
        sys.exit(1)
        
    logger.info("\n=== SANITY CHECK PASSED ===")
