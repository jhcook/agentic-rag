"""Pytest configuration and fixtures."""
import logging

import pytest

from src.core import pgvector_store

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(autouse=True)
def log_test_info(request):
    """Log test start and finish."""
    logging.info('Starting test: %s', request.node.name)
    yield
    logging.info('Finished test: %s', request.node.name)


@pytest.fixture()
def require_pgvector() -> None:
    """Skip tests that require a running pgvector instance."""

    ok, msg = pgvector_store.test_connection()
    if not ok:
        pytest.skip(f"pgvector unavailable: {msg}")
