"""Pytest configuration and fixtures."""
import logging

import pytest

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(autouse=True)
def log_test_info(request):
    """Log test start and finish."""
    logging.info('Starting test: %s', request.node.name)
    yield
    logging.info('Finished test: %s', request.node.name)
