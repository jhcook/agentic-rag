import pytest
import logging

logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(autouse=True)
def log_test_info(request):
    logging.info(f'Starting test: {request.node.name}')
    yield
    logging.info(f'Finished test: {request.node.name}')