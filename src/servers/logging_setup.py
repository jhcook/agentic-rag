import os
import logging

def setup_logging():
    os.makedirs('log', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('log/mcp_server.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    access_logger = logging.getLogger('mcp_access')
    access_logger.setLevel(logging.INFO)
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.addHandler(logging.NullHandler())

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers.clear()
    uvicorn_access_logger.propagate = False
    uvicorn_access_logger.disabled = True

    try:
        import uvicorn
        from copy import deepcopy
        patched = deepcopy(getattr(uvicorn.config, "LOGGING_CONFIG", {}))
        if patched and "loggers" in patched and "uvicorn.access" in patched["loggers"]:
            patched["loggers"]["uvicorn.access"]["handlers"] = []
            patched["loggers"]["uvicorn.access"]["propagate"] = False
            patched["loggers"]["uvicorn.access"]["level"] = "WARNING"
            uvicorn.config.LOGGING_CONFIG = patched
        os.environ.setdefault("UVICORN_ACCESS_LOG", "false")
    except Exception as exc:
        logger.debug("Unable to patch uvicorn access logging: %s", exc)

    return logger, access_logger
