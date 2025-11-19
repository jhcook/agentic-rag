import gc
import logging
import os
import sys
import threading
import time
import psutil

logger = logging.getLogger(__name__)

MEMORY_CHECK_INTERVAL = 30  # seconds
MEMORY_LOG_STEP_MB = 256
MEMORY_MONITOR_THREAD = None
LAST_MEMORY_LOG_BUCKET = None


def get_system_memory_mb() -> float:
    """Get total available system memory in MB."""
    return psutil.virtual_memory().available / 1024 / 1024


def get_memory_usage() -> float:
    """Get current process RSS in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_max_memory_mb() -> int:
    """Get the max memory limit from env or 75% of system memory."""
    max_memory_env = os.getenv("MAX_MEMORY_MB")
    if max_memory_env:
        return int(max_memory_env)
    return int(get_system_memory_mb() * 0.75)


MAX_MEMORY_MB = get_max_memory_mb()


def memory_monitor(graceful_shutdown) -> None:
    """Monitor memory usage and trigger cleanup if needed."""
    global LAST_MEMORY_LOG_BUCKET
    consecutive_limit_hits = 0
    while True:
        try:
            memory_mb = get_memory_usage()
            current_bucket = int(memory_mb // MEMORY_LOG_STEP_MB)
            if LAST_MEMORY_LOG_BUCKET is None or current_bucket > LAST_MEMORY_LOG_BUCKET:
                LAST_MEMORY_LOG_BUCKET = current_bucket
                logger.info(
                    "Memory usage crossed %dMB: %.1fMB used (limit %dMB)",
                    current_bucket * MEMORY_LOG_STEP_MB,
                    memory_mb,
                    MAX_MEMORY_MB,
                )
            if memory_mb > MAX_MEMORY_MB:
                consecutive_limit_hits += 1
                logger.warning(
                    "Memory usage %.1fMB exceeds limit %dMB. Triggering garbage collection (strike %d).",
                    memory_mb,
                    MAX_MEMORY_MB,
                    consecutive_limit_hits,
                )
                gc.collect()
                memory_mb = get_memory_usage()
                if memory_mb > MAX_MEMORY_MB * 0.95 and consecutive_limit_hits >= 3:
                    logger.critical(
                        "Memory usage %.1fMB stayed above limit for %d checks. Initiating graceful shutdown.",
                        memory_mb,
                        consecutive_limit_hits,
                    )
                    graceful_shutdown()
                    os._exit(1)
                if memory_mb <= MAX_MEMORY_MB * 0.95:
                    logger.info("Memory usage after GC: %.1fMB", memory_mb)
                    consecutive_limit_hits = 0
            else:
                consecutive_limit_hits = 0
            time.sleep(MEMORY_CHECK_INTERVAL)
        except Exception as exc:
            logger.error("Error in memory monitor: %s", exc)
            time.sleep(MEMORY_CHECK_INTERVAL)


def start_memory_monitor(graceful_shutdown) -> None:
    """Start the memory monitoring thread."""
    global MEMORY_MONITOR_THREAD
    if MEMORY_MONITOR_THREAD is None:
        MEMORY_MONITOR_THREAD = threading.Thread(
            target=memory_monitor, args=(graceful_shutdown,), daemon=True
        )
        MEMORY_MONITOR_THREAD.start()
        logger.info("Started memory monitor. Max memory limit: %dMB", MAX_MEMORY_MB)


def set_memory_limits() -> None:
    """Set OS-level memory limits when supported."""
    if sys.platform == "darwin":
        logger.warning(
            "Skipping RLIMIT_AS enforcement on macOS — rely on memory monitor instead."
        )
        return
    try:
        import resource  # lazy import to avoid errors on non-Unix

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        logger.info("Set RLIMIT_AS soft limit to %d MB", MAX_MEMORY_MB)
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to set memory limits: %s", exc)
