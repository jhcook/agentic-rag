#!/usr/bin/env python3
"""
Mini control daemon to start/stop/restart the stack without relying on the REST server.
Run on demand (e.g., python src/servers/control_daemon.py --port 8055 --env .env).
"""

from __future__ import annotations
import argparse
import errno
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_CONTROL_LOG = LOG_DIR / "system_control.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "control_daemon.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("control-daemon")


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def _install_signal_logging() -> None:
    def _make_handler(prev, label: str):
        def handler(signum, frame):
            logger.info("Control daemon received signal %s (%d); shutting down", label, signum)
            if prev not in (None, signal.SIG_DFL, signal.SIG_IGN, handler):
                prev(signum, frame)
        return handler

    for sig in (signal.SIGINT, signal.SIGTERM):
        prev = signal.getsignal(sig)
        signal.signal(sig, _make_handler(prev, _signal_name(sig)))


_install_signal_logging()

app = FastAPI(title="agentic-control-daemon")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOCK = threading.Lock()
_STATE = {
    "status": "idle",
    "last_action": None,
    "last_error": None,
    "started_at": None,
}
_DAEMON_TIMERS = {"last_activity": time.monotonic(), "start_time": time.monotonic()}
CONTROL_DAEMON_IDLE_SECONDS = float(os.getenv("CONTROL_DAEMON_IDLE_SECONDS", "60"))
CONTROL_DAEMON_MAX_LIFETIME_SECONDS = float(os.getenv("CONTROL_DAEMON_MAX_LIFETIME_SECONDS", "90"))


class SystemActionReq(BaseModel):
    env_file: str = ".env"


def _append_system_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [control-daemon] {message}\n"
    with open(SYSTEM_CONTROL_LOG, "a", buffering=1, encoding="utf-8") as f:
        f.write(line)


def _run_script(script_name: str, env_file: str) -> int:
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"{script_name} not found at {script_path}")
    _append_system_log(f"Invoking {script_name} with env {env_file}")
    with open(SYSTEM_CONTROL_LOG, "a", buffering=1, encoding="utf-8") as f:
        proc = subprocess.Popen(
            ["/bin/bash", str(script_path), "--env", env_file],
            cwd=str(PROJECT_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        return proc.wait()


def _note_activity() -> None:
    _DAEMON_TIMERS["last_activity"] = time.monotonic()


def _idle_exit_monitor() -> None:
    """Shut down the process once the daemon stays idle or exceeds its max lifetime."""
    while True:
        time.sleep(5)
        now = time.monotonic()
        elapsed = now - _DAEMON_TIMERS["start_time"]
        if elapsed >= CONTROL_DAEMON_MAX_LIFETIME_SECONDS:
            logger.info(
                "Max lifetime (%ss) reached (idle timeout=%ss); exiting control daemon",
                CONTROL_DAEMON_MAX_LIFETIME_SECONDS,
                CONTROL_DAEMON_IDLE_SECONDS,
            )
            os._exit(0)
        with _LOCK:
            status = _STATE["status"]
        if status == "idle" and now - _DAEMON_TIMERS["last_activity"] >= CONTROL_DAEMON_IDLE_SECONDS:
            logger.info("Idle timeout (%ss) reached; exiting control daemon", CONTROL_DAEMON_IDLE_SECONDS)
            os._exit(0)


def _set_state(status: str, action: Optional[str], error: Optional[str], started: Optional[float]):
    with _LOCK:
        _STATE.update(status=status, last_action=action, last_error=error, started_at=started)
        _note_activity()


def _perform(action: str, env_file: str):
    error = None
    _note_activity()
    try:
        if action == "restart":
            _append_system_log("Stopping services...")
            _run_script("stop.sh", env_file)
            _append_system_log("Starting services...")
            _run_script("start.sh", env_file)
        elif action == "start":
            _run_script("start.sh", env_file)
        elif action == "stop":
            _run_script("stop.sh", env_file)
        else:
            raise ValueError(f"Unknown action {action}")
        _append_system_log(f"Action '{action}' completed")
    except Exception as exc:
        error = str(exc)
        _append_system_log(f"Action '{action}' failed: {exc}")
    finally:
        _set_state("idle", action, error, None)


def _queue(action: str, env_file: str):
    with _LOCK:
        if _STATE["status"] == "running":
            raise HTTPException(status_code=409, detail={"error": "busy", "action": _STATE["last_action"]})
        _set_state("running", action, None, time.time())
    threading.Thread(target=_perform, args=(action, env_file), daemon=True).start()
    return {"status": "accepted", "action": action, "log": str(SYSTEM_CONTROL_LOG)}


@app.get("/health")
def health():
    with _LOCK:
        state = dict(_STATE)
    return {"status": "ok", **state}


@app.post("/system/start")
def start_stack(req: SystemActionReq):
    return _queue("start", req.env_file or ".env")


@app.post("/system/stop")
def stop_stack(req: SystemActionReq):
    return _queue("stop", req.env_file or ".env")


@app.post("/system/restart")
def restart_stack(req: SystemActionReq):
    return _queue("restart", req.env_file or ".env")


def _wait_for_port_free(
    host: str, port: int, timeout: float = 10.0, context: Optional[str] = None
) -> None:
    """Try to bind to the requested port until it's free, otherwise raise."""
    label = context or f"{host}:{port}"
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
            return
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"port {label} still in use after {timeout:.1f}s") from exc
                logger.info("Waiting for %s to become available", label)
                time.sleep(0.5)
                continue
            if exc.errno == errno.EPERM:
                logger.warning("Permission denied checking port %s; assuming it is available", label)
                return
            raise


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG control daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("CONTROL_DAEMON_PORT", "8055")))
    args = parser.parse_args()
    logger.info("Starting control daemon on %s:%s", args.host, args.port)
    try:
        _wait_for_port_free(args.host, args.port)
    except RuntimeError as exc:
        logger.error("Unable to acquire port before startup: %s", exc)
        raise SystemExit(1) from exc
    threading.Thread(target=_idle_exit_monitor, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
