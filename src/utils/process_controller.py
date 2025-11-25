#!/usr/bin/env python3
"""
Lightweight controller to manage individual Agentic RAG services.

Responsibilities:
- Pick an available port from a provided or default range.
- Launch the requested service command.
- Enforce a maximum lifetime (default: 60 seconds), terminating the child cleanly.
- Append lifecycle logs to log/process_controller.log.

This controller is intended to be spawned by the REST API service endpoints.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_RANGES: Dict[str, Tuple[int, int]] = {
    "rest": (8001, 8015),
    "mcp": (8000, 8014),
    "ui": (5173, 5185),
    "ollama": (11434, 11444),
}

DEFAULT_HOSTS = {
    "rest": "127.0.0.1",
    "mcp": "127.0.0.1",
    "ui": "127.0.0.1",
    "ollama": "127.0.0.1",
}

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOG_PATH = ROOT_DIR / "log" / "process_controller.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    """Append a timestamped message to the controller log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def is_port_free(host: str, port: int) -> bool:
    """Return True if a port is available on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def pick_port(service: str, host: str, port_range: Optional[Tuple[int, int]]) -> int:
    """Pick the first free port in the provided or default range."""
    rng = port_range or DEFAULT_RANGES.get(service, (0, 0))
    start, end = rng
    if not start or not end or start > end:
        raise RuntimeError(f"Invalid port range for {service}: {rng}")

    for port in range(start, end + 1):
        if is_port_free(host, port):
            return port
    raise RuntimeError(f"No free ports available for {service} in range {start}-{end}")


def build_command(service: str, host: str, port: int) -> Tuple[list[str], Path]:
    """
    Build the launch command and log destination for a service.
    Returns (command_list, log_file_path).
    """
    log_dir = ROOT_DIR / "log"
    log_dir.mkdir(exist_ok=True)

    if service == "rest":
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.servers.rest_server:app",
            "--host",
            host,
            "--port",
            str(port),
            "--no-access-log",
        ]
        log_file = log_dir / "rest_server.log"
    elif service == "mcp":
        cmd = [sys.executable, "-m", "src.servers.mcp_server"]
        log_file = log_dir / "mcp_server.log"
    elif service == "ui":
        # Run the Vite dev server with a specified port/host
        cmd = [
            "npm",
            "--prefix",
            "ui",
            "run",
            "dev",
            "--",
            "--host",
            host,
            "--port",
            str(port),
            "--strictPort",
        ]
        log_file = log_dir / "ui_server.log"
    elif service == "ollama":
        cmd = ["ollama", "serve"]
        log_file = log_dir / "ollama_server.log"
    else:
        raise ValueError(f"Unsupported service '{service}'")

    return cmd, log_file


def launch_service(service: str, host: str, port: int, lifetime: int) -> None:
    """Launch the service, enforce lifetime, and manage shutdown."""
    env = os.environ.copy()

    # Ensure host/port envs propagate to child where applicable
    if service == "rest":
        env["RAG_HOST"] = host
        env["RAG_PORT"] = str(port)
    elif service == "mcp":
        env["MCP_HOST"] = host
        env["MCP_PORT"] = str(port)
    elif service == "ui":
        env["UI_HOST"] = host
        env["UI_PORT"] = str(port)
        env["PORT"] = str(port)
        env["VITE_PORT"] = str(port)
    elif service == "ollama":
        env["OLLAMA_API_BASE"] = f"http://{host}:{port}"

    cmd, service_log = build_command(service, host, port)
    log(f"[controller] Starting {service} on {host}:{port} using: {' '.join(cmd)}")

    with service_log.open("a", encoding="utf-8") as svc_log:
        process = subprocess.Popen(
            cmd,
            stdout=svc_log,
            stderr=svc_log,
            env=env,
            cwd=str(ROOT_DIR),
        )

    def _shutdown_child(sig: int) -> None:
        if process.poll() is None:
            log(f"[controller] Received signal {sig}, stopping {service} (PID {process.pid})")
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass

    # Ensure child is terminated when controller gets a signal
    signal.signal(signal.SIGTERM, lambda _s, _f: _shutdown_child(signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda _s, _f: _shutdown_child(signal.SIGINT))

    deadline = time.time() + lifetime
    exit_reason = "normal"

    try:
        while time.time() < deadline:
            ret = process.poll()
            if ret is not None:
                exit_reason = f"child exited (code {ret})"
                break
            time.sleep(1)
        else:
            exit_reason = "lifetime_exceeded"
    finally:
        if process.poll() is None:
            log(f"[controller] Lifetime reached; stopping {service} (PID {process.pid})")
            _shutdown_child(signal.SIGTERM)

    log(f"[controller] {service} controller exiting: {exit_reason}")


def parse_range(range_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not range_str:
        return None
    parts = range_str.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return (start, end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process controller for Agentic RAG services.")
    parser.add_argument("--service", required=True, choices=["rest", "mcp", "ui", "ollama"])
    parser.add_argument("--host", default=None, help="Host to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    parser.add_argument("--port-range", default=None, help="Optional port range start-end")
    parser.add_argument("--lifetime", type=int, default=60, help="Maximum lifetime in seconds")

    args = parser.parse_args()
    service = args.service
    host = args.host or DEFAULT_HOSTS.get(service, "127.0.0.1")
    port_range = parse_range(args.port_range)

    try:
        port = args.port if args.port else pick_port(service, host, port_range)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log(f"[controller] Failed to choose port for {service}: {exc}")
        sys.exit(1)

    try:
        launch_service(service, host, port, args.lifetime)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        log(f"[controller] Failed to launch {service} on {host}:{port}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
