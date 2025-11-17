#!/usr/bin/env python3
"""
Terminal dashboard for MCP server Prometheus metrics with live log tail.

Top half shows metrics (left: MCP; right: REST). Bottom half tails
MCP and REST logs side-by-side. Press Ctrl+C to exit.
"""

from __future__ import annotations

import argparse
import curses
from pathlib import Path
import sys
import time
from typing import Dict, Any, Tuple, List

import requests
from prometheus_client.parser import text_string_to_metric_families


DEFAULT_URL = "http://localhost:8000/metrics"
DEFAULT_LOG_PATH = Path("log/mcp_server.log")
DEFAULT_REST_URL = "http://localhost:8001/metrics"
DEFAULT_REST_LOG_PATH = Path("log/rest_server.log")
REFRESH_SECONDS = 5
BAR_WIDTH = 30

# Metrics we care about and their friendly labels/units
SIMPLE_GAUGES = {
    "mcp_documents_indexed_total": ("Documents Indexed", ""),
    "mcp_embedding_vectors_total": ("Embedding Vectors", ""),
    "mcp_embedding_chunks_total": ("Embedding Chunks", ""),
    "mcp_embedding_dimension": ("Embedding Dimension", ""),
    "mcp_memory_usage_megabytes": ("Memory Used", "MB"),
    "mcp_memory_limit_megabytes": ("Memory Limit", "MB"),
    "ollama_running_models": ("Ollama Running Models", ""),
    "ollama_available_models": ("Ollama Available Models", ""),
    "ollama_up": ("Ollama Up (0/1)", ""),
}

RUNNING_MODEL_SIZE = "ollama_running_model_size_bytes"
RUNNING_MODEL_VRAM = "ollama_running_model_vram_bytes"
AVAILABLE_MODEL_SIZE = "ollama_model_size_bytes"

REST_GAUGES = {
    "rest_documents_indexed_total": ("REST Documents Indexed", ""),
    "rest_embedding_vectors_total": ("REST Embedding Vectors", ""),
    "rest_embedding_chunks_total": ("REST Embedding Chunks", ""),
    "rest_embedding_dimension": ("REST Embedding Dimension", ""),
    "rest_inflight_requests": ("REST Inflight Requests", ""),
    "rest_memory_usage_megabytes": ("REST Memory Used", "MB"),
}
REST_COUNTERS = {
    "rest_http_requests_total": "REST HTTP Requests",
}
REST_HISTOGRAM_PREFIX = "rest_http_request_duration_seconds"

EMBED_REQUESTS = "embedding_requests_total"
EMBED_ERRORS = "embedding_errors_total"
EMBED_DURATION_PREFIX = "embedding_duration_seconds"


def fetch_metrics(url: str) -> Dict[str, Any]:
    """Scrape Prometheus metrics text and parse into a structured dict."""
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()

    data: Dict[str, Any] = {
        "gauges": {},
        "running_models": {},
        "available_models": {},
    }

    for family in text_string_to_metric_families(resp.text):
        for sample in family.samples:
            name, labels, value = sample.name, sample.labels, sample.value

            if name in SIMPLE_GAUGES:
                data["gauges"][name] = value
                continue

            if name == RUNNING_MODEL_SIZE:
                key = labels.get("model", "unknown")
                data["running_models"].setdefault(key, {})["size_bytes"] = value
            elif name == RUNNING_MODEL_VRAM:
                key = labels.get("model", "unknown")
                data["running_models"].setdefault(key, {})["vram_bytes"] = value
            elif name == AVAILABLE_MODEL_SIZE:
                key = labels.get("model", "unknown")
                data["available_models"].setdefault(key, {})["size_bytes"] = value

    return data


def threshold_ratio(value: float, denom: float) -> float:
    """Return percentage value vs. denominator."""
    if denom <= 0:
        return 0.0
    return (value / denom) * 100.0


def build_bar(value: float, limit: float) -> str:
    """Render a bar chart string for value vs. limit."""
    if limit <= 0:
        return f"{value:.1f} MB"

    val = max(value, 0)
    filled = int(min(BAR_WIDTH, (val / limit) * BAR_WIDTH))
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    pct = (val / limit) * 100
    return f"[{bar}] {val:.1f} / {limit:.1f} MB ({pct:.1f}%)"


def human_bytes(num: float) -> Tuple[float, str]:
    """Convert bytes to a human-readable value/unit pair."""
    units = ["B", "KB", "MB", "GB", "TB"]
    val = num
    for unit in units:
        if abs(val) < 1024 or unit == units[-1]:
            return val, unit
        val /= 1024
    return val, units[-1]


def tail_file(path: Path, max_lines: int) -> List[str]:
    """Return the last max_lines lines from a file."""
    if not path.exists():
        return [f"[log not found: {path}]"]
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines[-max_lines:]]
    except OSError as exc:
        return [f"[error reading log: {exc}]"]


def build_metrics_lines(metrics: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Turn metrics into display lines with optional color pair ids."""
    lines: List[Tuple[str, int]] = []
    gauges = metrics.get("gauges", {})

    mem_limit = gauges.get("mcp_memory_limit_megabytes", 0) or 0
    mem_used = gauges.get("mcp_memory_usage_megabytes", 0) or 0

    if "mcp_memory_usage_megabytes" in gauges:
        bar_text = build_bar(mem_used, mem_limit) if mem_limit else f"{mem_used:.1f} MB"
        ratio = threshold_ratio(mem_used, mem_limit) if mem_limit else 0
        color = 0
        if mem_limit:
            if ratio < 75:
                color = 1
            elif ratio <= 90:
                color = 2
            else:
                color = 3
        lines.append((f"{'Memory Used':<28} {bar_text}", color))

    for key in (
        "mcp_documents_indexed_total",
        "mcp_embedding_vectors_total",
        "mcp_embedding_chunks_total",
        "mcp_embedding_dimension",
        "ollama_running_models",
        "ollama_available_models",
        "ollama_up",
    ):
        if key not in gauges:
            continue
        label, unit = SIMPLE_GAUGES[key]
        value = gauges.get(key, 0) or 0
        suffix = f" {unit}" if unit else ""
        lines.append((f"{label:<28} {value:>12.0f}{suffix}", 0))

    running = metrics.get("running_models", {})
    if running:
        lines.append(("-- Running Models --", 0))
        for model, vals in running.items():
            size_val, size_unit = human_bytes(vals.get("size_bytes", 0) or 0)
            vram_val, vram_unit = human_bytes(vals.get("vram_bytes", 0) or 0)
            lines.append((f"{(model + ' size'):<28} {size_val:>12.2f} {size_unit}", 0))
            lines.append((f"{(model + ' vram'):<28} {vram_val:>12.2f} {vram_unit}", 0))

    available = metrics.get("available_models", {})
    if available:
        lines.append(("-- Available Models --", 0))
        for model, vals in available.items():
            size_val, size_unit = human_bytes(vals.get("size_bytes", 0) or 0)
            lines.append((f"{(model + ' size'):<28} {size_val:>12.2f} {size_unit}", 0))

    return lines


def fetch_rest_metrics(url: str) -> Dict[str, Any]:
    """Scrape REST server metrics and compute aggregates we care about."""
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()

    data: Dict[str, Any] = {
        "gauges": {},
        "requests_total": 0.0,
        "requests_by_status": {},
        "latency_count": 0.0,
        "latency_sum": 0.0,
        "embed_requests": {},
        "embed_errors": {},
        "embed_latency": {},
    }

    for family in text_string_to_metric_families(resp.text):
        for sample in family.samples:
            name, labels, value = sample.name, sample.labels, sample.value

            if name in REST_GAUGES:
                data["gauges"][name] = value
                continue

            if name in REST_COUNTERS:
                status = labels.get("status", "unknown")
                data["requests_by_status"].setdefault(status, 0.0)
                data["requests_by_status"][status] += value
                data["requests_total"] += value
                continue

            if name == f"{REST_HISTOGRAM_PREFIX}_count":
                data["latency_count"] = value
                continue
            if name == f"{REST_HISTOGRAM_PREFIX}_sum":
                data["latency_sum"] = value
                continue

            if name == EMBED_REQUESTS:
                stage = labels.get("stage", "unknown")
                data["embed_requests"][stage] = data["embed_requests"].get(stage, 0.0) + value
                continue
            if name == EMBED_ERRORS:
                stage = labels.get("stage", "unknown")
                data["embed_errors"][stage] = data["embed_errors"].get(stage, 0.0) + value
                continue
            if name == f"{EMBED_DURATION_PREFIX}_count":
                stage = labels.get("stage", "unknown")
                entry = data["embed_latency"].setdefault(stage, {"count": 0.0, "sum": 0.0})
                entry["count"] += value
                continue
            if name == f"{EMBED_DURATION_PREFIX}_sum":
                stage = labels.get("stage", "unknown")
                entry = data["embed_latency"].setdefault(stage, {"count": 0.0, "sum": 0.0})
                entry["sum"] += value
                continue

    return data


def build_rest_lines(metrics: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Format REST metrics for display."""
    lines: List[Tuple[str, int]] = []
    gauges = metrics.get("gauges", {})

    # Memory bar (only usage available)
    mem_used = gauges.get("rest_memory_usage_megabytes")
    if mem_used is not None:
        lines.append((f"{'REST Memory Used':<28} {mem_used:>12.1f} MB", 0))

    for key in (
        "rest_documents_indexed_total",
        "rest_embedding_vectors_total",
        "rest_embedding_chunks_total",
        "rest_embedding_dimension",
        "rest_inflight_requests",
    ):
        if key not in gauges:
            continue
        label, unit = REST_GAUGES[key]
        value = gauges.get(key, 0) or 0
        suffix = f" {unit}" if unit else ""
        lines.append((f"{label:<28} {value:>12.0f}{suffix}", 0))

    requests_total = metrics.get("requests_total", 0) or 0
    if requests_total:
        lines.append((f"{'REST HTTP Requests':<28} {requests_total:>12.0f}", 0))
        by_status = metrics.get("requests_by_status", {})
        if by_status:
            for status in sorted(by_status):
                lines.append((f"  status {status:<17} {by_status[status]:>12.0f}", 0))

    latency_count = metrics.get("latency_count", 0) or 0
    latency_sum = metrics.get("latency_sum", 0) or 0
    if latency_count:
        avg_ms = (latency_sum / latency_count) * 1000
        lines.append((f"{'HTTP Latency avg':<28} {avg_ms:>11.1f} ms", 0))

    # Embedding metrics
    embed_reqs = metrics.get("embed_requests", {})
    embed_errs = metrics.get("embed_errors", {})
    embed_lat = metrics.get("embed_latency", {})
    if embed_reqs:
        total_embed = sum(embed_reqs.values())
        lines.append((f"{'Embedding requests':<28} {total_embed:>12.0f}", 0))
        for stage, val in embed_reqs.items():
            err = embed_errs.get(stage, 0)
            lat = embed_lat.get(stage, {})
            avg_ms = ((lat.get("sum", 0) / lat.get("count", 1)) * 1000) if lat.get("count") else 0
            lines.append((f"  {stage:<22} {val:>12.0f} req", 0))
            if err:
                lines.append((f"    errors{':':<18} {err:>12.0f}", 0))
            if lat.get("count"):
                lines.append((f"    avg {avg_ms:>6.1f} ms", 0))

    return lines


def pad_lines(lines: List[Tuple[str, int]], target: int) -> List[Tuple[str, int]]:
    """Pad a lines list to at least target length with empty strings."""
    if len(lines) >= target:
        return lines
    return lines + [("", 0)] * (target - len(lines))


def render_dashboard(stdscr, url: str, rest_url: str, log_path: Path, rest_log_path: Path, refresh: int) -> None:
    """Main curses loop."""
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)

    while True:
        try:
            metrics = fetch_metrics(url)
            rest_metrics = fetch_rest_metrics(rest_url)
        except Exception as exc:
            metrics_lines = [(f"Failed to fetch metrics from {url}: {exc}", 3)]
            rest_lines = [(f"Failed to fetch REST metrics from {rest_url}: {exc}", 3)]
        else:
            metrics_lines = build_metrics_lines(metrics)
            rest_lines = build_rest_lines(rest_metrics)

        height, width = stdscr.getmaxyx()
        top_height = max(8, height // 2)
        bottom_height = height - top_height - 1
        mid = max(20, width // 2)

        stdscr.erase()
        header = f"MCP Metrics (source: {url})"
        rest_header = f"REST Metrics (source: {rest_url})"
        updated = f"Updated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        stdscr.addnstr(0, 0, header, mid - 1)
        stdscr.addnstr(0, mid + 1, rest_header, width - mid - 2)
        stdscr.addnstr(1, 0, updated, width - 1)
        stdscr.hline(2, 0, "-", width)

        max_lines = top_height - 3
        left_lines = pad_lines(metrics_lines, max_lines)
        right_lines = pad_lines(rest_lines, max_lines)

        stdscr.vline(3, mid, "|", max_lines)
        for idx in range(max_lines):
            row = 3 + idx
            if row >= top_height:
                break
            l_text, l_color = left_lines[idx]
            r_text, r_color = right_lines[idx]
            stdscr.addnstr(row, 0, l_text.ljust(mid - 1), mid - 1, curses.color_pair(l_color))
            stdscr.addnstr(row, mid + 1, r_text.ljust(width - mid - 2), width - mid - 2, curses.color_pair(r_color))

        divider_row = top_height
        stdscr.hline(divider_row, 0, "-", width)
        log_lines_left = tail_file(log_path, max(bottom_height - 1, 1))
        log_lines_right = tail_file(rest_log_path, max(bottom_height - 1, 1))
        max_log_lines = max(len(log_lines_left), len(log_lines_right))
        stdscr.vline(divider_row + 1, mid, "|", bottom_height - 1)
        for idx in range(min(max_log_lines, bottom_height - 1)):
            left = log_lines_left[idx] if idx < len(log_lines_left) else ""
            right = log_lines_right[idx] if idx < len(log_lines_right) else ""
            stdscr.addnstr(divider_row + 1 + idx, 0, left.ljust(mid - 1), mid - 1)
            stdscr.addnstr(divider_row + 1 + idx, mid + 1, right.ljust(width - mid - 2), width - mid - 2)

        stdscr.refresh()

        try:
            time.sleep(max(1, refresh))
        except KeyboardInterrupt:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal dashboard for MCP Prometheus metrics.")
    parser.add_argument("--url", default=DEFAULT_URL, help="MCP metrics endpoint (default: %(default)s)")
    parser.add_argument("--rest-url", default=DEFAULT_REST_URL, help="REST metrics endpoint (default: %(default)s)")
    parser.add_argument("--refresh", type=int, default=REFRESH_SECONDS, help="Refresh interval seconds (default: %(default)s)")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="MCP log file to tail (default: %(default)s)")
    parser.add_argument("--rest-log-file", type=Path, default=DEFAULT_REST_LOG_PATH, help="REST log file to tail (default: %(default)s)")
    args = parser.parse_args()

    try:
        curses.wrapper(render_dashboard, args.url, args.rest_url, args.log_file, args.rest_log_file, args.refresh)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
