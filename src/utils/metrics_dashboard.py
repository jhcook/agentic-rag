#!/usr/bin/env python3
"""
Terminal dashboard for MCP server Prometheus metrics with live log tail.

Top half shows metrics (memory bar + aligned values); bottom half tails
log/mcp_server.log. Press Ctrl+C to exit.
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


def render_dashboard(stdscr, url: str, log_path: Path, refresh: int) -> None:
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
        except Exception as exc:
            metrics_lines = [(f"Failed to fetch metrics from {url}: {exc}", 3)]
        else:
            metrics_lines = build_metrics_lines(metrics)

        height, width = stdscr.getmaxyx()
        top_height = max(8, height // 2)
        bottom_height = height - top_height - 1

        stdscr.erase()
        header = f"MCP Metrics Dashboard  (source: {url})"
        updated = f"Updated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        stdscr.addnstr(0, 0, header, width - 1)
        stdscr.addnstr(1, 0, updated, width - 1)
        stdscr.hline(2, 0, "-", width)

        row = 3
        for text, color_id in metrics_lines:
            if row >= top_height:
                break
            stdscr.addnstr(row, 0, text.ljust(width), width - 1, curses.color_pair(color_id))
            row += 1

        divider_row = top_height
        stdscr.hline(divider_row, 0, "-", width)
        log_lines = tail_file(log_path, max(bottom_height - 1, 1))
        for idx, line in enumerate(log_lines):
            if idx >= bottom_height - 1:
                break
            stdscr.addnstr(divider_row + 1 + idx, 0, line.ljust(width), width - 1)

        stdscr.refresh()

        try:
            time.sleep(max(1, refresh))
        except KeyboardInterrupt:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal dashboard for MCP Prometheus metrics.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Metrics endpoint (default: %(default)s)")
    parser.add_argument("--refresh", type=int, default=REFRESH_SECONDS, help="Refresh interval seconds (default: %(default)s)")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="Log file to tail (default: %(default)s)")
    args = parser.parse_args()

    try:
        curses.wrapper(render_dashboard, args.url, args.log_file, args.refresh)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
