from __future__ import annotations

from pathlib import Path
import re


def test_start_sh_skip_ollama_does_not_disable_ollama_backend() -> None:
    """Regression test: `--skip-ollama` must not disable Ollama Cloud usage."""

    repo_root = Path(__file__).resolve().parents[1]
    start_sh = repo_root / "start.sh"
    text = start_sh.read_text(encoding="utf-8")

    # Sanity check that the flag exists.
    assert "--skip-ollama" in text

    # Critical: skipping local Ollama startup must not disable the Ollama backend,
    # otherwise Ollama Cloud cannot be used.
    assert re.search(r"^\s*export\s+DISABLE_OLLAMA_BACKEND\b", text, re.MULTILINE) is None
