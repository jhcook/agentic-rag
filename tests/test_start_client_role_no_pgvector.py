import sys


def test_client_role_does_not_require_pgvector(monkeypatch, tmp_path):
    import start

    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True, exist_ok=True)

    env_file = tmp_path / "noop.env"
    env_file.write_text("", encoding="utf-8")

    calls = {"pgvector": 0}

    def _fake_ensure_pgvector_running(_env_vars):
        calls["pgvector"] += 1
        return True

    def _fake_sleep(_seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(start, "ensure_pgvector_running", _fake_ensure_pgvector_running)
    monkeypatch.setattr(start.time, "sleep", _fake_sleep)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "start.py",
            "--role",
            "client",
            "--skip-ui",
            "--no-browser",
            "--venv",
            str(venv_dir),
            "--env",
            str(env_file),
        ],
    )

    rc = start.main()
    assert rc == 0
    assert calls["pgvector"] == 0
