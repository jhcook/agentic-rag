import psutil
from types import SimpleNamespace

from src.servers.rest_server import (
    CONTROL_DAEMON_DEFAULT_PORT,
    CONTROL_DAEMON_MAX_PORT,
    _candidate_control_daemon_ports,
    _set_control_daemon_port,
    _terminate_control_daemon_on_port,
)


class FakeProcess:
    def __init__(self, pid: int, cmdline: list[str], terminate_calls: list[int], kill_calls: list[int]):
        self.pid = pid
        self._cmdline = cmdline
        self._terminate_calls = terminate_calls
        self._kill_calls = kill_calls

    def cmdline(self) -> list[str]:
        return self._cmdline

    def terminate(self) -> None:
        self._terminate_calls.append(self.pid)

    def wait(self, timeout: float = 0.0) -> None:
        return

    def kill(self) -> None:
        self._kill_calls.append(self.pid)


class FakeConn(SimpleNamespace):
    pass


def test_candidates_wrap_around():
    _set_control_daemon_port(CONTROL_DAEMON_DEFAULT_PORT + 5)
    ports = _candidate_control_daemon_ports()
    assert ports[0] == CONTROL_DAEMON_DEFAULT_PORT + 5
    assert CONTROL_DAEMON_MAX_PORT in ports
    assert CONTROL_DAEMON_DEFAULT_PORT in ports


def test_terminate_control_daemon_on_port(monkeypatch):
    terminate_calls: list[int] = []
    kill_calls: list[int] = []

    fake_conn = FakeConn(
        status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(port=8055),
        pid=4242,
    )

    monkeypatch.setattr(psutil, "net_connections", lambda kind=None: [fake_conn])
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: FakeProcess(pid, ["python", "control_daemon.py"], terminate_calls, kill_calls),
    )

    _terminate_control_daemon_on_port(8055)

    assert terminate_calls == [4242]
    assert not kill_calls


def test_terminate_control_daemon_on_port_skips(monkeypatch):
    terminate_calls: list[int] = []
    kill_calls: list[int] = []

    fake_conn = FakeConn(
        status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(port=9000),
        pid=4243,
    )

    monkeypatch.setattr(psutil, "net_connections", lambda kind=None: [fake_conn])
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: FakeProcess(pid, ["python", "other_process.py"], terminate_calls, kill_calls),
    )

    _terminate_control_daemon_on_port(8055)

    assert terminate_calls == []
    assert kill_calls == []
