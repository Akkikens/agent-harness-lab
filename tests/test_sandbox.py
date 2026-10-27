"""Real-container tests. Skipped (not faked) when Docker isn't running."""
import os
from pathlib import Path

import pytest

from harness.sandbox import DockerSandbox, SubprocessSandbox, pick_sandbox

pytestmark = pytest.mark.docker


def test_docker_command_has_every_isolation_flag(ws):
    cmd = DockerSandbox().command("n", ws, writable=False)
    joined = " ".join(cmd)
    for flag in ["--network none", "--read-only", "--cap-drop ALL", "no-new-privileges",
                 "--pids-limit", "--memory", "noexec"]:
        assert flag in joined
    user = cmd[cmd.index("--user") + 1]
    assert not user.startswith("0:"), "container must never run as root"
    assert f"{ws}:/work:ro" in joined


def test_docker_command_mounts_rw_only_when_granted(ws):
    assert f"{ws}:/work:rw" in " ".join(DockerSandbox().command("n", ws, writable=True))


def test_pick_sandbox_refuses_to_silently_downgrade(monkeypatch):
    monkeypatch.setattr(DockerSandbox, "daemon_up", staticmethod(lambda: False))
    with pytest.raises(RuntimeError):
        pick_sandbox()
    assert isinstance(pick_sandbox(allow_weak=True), SubprocessSandbox)


def test_runs_code(docker_box, ws):
    r = docker_box.run_python("print(6*7)", ws, writable=False, timeout=30)
    assert (r.exit_code, r.stdout.strip()) == (0, "42")


def test_network_is_blocked(docker_box, ws):
    code = "import socket\ntry:\n  socket.create_connection(('1.1.1.1', 53), 3)\n  print('REACHED')\nexcept OSError as e:\n  print('BLOCKED')"
    assert docker_box.run_python(code, ws, writable=False, timeout=30).stdout.strip() == "BLOCKED"


def test_host_secrets_never_enter_the_sandbox(docker_box, ws, monkeypatch):
    monkeypatch.setenv("HOST_ONLY_SECRET", "hunter2-do-not-leak")
    r = docker_box.run_python("import os; print(dict(os.environ))", ws, writable=False, timeout=30)
    assert "hunter2" not in r.stdout and "HOST_ONLY_SECRET" not in r.stdout


def test_workspace_read_only_by_default(docker_box, ws):
    r = docker_box.run_python("open('/work/x.txt','w').write('x')", ws, writable=False, timeout=30)
    assert r.exit_code != 0 and not (ws / "x.txt").exists()


def test_workspace_writable_when_granted(docker_box, ws):
    r = docker_box.run_python("open('/work/x.txt','w').write('ok')", ws, writable=True, timeout=30)
    assert r.exit_code == 0 and (ws / "x.txt").read_text() == "ok"


def test_cannot_write_outside_workspace_or_read_host_files(docker_box, ws):
    r = docker_box.run_python("open('/etc/pwned','w').write('x')", ws, writable=True, timeout=30)
    assert r.exit_code != 0
    r = docker_box.run_python("print(len(__import__('os').listdir('/Users')) if __import__('os').path.exists('/Users') else 'no /Users')",
                              ws, writable=False, timeout=30)
    assert r.stdout.strip() == "no /Users"


def test_timeout_kills_the_container(docker_box, ws):
    r = docker_box.run_python("while True: pass", ws, writable=False, timeout=3)
    assert r.timed_out
    import subprocess
    assert subprocess.run(["docker", "ps", "-q", "--filter", "name=harness-"], capture_output=True, text=True).stdout.strip() == ""


def test_memory_limit_enforced(docker_box, ws):
    r = docker_box.run_python("x = bytearray(512*1024*1024); print('ALLOCATED')", ws, writable=False, timeout=30)
    assert "ALLOCATED" not in r.stdout
