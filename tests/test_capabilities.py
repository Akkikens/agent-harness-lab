import os

import pytest

from harness.capabilities import Capability, CapabilityDenied, CapabilitySet
from harness.sandbox import SandboxResult
from harness.secrets import SecretBroker
from harness.tools import HttpGet, ReadFile, RunPython, ToolContext, ToolRegistry, WriteFile
from harness.types import ToolCall


class NoSandbox:
    def run_python(self, *a, **k):  # pragma: no cover - must never be reached in these tests
        raise AssertionError("sandbox reached despite denied capability")


def ctx(ws, broker=None):
    return ToolContext(ws, NoSandbox(), broker or SecretBroker())


def registry():
    return ToolRegistry([ReadFile(), WriteFile(), RunPython(), HttpGet(secret_key="tok", transport=lambda u, h: f"GET {u} {h}")])


def test_read_inside_scope_allowed(ws):
    (ws / "a.txt").write_text("hi")
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    r = registry().invoke(ToolCall("1", "read_file", {"path": "a.txt"}), caps, ctx(ws))
    assert r.ok and r.output == "hi"


@pytest.mark.parametrize("path", ["../secret.txt", "sub/../../secret.txt", "/etc/hosts"])
def test_path_escape_denied(ws, path):
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    r = registry().invoke(ToolCall("1", "read_file", {"path": path}), caps, ctx(ws))
    assert r.denied and "missing capability" in r.output


def test_symlink_escape_denied(ws, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("top secret")
    os.symlink(outside, ws / "link.txt")
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    r = registry().invoke(ToolCall("1", "read_file", {"path": "link.txt"}), caps, ctx(ws))
    assert r.denied and "top secret" not in r.output


def test_sibling_prefix_is_not_inside_scope(ws, tmp_path):
    # /tmp/x/ws-evil must not match a grant on /tmp/x/ws (string-prefix bug)
    evil = tmp_path / (ws.name + "-evil")
    evil.mkdir()
    (evil / "f.txt").write_text("x")
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    r = registry().invoke(ToolCall("1", "read_file", {"path": f"../{evil.name}/f.txt"}), caps, ctx(ws))
    assert r.denied


def test_write_needs_write_capability_not_read(ws):
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    r = registry().invoke(ToolCall("1", "write_file", {"path": "o.txt", "content": "x"}), caps, ctx(ws))
    assert r.denied and not (ws / "o.txt").exists()


def test_exec_denied_before_sandbox_is_touched(ws):
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    r = registry().invoke(ToolCall("1", "run_python", {"code": "print(1)"}), caps, ctx(ws))
    assert r.denied  # NoSandbox would have raised AssertionError otherwise


def test_run_python_with_writes_also_needs_fs_write(ws):
    caps = CapabilitySet([Capability("exec", "python")])
    r = registry().invoke(ToolCall("1", "run_python", {"code": "x", "writes": True}), caps, ctx(ws))
    assert r.denied and "fs.write" in r.output


def test_attenuate_narrows_but_never_widens(ws):
    (ws / "sub").mkdir()
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    narrow = caps.attenuate(Capability("fs.read", str(ws / "sub")))
    assert narrow.allows(Capability("fs.read", str(ws / "sub" / "f")))
    assert not narrow.allows(Capability("fs.read", str(ws / "other")))
    with pytest.raises(CapabilityDenied):
        narrow.attenuate(Capability("fs.read", str(ws)))
    with pytest.raises(CapabilityDenied):
        caps.attenuate(Capability("net", "example.com"))


def test_net_scope_is_per_host(ws):
    caps = CapabilitySet([Capability("net", "api.example.com")])
    ok = registry().invoke(ToolCall("1", "http_get", {"url": "https://api.example.com/x"}), caps,
                           ctx(ws, _broker_with_grant()))
    bad = registry().invoke(ToolCall("2", "http_get", {"url": "https://evil.test/x"}), caps, ctx(ws, _broker_with_grant()))
    assert ok.ok and bad.denied


def _broker_with_grant():
    b = SecretBroker({"tok": "s3cr3t-token-value"})
    b.grant("http_get", "tok")
    return b


def test_secret_injected_host_side_and_redacted_from_output(ws):
    caps = CapabilitySet([Capability("net", "api.example.com")])
    r = registry().invoke(ToolCall("1", "http_get", {"url": "https://api.example.com/x"}), caps,
                          ctx(ws, _broker_with_grant()))
    # the fake transport echoes the headers it was given, i.e. the token reached the tool...
    assert "Authorization" in r.output
    # ...but never survives into what the model sees
    assert "s3cr3t-token-value" not in r.output and "[REDACTED:tok]" in r.output


def test_secret_without_grant_is_denied(ws):
    caps = CapabilitySet([Capability("net", "api.example.com")])
    broker = SecretBroker({"tok": "s3cr3t-token-value"})  # no grant for http_get
    r = registry().invoke(ToolCall("1", "http_get", {"url": "https://api.example.com/x"}), caps, ctx(ws, broker))
    assert not r.ok and "s3cr3t-token-value" not in r.output


def test_denial_messages_hide_the_host_workspace_path(ws):
    caps = CapabilitySet([Capability("fs.read", str(ws))])
    r = registry().invoke(ToolCall("1", "read_file", {"path": "../x"}), caps, ctx(ws))
    assert r.denied and str(ws) not in r.output and os.path.realpath(ws) not in r.output
    assert r.output == "missing capability fs.read for ../x"
