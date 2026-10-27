"""Drives the real `harness acp` subprocess over stdio, like an ACP client would."""
import json
import queue
import subprocess
import sys
import threading

import pytest

TURNS = [
    {"text": "writing the file", "tool_calls": [{"id": "call_1", "name": "write_file",
                                                  "args": {"path": "hello.txt", "content": "hi"}}]},
    {"text": "done"},
]


class Client:
    def __init__(self, tmp_path, turns):
        script = tmp_path / "script.json"
        script.write_text(json.dumps(turns))
        self.p = subprocess.Popen(
            [sys.executable, "-m", "harness.cli", "acp", "--script", str(script), "--weak-sandbox",
             "--db", str(tmp_path / "j.db")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._id = 0

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(json.loads(line))

    def send(self, obj):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", **obj}) + "\n")
        self.p.stdin.flush()

    def call(self, method, params, on_server_request=None):
        """Send a request; collect notifications; answer server requests; return (result|error, updates)."""
        self._id += 1
        rid = self._id
        self.send({"id": rid, "method": method, "params": params})
        updates = []
        while True:
            msg = self.q.get(timeout=30)
            if msg.get("id") == rid and "method" not in msg:
                return msg, updates
            if msg.get("method") == "session/update":
                updates.append(msg["params"]["update"])
            elif "id" in msg and "method" in msg:  # agent -> client request
                updates.append({"sessionUpdate": f"REQUEST:{msg['method']}", **msg["params"]})
                self.send({"id": msg["id"], "result": on_server_request(msg)})

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=15)


@pytest.fixture
def client_factory(tmp_path):
    made = []

    def make(turns=TURNS):
        c = Client(tmp_path, turns)
        made.append(c)
        return c
    yield make
    for c in made:
        try:
            c.close()
        except Exception:
            c.p.kill()


def start_session(c, ws):
    init, _ = c.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
    sess, _ = c.call("session/new", {"cwd": str(ws), "mcpServers": []})
    return init["result"], sess["result"]["sessionId"]


def allow(_msg):
    return {"outcome": {"outcome": "selected", "optionId": "allow-once"}}


def reject(_msg):
    return {"outcome": {"outcome": "selected", "optionId": "reject-once"}}


def test_initialize_negotiates_version_and_capabilities(client_factory, ws):
    init, sid = start_session(client_factory(), ws)
    assert init["protocolVersion"] == 1
    assert init["agentCapabilities"]["loadSession"] is False
    assert init["agentInfo"]["name"] == "agent-harness-lab" and sid.startswith("sess_")


def test_prompt_turn_with_permission_granted(client_factory, ws):
    c = client_factory()
    _, sid = start_session(c, ws)
    resp, updates = c.call("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "make hello.txt"}]}, allow)
    assert resp["result"] == {"stopReason": "end_turn"}
    kinds = [u["sessionUpdate"] for u in updates]
    assert kinds == ["agent_message_chunk", "tool_call", "REQUEST:session/request_permission",
                     "tool_call_update", "tool_call_update", "agent_message_chunk"]
    tool_call = updates[1]
    assert (tool_call["toolCallId"], tool_call["kind"], tool_call["status"]) == ("call_1", "edit", "pending")
    assert [u["status"] for u in updates if u["sessionUpdate"] == "tool_call_update"] == ["in_progress", "completed"]
    assert (ws / "hello.txt").read_text() == "hi"


def test_permission_rejected_blocks_the_tool(client_factory, ws):
    c = client_factory()
    _, sid = start_session(c, ws)
    resp, updates = c.call("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "x"}]}, reject)
    assert resp["result"]["stopReason"] == "end_turn"
    final = [u for u in updates if u["sessionUpdate"] == "tool_call_update"][-1]
    assert final["status"] == "failed" and "rejected by user" in final["content"][0]["content"]["text"]
    assert not (ws / "hello.txt").exists()


def test_cancel_notification_ends_turn_with_cancelled(client_factory, ws):
    turns = TURNS[:1] + [{"text": "second", "tool_calls": [{"id": "call_2", "name": "read_file", "args": {"path": "hello.txt"}}]}, {"text": "end"}]
    c = client_factory(turns)
    _, sid = start_session(c, ws)

    def cancel_then_allow(msg):
        c.send({"method": "session/cancel", "params": {"sessionId": sid}})
        return allow(msg)

    resp, _ = c.call("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "x"}]}, cancel_then_allow)
    assert resp["result"] == {"stopReason": "cancelled"}


def test_unknown_session_and_method_errors(client_factory, ws):
    c = client_factory()
    start_session(c, ws)
    bad, _ = c.call("session/prompt", {"sessionId": "nope", "prompt": []})
    assert bad["error"]["code"] == -32002
    unk, _ = c.call("session/teleport", {})
    assert unk["error"]["code"] == -32601
