"""Agent Client Protocol (ACP) server over stdio.

Implements the baseline flow from https://agentclientprotocol.com :
  initialize -> session/new -> session/prompt, with session/update
  notifications (agent_message_chunk, tool_call, tool_call_update), a
  session/request_permission round trip before side-effecting tools, and
  session/cancel (a notification) ending the turn with stopReason "cancelled".

Not implemented: authenticate, session/load, MCP server passthrough, client fs/terminal
capabilities, plans, modes. Messages are newline-delimited JSON-RPC 2.0.
"""
from __future__ import annotations

import itertools
import json
import threading
import uuid
from pathlib import Path
from typing import Callable, TextIO

from .agent import Agent
from .capabilities import Capability, CapabilitySet
from .journal import DurableRun, SqliteJournal
from .types import AgentObserver, ToolCall, ToolResult

PROTOCOL_VERSION = 1
UNKNOWN_SESSION = -32002


class _Session:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.cancel = threading.Event()
        self.turns = itertools.count()


class _AcpObserver(AgentObserver):
    def __init__(self, server: "AcpServer", session_id: str):
        self.server, self.sid = server, session_id

    def _update(self, update: dict) -> None:
        self.server.notify("session/update", {"sessionId": self.sid, "update": update})

    def on_text(self, text: str) -> None:
        self._update({"sessionUpdate": "agent_message_chunk", "messageId": f"msg_{uuid.uuid4().hex[:8]}",
                      "content": {"type": "text", "text": text}})

    def on_tool_start(self, call: ToolCall, kind: str) -> None:
        self._update({"sessionUpdate": "tool_call", "toolCallId": call.id, "name": call.name,
                      "title": call.name, "kind": kind, "status": "pending", "rawInput": call.args})

    def approve(self, call: ToolCall) -> bool:
        outcome = self.server.request("session/request_permission", {
            "sessionId": self.sid, "toolCall": {"toolCallId": call.id},
            "options": [{"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                        {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"}]})
        out = (outcome or {}).get("outcome", {})
        return out.get("outcome") == "selected" and out.get("optionId") == "allow-once"

    def on_tool_running(self, call: ToolCall) -> None:
        self._update({"sessionUpdate": "tool_call_update", "toolCallId": call.id, "status": "in_progress"})

    def on_tool_end(self, call: ToolCall, result: ToolResult) -> None:
        self._update({"sessionUpdate": "tool_call_update", "toolCallId": call.id,
                      "status": "completed" if result.ok else "failed",
                      "content": [{"type": "content", "content": {"type": "text", "text": result.output}}]})


class AcpServer:
    def __init__(self, build_agent: Callable[[Path, AgentObserver, threading.Event], Agent],
                 stdin: TextIO, stdout: TextIO, journal: SqliteJournal | None = None):
        self.build_agent, self.stdin, self.stdout = build_agent, stdin, stdout
        self.journal = journal or SqliteJournal()
        self.sessions: dict[str, _Session] = {}
        self._out_lock = threading.Lock()
        self._ids = itertools.count(1000)
        self._pending: dict[int, tuple[threading.Event, list]] = {}
        self._workers: list[threading.Thread] = []

    # --- transport -----------------------------------------------------
    def _send(self, obj: dict) -> None:
        with self._out_lock:
            self.stdout.write(json.dumps(obj) + "\n")
            self.stdout.flush()

    def notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict) -> dict | None:
        rid = next(self._ids)
        done, slot = threading.Event(), []
        self._pending[rid] = (done, slot)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        done.wait()
        return slot[0] if slot else None

    def _reply(self, rid, result: dict) -> None:
        self._send({"jsonrpc": "2.0", "id": rid, "result": result})

    def _error(self, rid, code: int, message: str) -> None:
        self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})

    # --- dispatch ------------------------------------------------------
    def serve(self) -> None:
        for line in self.stdin:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._error(None, -32700, "parse error")
                continue
            if "method" not in msg:  # response to a request we made
                pending = self._pending.pop(msg.get("id"), None)
                if pending:
                    pending[1].append(msg.get("result"))
                    pending[0].set()
            elif "id" in msg:
                self._on_request(msg)
            else:
                self._on_notification(msg)
        for w in self._workers:
            w.join()

    def _on_request(self, msg: dict) -> None:
        rid, method, params = msg["id"], msg["method"], msg.get("params") or {}
        if method == "initialize":
            self._reply(rid, {"protocolVersion": PROTOCOL_VERSION, "authMethods": [],
                              "agentInfo": {"name": "agent-harness-lab", "title": "Agent Harness Lab", "version": "0.1.0"},
                              "agentCapabilities": {"loadSession": False, "promptCapabilities": {
                                  "image": False, "audio": False, "embeddedContext": False}}})
        elif method == "session/new":
            sid = f"sess_{uuid.uuid4().hex[:12]}"
            self.sessions[sid] = _Session(Path(params["cwd"]))
            self._reply(rid, {"sessionId": sid})
        elif method == "session/prompt":
            sess = self.sessions.get(params.get("sessionId", ""))
            if sess is None:
                self._error(rid, UNKNOWN_SESSION, "unknown session")
                return
            worker = threading.Thread(target=self._run_prompt, args=(rid, params["sessionId"], sess, params), daemon=True)
            self._workers.append(worker)
            worker.start()
        else:
            self._error(rid, -32601, f"method not found: {method}")

    def _on_notification(self, msg: dict) -> None:
        if msg["method"] == "session/cancel":
            sess = self.sessions.get((msg.get("params") or {}).get("sessionId", ""))
            if sess:
                sess.cancel.set()

    def _run_prompt(self, rid, sid: str, sess: _Session, params: dict) -> None:
        text = "\n".join(b.get("text", b.get("uri", "")) for b in params.get("prompt", [])
                         if b.get("type") in ("text", "resource_link"))
        sess.cancel.clear()
        try:
            agent = self.build_agent(sess.workspace, _AcpObserver(self, sid), sess.cancel)
            run_id = f"{sid}:{next(sess.turns)}"
            result = agent.run(text, DurableRun(self.journal, run_id))
            self._reply(rid, {"stopReason": result.stop_reason})
        except Exception as e:  # surface as a JSON-RPC error rather than killing the server
            self._error(rid, -32603, f"{type(e).__name__}: {e}")


def default_caps(workspace: Path) -> CapabilitySet:
    ws = str(workspace.resolve())
    return CapabilitySet([Capability("fs.read", ws), Capability("fs.write", ws), Capability("exec", "python")])
