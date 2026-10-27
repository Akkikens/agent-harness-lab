"""Model backends. All satisfy types.ModelBackend, so the loop can't tell them apart.

  ScriptedModel   - fixed turns; used by tests and the demo
  RecordingModel  - decorator: wraps any backend and writes a cassette
  ReplayModel     - serves a cassette offline and fails on drift
  AnthropicModel  - Messages API over urllib (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from .types import ModelBackend, ModelTurn, ToolCall


def request_key(messages: list[dict], tools: list[dict]) -> str:
    blob = json.dumps({"messages": messages, "tools": tools}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]):
        self._turns = list(turns)
        self.calls = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "ScriptedModel":
        return cls([ModelTurn.from_dict(t) for t in json.loads(Path(path).read_text())])

    def complete(self, messages, tools):
        # Index by assistant turns already in the transcript, not by call count,
        # so a resumed run (earlier turns replayed from the journal) lines up.
        idx = sum(1 for m in messages if m["role"] == "assistant")
        self.calls += 1
        if idx >= len(self._turns):
            return ModelTurn(text="(script exhausted)")
        return self._turns[idx]


class RecordingModel:
    def __init__(self, inner: ModelBackend, cassette: str | Path):
        self.inner, self.path = inner, Path(cassette)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def complete(self, messages, tools):
        turn = self.inner.complete(messages, tools)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": request_key(messages, tools), "turn": turn.to_dict()}) + "\n")
        return turn


class CassetteDrift(RuntimeError):
    """The agent's request no longer matches what was recorded."""


class ReplayModel:
    def __init__(self, cassette: str | Path):
        self._entries = [json.loads(line) for line in Path(cassette).read_text().splitlines() if line.strip()]
        self._i = 0

    def complete(self, messages, tools):
        if self._i >= len(self._entries):
            raise CassetteDrift(f"cassette exhausted after {self._i} turns; the agent made an extra model call")
        entry = self._entries[self._i]
        got = request_key(messages, tools)
        if got != entry["key"]:
            raise CassetteDrift(f"turn {self._i}: request {got} != recorded {entry['key']} "
                                "(prompt, tool schema or tool output changed; re-record if intended)")
        self._i += 1
        return ModelTurn.from_dict(entry["turn"])


class AnthropicModel:
    """Not exercised against the live API in this repo's tests (no key in CI)."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-sonnet-5-5", system: str = "", max_tokens: int = 2048,
                 api_key: str | None = None):
        self.model, self.system, self.max_tokens = model, system, max_tokens
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    @staticmethod
    def to_api(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                           for c in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": blocks})
            else:  # tool result; consecutive results share one user message
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": m["content"], "is_error": m.get("is_error", False)}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, messages, tools):
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": self.system,
                "messages": self.to_api(messages), "tools": tools}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(), method="POST", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.load(resp)
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b["input"]) for b in data["content"] if b["type"] == "tool_use"]
        return ModelTurn(text, calls)
