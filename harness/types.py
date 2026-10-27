"""Shared value types and the small interfaces the agent loop depends on.

The loop depends on these Protocols, never on a concrete model, journal or
sandbox, so each can be swapped (scripted model in tests, recording model for
evals, Docker vs. fallback sandbox) without touching the loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelTurn":
        return cls(d.get("text", ""), [ToolCall(**c) for c in d.get("tool_calls", [])])


@dataclass
class ToolResult:
    ok: bool
    output: str
    denied: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ModelBackend(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> ModelTurn: ...


class AgentObserver:
    """Hooks for a front end (ACP, CLI). All no-ops by default; override what you need."""

    def on_text(self, text: str) -> None: ...
    def on_tool_start(self, call: ToolCall, kind: str) -> None: ...
    def approve(self, call: ToolCall) -> bool:
        return True
    def on_tool_running(self, call: ToolCall) -> None: ...
    def on_tool_end(self, call: ToolCall, result: ToolResult) -> None: ...
