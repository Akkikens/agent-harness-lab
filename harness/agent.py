"""The agent loop. Composition only: it is handed a model, a tool registry,
a capability set and a durable run, and owns none of their construction."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from .capabilities import CapabilitySet
from .journal import DurableRun
from .sandbox import Sandbox
from .secrets import SecretBroker
from .tools import ToolContext, ToolRegistry
from .types import AgentObserver, ModelBackend, ModelTurn, ToolCall, ToolResult


@dataclass
class RunResult:
    stop_reason: str  # end_turn | max_turn_requests | cancelled
    final_text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    denied: int = 0
    turns: int = 0


class Agent:
    def __init__(self, model: ModelBackend, registry: ToolRegistry, caps: CapabilitySet, *,
                 workspace: Path, sandbox: Sandbox, broker: SecretBroker | None = None,
                 max_turns: int = 8, observer: AgentObserver | None = None,
                 cancel: threading.Event | None = None):
        self.model, self.registry, self.caps = model, registry, caps
        self.workspace, self.sandbox = Path(workspace), sandbox
        self.broker = broker or SecretBroker()
        self.max_turns = max_turns
        self.observer = observer or AgentObserver()
        self.cancel = cancel or threading.Event()

    def run(self, task: str, durable: DurableRun) -> RunResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        result = RunResult("max_turn_requests")
        for turn in range(self.max_turns):
            if self.cancel.is_set():
                result.stop_reason = "cancelled"
                return result
            payload = durable.step("model", f"turn:{turn}", lambda: self.model.complete(
                messages, self.registry.specs()).to_dict())
            model_turn = ModelTurn.from_dict(payload)
            result.turns += 1
            result.final_text = model_turn.text
            if model_turn.text:
                self.observer.on_text(model_turn.text)
            messages.append({"role": "assistant", "text": model_turn.text,
                             "tool_calls": [{"id": c.id, "name": c.name, "args": c.args}
                                            for c in model_turn.tool_calls]})
            if not model_turn.tool_calls:
                result.stop_reason = "end_turn"
                return result
            for call in model_turn.tool_calls:
                if self.cancel.is_set():
                    result.stop_reason = "cancelled"
                    return result
                res = ToolResult(**durable.step("tool", f"{call.name}:{call.id}",
                                                lambda c=call: self._invoke(c, durable).to_dict()))
                result.tool_calls.append(call.name)
                result.denied += res.denied
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": res.output, "is_error": not res.ok})
        return result

    def _invoke(self, call: ToolCall, durable: DurableRun) -> ToolResult:
        tool = self.registry.get(call.name)
        self.observer.on_tool_start(call, tool.kind if tool else "other")
        if tool is not None and tool.side_effects and not self.observer.approve(call):
            res = ToolResult(False, "rejected by user", denied=True)
        else:
            self.observer.on_tool_running(call)
            ctx = ToolContext(self.workspace, self.sandbox, self.broker, durable.idempotency_key)
            res = self.registry.invoke(call, self.caps, ctx)
        self.observer.on_tool_end(call, res)
        return res
