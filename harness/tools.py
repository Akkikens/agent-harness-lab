"""Tools and the registry that gates them on capabilities.

Each tool is one small class that declares (1) the capabilities a call needs,
and (2) what it does. It knows nothing about sessions, models or the journal.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlparse

from .capabilities import Capability, CapabilityDenied, CapabilitySet
from .sandbox import Sandbox
from .secrets import SecretBroker
from .types import ToolCall, ToolResult

MAX_OUTPUT = 4000


@dataclass
class ToolContext:
    workspace: Path
    sandbox: Sandbox
    broker: SecretBroker
    idempotency_key: str = ""

    def resolve(self, rel: str) -> Path:
        # Resolve once and use the same path for the capability check and the
        # I/O, so a symlink swapped in between cannot redirect the access.
        return Path(os.path.realpath(self.workspace / rel))


class Tool(Protocol):
    name: str
    description: str
    schema: dict
    kind: str  # ACP tool kind: read | edit | execute | fetch
    side_effects: bool

    def required(self, args: dict, ctx: ToolContext) -> list[Capability]: ...
    def run(self, args: dict, ctx: ToolContext) -> str: ...


class ReadFile:
    name, kind, side_effects = "read_file", "read", False
    description = "Read a UTF-8 text file from the workspace."
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def required(self, args, ctx):
        return [Capability("fs.read", str(ctx.resolve(args["path"])))]

    def run(self, args, ctx):
        return ctx.resolve(args["path"]).read_text()


class WriteFile:
    name, kind, side_effects = "write_file", "edit", True
    description = "Write (overwrite) a UTF-8 text file in the workspace. Idempotent."
    schema = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
              "required": ["path", "content"]}

    def required(self, args, ctx):
        return [Capability("fs.write", str(ctx.resolve(args["path"])))]

    def run(self, args, ctx):
        target = ctx.resolve(args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"])
        return f"wrote {len(args['content'])} bytes to {args['path']}"


class RunPython:
    name, kind, side_effects = "run_python", "execute", True
    description = "Run Python in an isolated sandbox (no network, read-only unless writes=true)."
    schema = {"type": "object", "properties": {"code": {"type": "string"}, "writes": {"type": "boolean"}},
              "required": ["code"]}

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def required(self, args, ctx):
        needs = [Capability("exec", "python")]
        if args.get("writes"):
            needs.append(Capability("fs.write", str(ctx.workspace.resolve())))
        return needs

    def run(self, args, ctx):
        r = ctx.sandbox.run_python(args["code"], ctx.workspace, writable=bool(args.get("writes")),
                                   timeout=self.timeout)
        body = (r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")).strip()
        if r.timed_out:
            raise RuntimeError(f"timed out after {self.timeout}s\n{body}")
        if r.exit_code != 0:
            raise RuntimeError(f"exit {r.exit_code}\n{body}")
        return body


def _urlopen(url: str, headers: dict[str, str]) -> str:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - gated by net capability
        return resp.read().decode("utf-8", "replace")


class HttpGet:
    """Host-side HTTP GET. The bearer token (if any) is injected here, from the
    broker, so the model and any sandboxed code never see it."""

    name, kind, side_effects = "http_get", "fetch", True
    description = "HTTP GET a URL on an allowed host."
    schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

    def __init__(self, secret_key: str | None = None, transport: Callable[[str, dict], str] = _urlopen):
        self.secret_key, self.transport = secret_key, transport

    def required(self, args, ctx):
        return [Capability("net", urlparse(args["url"]).hostname or "")]

    def run(self, args, ctx):
        headers = {}
        if self.secret_key:
            headers["Authorization"] = f"Bearer {ctx.broker.resolve(self.name, self.secret_key)}"
        return self.transport(args["url"], headers)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.schema}
                for t in self._tools.values()]

    @staticmethod
    def _clean(text: str, ctx: ToolContext) -> str:
        """Redact secrets and hide the host's workspace path, so the model never
        sees it and recorded transcripts replay identically from any directory."""
        for root in {str(ctx.workspace), os.path.realpath(ctx.workspace)}:
            text = text.replace(root, "<workspace>")
        return ctx.broker.redact(text)[:MAX_OUTPUT]

    def invoke(self, call: ToolCall, caps: CapabilitySet, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(False, f"unknown tool {call.name!r}")
        try:
            caps.check(tool.required(call.args, ctx))
            return ToolResult(True, self._clean(tool.run(call.args, ctx), ctx))
        except CapabilityDenied as e:
            return ToolResult(False, self._clean(str(e), ctx), denied=True)
        except Exception as e:  # tool failures are results the model can read, not crashes
            return ToolResult(False, self._clean(f"{type(e).__name__}: {e}", ctx))
