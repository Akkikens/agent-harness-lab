"""Offline replay evals with a regression gate.

Cases pair a task with a recorded cassette (model turns). The model is
replayed, but tools run for real against a throwaway workspace, so the eval
catches regressions in the harness, the tools, the capability policy and the
prompts/tool schemas (which show up as cassette drift) with no network and no
API cost. `gate` compares against a stored baseline and returns a CI exit code.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .agent import Agent
from .capabilities import Capability, CapabilitySet
from .journal import DurableRun, SqliteJournal
from .models import CassetteDrift, ReplayModel
from .sandbox import Sandbox
from .tools import HttpGet, ReadFile, RunPython, ToolRegistry, WriteFile


@dataclass
class CaseResult:
    id: str
    status: str  # pass | fail | skip
    reason: str = ""


def default_registry() -> ToolRegistry:
    return ToolRegistry([ReadFile(), WriteFile(), RunPython(), HttpGet()])


def load_cases(path: Path) -> list[dict]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    return [json.loads(f.read_text()) for f in files]


def run_case(case: dict, base: Path, sandbox: Sandbox | None) -> CaseResult:
    if case.get("needs_sandbox") and sandbox is None:
        return CaseResult(case["id"], "skip", "needs a sandbox (Docker not available)")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp).resolve()
        for name, content in case.get("files", {}).items():
            (ws / name).parent.mkdir(parents=True, exist_ok=True)
            (ws / name).write_text(content)
        caps = CapabilitySet(Capability.parse(c.replace("{WS}", str(ws))) for c in case["capabilities"])
        agent = Agent(ReplayModel(base / case["cassette"]), default_registry(), caps,
                      workspace=ws, sandbox=sandbox)  # type: ignore[arg-type]
        try:
            res = agent.run(case["task"], DurableRun(SqliteJournal(), case["id"]))
        except CassetteDrift as e:
            return CaseResult(case["id"], "fail", f"cassette drift: {e}")
        expect = case.get("expect", {})
        problems = []
        for needle in expect.get("final_contains", []):
            if needle not in res.final_text:
                problems.append(f"final text missing {needle!r}")
        for rel, needle in expect.get("files", {}).items():
            p = ws / rel
            if not p.exists() or needle not in p.read_text():
                problems.append(f"{rel} missing or lacks {needle!r}")
        if "tools" in expect and res.tool_calls != expect["tools"]:
            problems.append(f"tool calls {res.tool_calls} != {expect['tools']}")
        if "denied" in expect and res.denied != expect["denied"]:
            problems.append(f"denied {res.denied} != {expect['denied']}")
        return CaseResult(case["id"], "fail" if problems else "pass", "; ".join(problems))


def gate(results: list[CaseResult], baseline: dict | None) -> tuple[bool, list[str]]:
    """Fail if any case that passed in the baseline no longer passes; also fail
    on any outright failure when there is no baseline."""
    notes: list[str] = []
    ran = {r.id: r for r in results}
    if baseline is None:
        bad = [r for r in results if r.status == "fail"]
        return (not bad, [f"FAIL {r.id}: {r.reason}" for r in bad])
    ok = True
    for case_id in baseline.get("passed", []):
        r = ran.get(case_id)
        if r is None:
            ok = False
            notes.append(f"REGRESSION {case_id}: case missing")
        elif r.status == "fail":
            ok = False
            notes.append(f"REGRESSION {case_id}: {r.reason}")
        elif r.status == "skip":
            notes.append(f"SKIPPED    {case_id}: {r.reason} (not counted as regression)")
    for r in results:
        if r.status == "pass" and r.id not in baseline.get("passed", []):
            notes.append(f"NEW PASS   {r.id}")
    return ok, notes
