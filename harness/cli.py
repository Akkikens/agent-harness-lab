"""harness run | resume | eval | acp"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .acp import AcpServer, default_caps
from .agent import Agent
from .capabilities import Capability, CapabilitySet
from .evalrun import default_registry, gate, load_cases, run_case
from .journal import DurableRun, SqliteJournal
from .models import AnthropicModel, RecordingModel, ScriptedModel
from .sandbox import DockerSandbox, pick_sandbox


def _model(args):
    if args.script:
        return ScriptedModel.from_json(args.script)
    return AnthropicModel(system="You are a careful engineering agent. Use tools to do the task.")


def _agent(args, workspace: Path) -> Agent:
    model = _model(args)
    if getattr(args, "record", None):
        model = RecordingModel(model, args.record)
    caps = CapabilitySet(Capability.parse(c.replace("{WS}", str(workspace.resolve()))) for c in args.grant) \
        if args.grant else default_caps(workspace)
    return Agent(model, default_registry(), caps, workspace=workspace, sandbox=pick_sandbox(args.weak_sandbox))


def cmd_run(args) -> int:
    ws = Path(args.workspace).resolve()
    journal = SqliteJournal(args.db)
    journal.set_status(args.run_id, "running", args.task)
    run = DurableRun(journal, args.run_id)
    result = _agent(args, ws).run(args.task, run)
    journal.set_status(args.run_id, "done")
    print(json.dumps({"run_id": args.run_id, "stop_reason": result.stop_reason, "turns": result.turns,
                      "tool_calls": result.tool_calls, "replayed_steps": run.replayed,
                      "executed_steps": run.executed, "final": result.final_text}, indent=2))
    return 0


def cmd_resume(args) -> int:
    journal = SqliteJournal(args.db)
    st = journal.status(args.run_id)
    if st is None:
        print(f"unknown run {args.run_id}", file=sys.stderr)
        return 2
    args.task = st[1]
    return cmd_run(args)


def cmd_eval(args) -> int:
    base = Path(args.cases)
    cases = load_cases(base)
    sandbox = None
    if any(c.get("needs_sandbox") for c in cases):
        try:
            sandbox = pick_sandbox(args.weak_sandbox)
        except RuntimeError:
            sandbox = None
    results = [run_case(c, base if base.is_dir() else base.parent, sandbox) for c in cases]
    for r in results:
        print(f"{r.status.upper():5} {r.id}  {r.reason}")
    baseline_path = Path(args.baseline) if args.baseline else None
    baseline = json.loads(baseline_path.read_text()) if baseline_path and baseline_path.exists() else None
    if args.update_baseline and baseline_path:
        baseline_path.write_text(json.dumps({"passed": sorted(r.id for r in results if r.status == "pass")}, indent=2))
        print(f"baseline written to {baseline_path}")
        return 0
    ok, notes = gate(results, baseline)
    for n in notes:
        print(n)
    passed = sum(r.status == "pass" for r in results)
    print(f"{passed}/{len(results)} passed; gate {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def cmd_acp(args) -> int:
    def build(workspace, observer, cancel):
        return Agent(_model(args), default_registry(), default_caps(workspace), workspace=workspace,
                     sandbox=pick_sandbox(args.weak_sandbox), observer=observer, cancel=cancel)
    AcpServer(build, sys.stdin, sys.stdout, SqliteJournal(args.db)).serve()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--script", help="JSON file of scripted model turns (default: Anthropic API)")
        sp.add_argument("--db", default=str(Path(tempfile.gettempdir()) / "harness-journal.db"))
        sp.add_argument("--weak-sandbox", action="store_true", help="allow the non-isolating subprocess fallback")

    r = sub.add_parser("run"); common(r)
    r.add_argument("task"); r.add_argument("--run-id", required=True); r.add_argument("--workspace", default=".")
    r.add_argument("--grant", action="append", help="capability, e.g. fs.read:{WS} (repeatable)")
    r.add_argument("--record", help="write a replay cassette here")
    r.set_defaults(fn=cmd_run)

    rs = sub.add_parser("resume"); common(rs)
    rs.add_argument("run_id"); rs.add_argument("--workspace", default="."); rs.add_argument("--grant", action="append")
    rs.set_defaults(fn=cmd_resume, record=None)

    e = sub.add_parser("eval")
    e.add_argument("cases"); e.add_argument("--baseline"); e.add_argument("--update-baseline", action="store_true")
    e.add_argument("--weak-sandbox", action="store_true")
    e.set_defaults(fn=cmd_eval)

    a = sub.add_parser("acp"); common(a); a.set_defaults(fn=cmd_acp)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
