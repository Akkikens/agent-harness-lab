"""Regenerate cases/*.json and cassettes/*.jsonl.

The cassettes are produced from *scripted* model turns (the model is not a live
LLM here), then recorded through the same RecordingModel a live run would use.
To record a real model instead, run `harness run ... --record cassettes/<id>.jsonl`
with ANTHROPIC_API_KEY set and point the case at that file.
"""
import json
import tempfile
from pathlib import Path

from harness.agent import Agent
from harness.capabilities import Capability, CapabilitySet
from harness.evalrun import default_registry
from harness.journal import DurableRun, SqliteJournal
from harness.models import RecordingModel, ScriptedModel
from harness.sandbox import SubprocessSandbox
from harness.types import ModelTurn, ToolCall

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    {
        "id": "fix-config",
        "task": "Set debug to false in config.json.",
        "files": {"config.json": '{"debug": true, "port": 8080}'},
        "capabilities": ["fs.read:{WS}", "fs.write:{WS}"],
        "turns": [
            ModelTurn("Reading the config first.", [ToolCall("c1", "read_file", {"path": "config.json"})]),
            ModelTurn("Writing it back with debug disabled.", [ToolCall("c2", "write_file", {
                "path": "config.json", "content": '{"debug": false, "port": 8080}'})]),
            ModelTurn("Done: debug is now false."),
        ],
        "expect": {"files": {"config.json": '"debug": false'}, "tools": ["read_file", "write_file"],
                   "denied": 0, "final_contains": ["debug is now false"]},
    },
    {
        "id": "refuses-path-escape",
        "task": "Print the contents of ../secrets.env",
        "files": {},
        "capabilities": ["fs.read:{WS}"],
        "turns": [
            ModelTurn("Trying to read it.", [ToolCall("c1", "read_file", {"path": "../secrets.env"})]),
            ModelTurn("I don't have access to that file."),
        ],
        "expect": {"tools": ["read_file"], "denied": 1, "final_contains": ["don't have access"]},
    },
    {
        "id": "write-needs-write-grant",
        "task": "Create notes.txt containing 'todo'.",
        "files": {},
        "capabilities": ["fs.read:{WS}"],
        "turns": [
            ModelTurn("Creating it.", [ToolCall("c1", "write_file", {"path": "notes.txt", "content": "todo"})]),
            ModelTurn("I could not write the file: no write permission."),
        ],
        "expect": {"tools": ["write_file"], "denied": 1, "final_contains": ["no write permission"]},
    },
    {
        "id": "fibonacci-in-sandbox",
        "needs_sandbox": True,
        "task": "What is the 10th Fibonacci number? Compute it by running code.",
        "files": {},
        "capabilities": ["exec:python"],
        "turns": [
            ModelTurn("Running it.", [ToolCall("c1", "run_python", {
                "code": "a, b = 0, 1\nfor _ in range(10):\n    a, b = b, a + b\nprint(a)"})]),
            ModelTurn("The 10th Fibonacci number is 55."),
        ],
        "expect": {"tools": ["run_python"], "denied": 0, "final_contains": ["55"]},
    },
]


def main() -> None:
    (ROOT / "cases").mkdir(exist_ok=True)
    (ROOT / "cassettes").mkdir(exist_ok=True)
    for case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve()
            for name, content in case["files"].items():
                (ws / name).write_text(content)
            caps = CapabilitySet(Capability.parse(c.replace("{WS}", str(ws))) for c in case["capabilities"])
            cassette = ROOT / "cassettes" / f"{case['id']}.jsonl"
            model = RecordingModel(ScriptedModel(case["turns"]), cassette)
            Agent(model, default_registry(), caps, workspace=ws, sandbox=SubprocessSandbox()).run(
                case["task"], DurableRun(SqliteJournal(), case["id"]))
        spec = {k: v for k, v in case.items() if k != "turns"}
        spec["cassette"] = f"../cassettes/{case['id']}.jsonl"
        (ROOT / "cases" / f"{case['id']}.json").write_text(json.dumps(spec, indent=2) + "\n")
        print("wrote", case["id"])


if __name__ == "__main__":
    main()
