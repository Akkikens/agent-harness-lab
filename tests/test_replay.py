import json

import pytest

from harness.agent import Agent
from harness.capabilities import Capability, CapabilitySet
from harness.evalrun import CaseResult, default_registry, gate, run_case
from harness.journal import DurableRun, SqliteJournal
from harness.models import CassetteDrift, RecordingModel, ReplayModel, ScriptedModel
from harness.tools import ReadFile, ToolRegistry, WriteFile
from harness.types import ModelTurn, ToolCall

TURNS = [
    ModelTurn("saving", [ToolCall("c1", "write_file", {"path": "out.txt", "content": "42"})]),
    ModelTurn("saved out.txt"),
]


def agent(ws, model, registry=None):
    caps = CapabilitySet([Capability("fs.write", str(ws))])
    return Agent(model, registry or ToolRegistry([ReadFile(), WriteFile()]), caps, workspace=ws, sandbox=None)


def record(ws, cassette):
    inner = ScriptedModel(TURNS)
    agent(ws, RecordingModel(inner, cassette)).run("save 42", DurableRun(SqliteJournal(), "rec"))
    return inner


def test_replay_reproduces_the_run_with_no_inner_model(ws, tmp_path):
    cassette = tmp_path / "c.jsonl"
    record(ws, cassette)
    (ws / "out.txt").unlink()
    res = agent(ws, ReplayModel(cassette)).run("save 42", DurableRun(SqliteJournal(), "rep"))
    assert res.stop_reason == "end_turn" and res.final_text == "saved out.txt"
    assert (ws / "out.txt").read_text() == "42", "tools still ran for real during replay"


def test_changed_tool_description_is_reported_as_drift(ws, tmp_path):
    cassette = tmp_path / "c.jsonl"
    record(ws, cassette)

    class Reworded(ReadFile):
        description = "Read a file. (reworded)"

    with pytest.raises(CassetteDrift, match="turn 0"):
        agent(ws, ReplayModel(cassette), ToolRegistry([Reworded(), WriteFile()])).run(
            "save 42", DurableRun(SqliteJournal(), "d"))


def test_changed_task_prompt_is_reported_as_drift(ws, tmp_path):
    cassette = tmp_path / "c.jsonl"
    record(ws, cassette)
    with pytest.raises(CassetteDrift):
        agent(ws, ReplayModel(cassette)).run("save forty-two", DurableRun(SqliteJournal(), "d"))


def test_extra_model_call_is_reported(ws, tmp_path):
    cassette = tmp_path / "c.jsonl"
    cassette.write_text(json.dumps({"key": "x", "turn": TURNS[1].to_dict()}) + "\n")
    m = ReplayModel(cassette)
    with pytest.raises(CassetteDrift):
        m.complete([], [])  # key mismatch
    m._i = 1
    with pytest.raises(CassetteDrift, match="exhausted"):
        m.complete([], [])


def case_files(tmp_path, ws_expect):
    cassette = tmp_path / "save.jsonl"
    (tmp_path / "w").mkdir(exist_ok=True)
    ws = (tmp_path / "w").resolve()
    (ws / "out.txt").unlink(missing_ok=True)
    # record with the same registry the eval runner uses, or the request keys won't match
    agent(ws, RecordingModel(ScriptedModel(TURNS), cassette), default_registry()).run(
        "save 42", DurableRun(SqliteJournal(), "rec"))
    case = {"id": "save-42", "task": "save 42", "cassette": "save.jsonl", "capabilities": ["fs.write:{WS}"],
            "expect": ws_expect}
    return case


def test_case_passes_then_fails_when_expectation_breaks(tmp_path):
    ok = run_case(case_files(tmp_path, {"files": {"out.txt": "42"}, "tools": ["write_file"], "denied": 0}), tmp_path, None)
    assert ok.status == "pass", ok.reason
    bad = run_case(case_files(tmp_path, {"files": {"out.txt": "43"}}), tmp_path, None)
    assert bad.status == "fail" and "out.txt" in bad.reason


def test_case_that_loses_a_capability_fails_with_denial(tmp_path):
    case = case_files(tmp_path, {"files": {"out.txt": "42"}})
    case["capabilities"] = ["fs.read:{WS}"]  # policy regression: write no longer granted
    r = run_case(case, tmp_path, None)
    assert r.status == "fail"


def test_gate_flags_regressions_against_baseline_and_ignores_known_failures():
    baseline = {"passed": ["a", "b"]}
    ok, notes = gate([CaseResult("a", "pass"), CaseResult("b", "pass"), CaseResult("c", "fail", "was never green")], baseline)
    assert ok
    ok, notes = gate([CaseResult("a", "pass"), CaseResult("b", "fail", "drift")], baseline)
    assert not ok and any("REGRESSION b" in n for n in notes)
    ok, _ = gate([CaseResult("a", "pass")], baseline)
    assert not ok, "a vanished case is a regression"


def test_gate_without_baseline_fails_on_any_failure():
    assert gate([CaseResult("a", "fail", "x")], None)[0] is False
    assert gate([CaseResult("a", "pass"), CaseResult("b", "skip", "no docker")], None)[0] is True
