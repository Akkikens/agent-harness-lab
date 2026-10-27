import threading

import pytest

from harness.agent import Agent
from harness.capabilities import Capability, CapabilitySet
from harness.journal import DurableRun, NonDeterminismError, SqliteJournal
from harness.models import ScriptedModel
from harness.secrets import SecretBroker
from harness.tools import ToolRegistry, WriteFile
from harness.types import AgentObserver, ModelTurn, ToolCall


class SimulatedCrash(BaseException):
    """BaseException so the registry's `except Exception` can't swallow it, like a SIGKILL."""


class CountingWrite(WriteFile):
    runs = 0

    def run(self, args, ctx):
        type(self).runs += 1
        return super().run(args, ctx)


class CrashOnce:
    name, description, kind, side_effects = "crash_once", "test tool", "other", False
    schema = {"type": "object", "properties": {}}

    def __init__(self):
        self.armed, self.runs = True, 0

    def required(self, args, ctx):
        return []

    def run(self, args, ctx):
        self.runs += 1
        if self.armed:
            raise SimulatedCrash()
        return "survived"


def script():
    return [
        ModelTurn("writing", [ToolCall("c1", "write_file", {"path": "a.txt", "content": "hello"})]),
        ModelTurn("now the risky step", [ToolCall("c2", "crash_once", {})]),
        ModelTurn("all done"),
    ]


def make_agent(ws, model, crash, **kw):
    caps = CapabilitySet([Capability("fs.write", str(ws))])
    return Agent(model, ToolRegistry([CountingWrite(), crash]), caps, workspace=ws, sandbox=None, broker=SecretBroker(), **kw)


def test_crash_then_resume_does_not_redo_finished_steps(ws, tmp_path):
    CountingWrite.runs = 0
    db = tmp_path / "journal.db"
    crash = CrashOnce()

    # run 1 dies inside the crash_once tool
    with pytest.raises(SimulatedCrash):
        make_agent(ws, ScriptedModel(script()), crash).run("task", DurableRun(SqliteJournal(db), "r1"))
    assert CountingWrite.runs == 1
    assert [e.name for e in SqliteJournal(db).events("r1")] == ["turn:0", "write_file:c1", "turn:1"]

    # run 2: new process state (fresh model, fresh journal connection), same run id
    crash.armed = False
    model2 = ScriptedModel(script())
    run2 = DurableRun(SqliteJournal(db), "r1")
    result = make_agent(ws, model2, crash).run("task", run2)

    assert result.stop_reason == "end_turn" and result.final_text == "all done"
    assert CountingWrite.runs == 1, "the finished write must be replayed from the journal, not re-run"
    assert run2.replayed == 3, "model turn 0, the write, and model turn 1 come from the journal"
    assert run2.executed == 2, "only the in-flight tool call and the final model turn run live"
    assert model2.calls == 1, "earlier model turns are not re-requested"
    assert crash.runs == 2, "the in-flight step is at-least-once: it ran again on resume"
    assert (ws / "a.txt").read_text() == "hello"


def test_resume_of_a_finished_run_executes_nothing(ws, tmp_path):
    CountingWrite.runs = 0
    j = SqliteJournal(tmp_path / "j.db")
    crash = CrashOnce()
    crash.armed = False
    make_agent(ws, ScriptedModel(script()), crash).run("t", DurableRun(j, "r"))
    run = DurableRun(j, "r")
    make_agent(ws, ScriptedModel(script()), crash).run("t", run)
    assert (run.executed, run.replayed) == (0, 5)
    assert CountingWrite.runs == 1


def test_diverging_code_path_is_detected():
    j = SqliteJournal()
    DurableRun(j, "r").step("model", "turn:0", lambda: {"x": 1})
    with pytest.raises(NonDeterminismError, match="turn:0.*turn:1"):
        DurableRun(j, "r").step("model", "turn:1", lambda: {"x": 1})


def test_runs_are_isolated_by_run_id():
    j = SqliteJournal()
    DurableRun(j, "a").step("model", "turn:0", lambda: {"v": "a"})
    r = DurableRun(j, "b")
    assert r.step("model", "turn:0", lambda: {"v": "b"}) == {"v": "b"} and r.executed == 1


def test_cancel_stops_between_steps(ws):
    cancel = threading.Event()

    class CancelAfterTool(AgentObserver):
        def on_tool_end(self, call, result):
            cancel.set()

    crash = CrashOnce()
    crash.armed = False
    result = make_agent(ws, ScriptedModel(script()), crash, observer=CancelAfterTool(), cancel=cancel) \
        .run("t", DurableRun(SqliteJournal(), "r"))
    assert result.stop_reason == "cancelled" and result.tool_calls == ["write_file"]
