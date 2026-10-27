# agent-harness-lab

A small, dependency-free agent harness (Python, stdlib only) built to work through the parts of an agent platform that are easy to hand-wave: **safe tool execution, durable runs, honest evals, and a standard editor protocol.** Built in a day with Claude Code as a learning/portfolio project. It is not a production system and has no users.

| Concern | What's here | Where |
|---|---|---|
| Sandboxed code execution | Docker sandbox: no network, read-only root, `--cap-drop ALL`, `no-new-privileges`, non-root user, memory/CPU/pid limits, noexec tmpfs, workspace mounted read-only unless the call was granted `fs.write`; timeout kills the container | `harness/sandbox.py` |
| Capability-based access control | Tools declare the exact capabilities a call needs (`fs.read:<path>`, `fs.write:<path>`, `exec:python`, `net:<host>`); checked against the session's grant before the tool runs. Paths are realpath-resolved (`..` and symlink escapes denied). A set can only be *attenuated*, never widened | `harness/capabilities.py` |
| Credential isolation | Secrets live only in the host process; a tool gets one only via an explicit grant; the sandbox environment is always empty; tool output is redacted and host paths are hidden before the model sees it | `harness/secrets.py`, `harness/tools.py` |
| Durable execution | Every model call and tool call is journaled to SQLite as it finishes. Re-running a run id replays finished steps from the journal (no model re-request, no tool re-run) and continues live from the first unfinished step. Divergent code paths raise `NonDeterminismError` | `harness/journal.py` |
| Offline replay + CI gating | `RecordingModel` writes a cassette; `ReplayModel` serves it with no network and **fails on drift** (changed prompt, tool schema, tool output). `harness eval` runs cases and exits non-zero if a case that passed in the stored baseline regresses | `harness/models.py`, `harness/evalrun.py` |
| ACP (Agent Client Protocol) | Stdio JSON-RPC server: `initialize`, `session/new`, `session/prompt`, `session/update` (message/tool_call/tool_call_update), `session/request_permission` before side-effecting tools, `session/cancel` | `harness/acp.py` |
| Design | Small interfaces (`ModelBackend`, `Tool`, `Sandbox`, `AgentObserver`), one responsibility per module, the loop composed from injected parts (no inheritance hierarchy). Swap a scripted model for a recorder for a real API without touching the loop | `harness/types.py`, `harness/agent.py` |

## Try it

```bash
uv venv && uv pip install -e '.[dev]'
pytest -q                                   # unit + ACP subprocess tests; Docker tests run if a daemon is up
harness eval cases --baseline eval-baseline.json   # offline replay evals, exit 1 on regression

# crash-safe run, with a scripted model (no API key)
harness run "make hello.txt" --run-id demo --workspace /tmp/ws --script demo-turns.json
harness resume demo --workspace /tmp/ws --script demo-turns.json   # replays journaled steps, continues

# editor integration (point any ACP client at this command)
harness acp --script demo-turns.json
```

With `ANTHROPIC_API_KEY` set and no `--script`, `run` uses the Anthropic Messages API backend.

## How the guarantees are tested

- **Capabilities** (`tests/test_capabilities.py`): `..`, absolute paths, symlinks and sibling-prefix directories are denied; a write grant is not implied by a read grant; denial happens *before* the sandbox is touched; secrets reach the tool but never the model; attenuation can't widen.
- **Durable execution** (`tests/test_durable.py`): kill the run inside a tool (`BaseException`, like a SIGKILL), then resume with a fresh model and journal connection: finished steps are replayed (the earlier write ran once, not twice), the model isn't asked for earlier turns again, and the in-flight step runs again.
- **Replay/eval** (`tests/test_replay.py`): a reworded tool description or changed task is reported as drift; the gate fails on a regression and on a vanished case.
- **ACP** (`tests/test_acp.py`): drives the real `harness acp` subprocess over stdio: handshake, the exact update sequence for a tool call, permission granted vs. rejected, cancel, unknown session/method errors.
- **Sandbox** (`tests/test_sandbox.py`, marker `docker`): network blocked, host env never visible, workspace read-only by default, writable only when granted, no write outside the mount, timeout kills the container, memory cap enforced. These need a Docker daemon; they **skip** (not fake-pass) without one, and CI fails the build if they skip.

## Limits, stated plainly

- **At-least-once for the in-flight step.** A step running when the process died has no journal entry, so it runs again on resume. Tools with side effects should be idempotent (`write_file` is) or use the idempotency key. This is not exactly-once and it is not a general workflow engine (no timers, signals, child workflows, or versioning).
- **`SubprocessSandbox` is not isolation.** It exists for development when Docker is down and is only used with `--weak-sandbox`; otherwise the harness refuses to run code.
- **Four of the five cassettes are scripted**, not recorded from a live model (`scripts/make_sample_cases.py` says so). `cases/live-primes.json` is the exception: its cassette was recorded from a real `claude-fable-5-1` run through `AnthropicModel` (`harness run ... --record`) and replays offline in CI. That one run is the extent of live-API testing here; the backend has no automated live test.
- **ACP is the baseline subset only**: no `authenticate`, `session/load`, MCP passthrough, client fs/terminal methods, plans or modes. Written against the published spec at agentclientprotocol.com; tool-call updates send both `name` and `title` because the docs and older clients differ.
- Path checks resolve once and use the resolved path for the I/O, but it is a lab: don't treat the capability layer as audited.
