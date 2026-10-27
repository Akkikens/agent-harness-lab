"""Durable execution: an append-only event journal plus deterministic replay.

Every non-deterministic step (a model call, a tool call) is recorded the
moment it finishes. After a crash, re-running the same run id replays the
recorded steps from the journal without re-executing them, then continues
live from the first step that never finished. This is the same model as a
workflow engine's event history, in ~80 lines on SQLite.

Delivery semantics: a step that was in flight when the process died has no
journal entry, so it runs again on resume. That is at-least-once, so tools
with side effects must be idempotent (write_file is) or use the
idempotency key the context carries.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class NonDeterminismError(RuntimeError):
    """The code asked for a different step than the journal recorded."""


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    name: str
    payload: dict


class SqliteJournal:
    def __init__(self, path: str | Path = ":memory:"):
        # One connection shared across the ACP reader and prompt-worker threads;
        # every access below takes the lock, so check_same_thread can be off.
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            run_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
            name TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY (run_id, seq))""")
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, task TEXT)")
        self.db.commit()

    def events(self, run_id: str) -> list[Event]:
        with self._lock:
            rows = self.db.execute("SELECT seq, kind, name, payload FROM events WHERE run_id=? ORDER BY seq",
                                   (run_id,)).fetchall()
        return [Event(s, k, n, json.loads(p)) for s, k, n, p in rows]

    def append(self, run_id: str, event: Event) -> None:
        with self._lock:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?)",
                            (run_id, event.seq, event.kind, event.name, json.dumps(event.payload)))
            self.db.commit()

    def set_status(self, run_id: str, status: str, task: str | None = None) -> None:
        with self._lock:
            self.db.execute("INSERT INTO runs (run_id, status, task) VALUES (?,?,?) "
                            "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
                            "task=COALESCE(excluded.task, runs.task)", (run_id, status, task))
            self.db.commit()

    def status(self, run_id: str) -> tuple[str, str | None] | None:
        with self._lock:
            row = self.db.execute("SELECT status, task FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return (row[0], row[1]) if row else None


class DurableRun:
    def __init__(self, journal: SqliteJournal, run_id: str):
        self.journal, self.run_id = journal, run_id
        self._recorded = journal.events(run_id)
        self._cursor = 0
        self.replayed = 0
        self.executed = 0

    @property
    def idempotency_key(self) -> str:
        return f"{self.run_id}:{self._cursor}"

    def step(self, kind: str, name: str, fn: Callable[[], dict]) -> dict:
        if self._cursor < len(self._recorded):
            ev = self._recorded[self._cursor]
            if (ev.kind, ev.name) != (kind, name):
                raise NonDeterminismError(
                    f"step {self._cursor}: journal has {ev.kind}/{ev.name}, code asked for {kind}/{name}")
            self._cursor += 1
            self.replayed += 1
            return ev.payload
        payload = fn()
        self.journal.append(self.run_id, Event(self._cursor, kind, name, payload))
        self._cursor += 1
        self.executed += 1
        return payload
