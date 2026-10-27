"""Sandboxed code execution.

DockerSandbox is the real boundary: no network, read-only root, all Linux
capabilities dropped, non-root user, memory/CPU/pid limits, tiny noexec tmpfs,
and the workspace mounted read-only unless the call was granted fs.write.

SubprocessSandbox is a development fallback. It scrubs the environment and
sets rlimits but it is NOT isolation (it has network and host filesystem
access), so it is never selected unless the caller opts in explicitly.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Sandbox(Protocol):
    def run_python(self, code: str, workdir: Path, *, writable: bool, timeout: float) -> SandboxResult: ...


def _text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else data


class DockerSandbox:
    def __init__(self, image: str = "python:3.12-alpine", memory: str = "128m", cpus: str = "0.5", pids: int = 64):
        self.image, self.memory, self.cpus, self.pids = image, memory, cpus, pids

    @staticmethod
    def _user() -> str:
        # Run as the caller's own uid so a granted workspace write works on Linux bind
        # mounts and created files stay owned by the caller; never root.
        uid, gid = os.getuid(), os.getgid()
        return f"{uid}:{gid}" if uid != 0 else "65534:65534"

    def command(self, name: str, workdir: Path, writable: bool) -> list[str]:
        mode = "rw" if writable else "ro"
        return [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", self._user(),
            "--pids-limit", str(self.pids),
            "--memory", self.memory,
            "--memory-swap", self.memory,  # equal to --memory: no swap, so the cap is real
            "--cpus", self.cpus,
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "-v", f"{workdir.resolve()}:/work:{mode}",
            "-w", "/work",
            self.image, "python", "-I", "-",
        ]

    @staticmethod
    def daemon_up() -> bool:
        try:
            return subprocess.run(["docker", "info"], capture_output=True, timeout=8).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def ensure_image(self) -> None:
        if subprocess.run(["docker", "image", "inspect", self.image], capture_output=True).returncode != 0:
            subprocess.run(["docker", "pull", self.image], check=True, capture_output=True)

    def run_python(self, code: str, workdir: Path, *, writable: bool, timeout: float) -> SandboxResult:
        name = f"harness-{uuid.uuid4().hex[:12]}"
        try:
            p = subprocess.run(self.command(name, workdir, writable), input=code,
                               capture_output=True, text=True, timeout=timeout)
            return SandboxResult(p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return SandboxResult(124, _text(e.stdout), _text(e.stderr), timed_out=True)


class SubprocessSandbox:
    """Weak dev fallback. Do not treat as a security boundary."""

    def run_python(self, code: str, workdir: Path, *, writable: bool, timeout: float) -> SandboxResult:
        def limits() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

        scratch = tempfile.mkdtemp(prefix="harness-")
        try:
            p = subprocess.run([sys.executable, "-I", "-"], input=code, capture_output=True, text=True,
                               timeout=timeout, cwd=workdir if writable else scratch, env={},
                               preexec_fn=limits)
            return SandboxResult(p.returncode, p.stdout, p.stderr)
        except subprocess.TimeoutExpired as e:
            return SandboxResult(124, _text(e.stdout), _text(e.stderr), timed_out=True)


def pick_sandbox(allow_weak: bool = False) -> Sandbox:
    if DockerSandbox.daemon_up():
        box = DockerSandbox()
        box.ensure_image()
        return box
    if allow_weak:
        return SubprocessSandbox()
    raise RuntimeError("Docker is not available; start it, or pass allow_weak=True for the non-isolating fallback")
